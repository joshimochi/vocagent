"""Lazy, accelerator-only speech recognition and sampled-frame safety providers.

The shared runtime owns model residency and serializes inference. Qwen's speech
wrapper keeps its small ASR and forced-aligner pair together; its official long
audio chunking merges timestamps back into the original audio time base.
"""

from io import BytesIO
import math
from numbers import Real
from pathlib import Path
from typing import Any, Sequence
import unicodedata

from .models import SafetyResult, TranscriptSegment
from .providers import SpeechToText, VideoSafetyModel
from .subtitles import validate_segments


ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
SHIELD_MODEL = "google/shieldgemma-2-4b-it"
SHIELD_POLICIES = ("dangerous", "sexual", "violence")


class LocalModelError(RuntimeError):
    """A local model or its input did not produce a usable result."""


def _model_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Model ID or directory must be nonempty.")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise LocalModelError(f"{name} must be a finite number.")
    return float(value)


def _text_positions(text: str) -> tuple[str, list[int]]:
    """Map lexical characters back to the unmodified ASR transcript.

    The aligner removes punctuation and splits CJK characters. Keeping source
    positions lets subtitles and transcript safety retain the original text.
    """
    letters: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            if normalized.isalnum():
                letters.append(normalized)
                positions.append(index)
    return "".join(letters), positions


def aligned_segments(
    text: str, items: Sequence[Any], *, duration: float,
    max_cue_seconds: float = 7.0, max_cue_characters: int = 84,
) -> list[TranscriptSegment]:
    """Build readable cues without dropping any recognized words or punctuation."""
    if not isinstance(text, str) or not text.strip() or not items:
        raise LocalModelError("Speech recognition returned no text or aligned timestamps.")
    canonical, positions = _text_positions(text)
    if not canonical:
        raise LocalModelError("Speech recognition returned no alignable words.")
    words: list[tuple[float, float, int]] = []
    offset = 0
    previous_start = -1.0
    for item in items:
        token = getattr(item, "text", None)
        if not isinstance(token, str):
            raise LocalModelError("Forced alignment returned a non-text word.")
        lexical, _ = _text_positions(token)
        if not lexical or not canonical.startswith(lexical, offset):
            raise LocalModelError("Forced alignment does not cover the recognized transcript in order.")
        start = _finite_number(getattr(item, "start_time", None), "Alignment start")
        end = _finite_number(getattr(item, "end_time", None), "Alignment end")
        if start < 0 or end < start or start < previous_start or end > duration + 0.25:
            raise LocalModelError("Forced alignment returned invalid or out-of-range timestamps.")
        words.append((min(start, duration), min(end, duration), positions[offset]))
        previous_start = start
        offset += len(lexical)
    if offset != len(canonical):
        raise LocalModelError("Forced alignment omitted part of the recognized transcript.")

    segments: list[TranscriptSegment] = []
    segment_sources: list[int] = []
    first = 0
    for index in range(len(words)):
        next_position = words[index + 1][2] if index + 1 < len(words) else len(text)
        source_start = 0 if first == 0 else words[first][2]
        finish = (
            index == len(words) - 1
            or words[index][1] - words[first][0] >= max_cue_seconds
            or next_position - source_start >= max_cue_characters
            or (index + 1 < len(words) and words[index + 1][0] - words[index][1] >= 0.8)
        )
        # Some aligned tokens have zero duration. Keep them with adjacent words
        # instead of deleting their text or inventing a timestamp.
        if finish and words[index][1] > words[first][0]:
            segments.append(TranscriptSegment(
                words[first][0], words[index][1], text[source_start:next_position].strip(),
            ))
            segment_sources.append(source_start)
            first = index + 1
    if first < len(words):
        if not segments:
            raise LocalModelError("Forced alignment returned no positive-duration speech.")
        last = segments[-1]
        segments[-1] = TranscriptSegment(
            last.start, max(last.end, words[-1][1]),
            text[segment_sources[-1]:].strip(),
        )
    validate_segments(segments)
    return segments


class Qwen3SpeechToText(SpeechToText):
    """Qwen3-ASR 1.7B and its official 0.6B forced aligner on one accelerator."""

    def __init__(
        self, runtime: Any, model: str = ASR_MODEL, aligner_model: str = ALIGNER_MODEL,
        *, language: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.model = _model_name(model)
        self.aligner_model = _model_name(aligner_model)
        self.language = language

    def _load(self) -> Any:
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise LocalModelError("Qwen speech inference requires the qwen-asr package.") from exc
        options = {
            "dtype": self.runtime.dtype,
            # safetensors cannot deserialize directly onto MUSA. These small
            # models are copied to the accelerator once, before any inference;
            # no offload hooks or CPU-resident model weights are retained.
            "device_map": None,
            "attn_implementation": "eager",
        }
        speech = Qwen3ASRModel.from_pretrained(
            self.model, **options,
            forced_aligner=self.aligner_model,
            forced_aligner_kwargs=dict(options),
            max_inference_batch_size=1,
            max_new_tokens=4096,
        )
        speech.model.to(self.runtime.device).eval()
        speech.forced_aligner.model.to(self.runtime.device).eval()
        if self.runtime.device.startswith("musa:"):
            from .gpu_ops import patch_musa_convolutions
            patch_musa_convolutions(speech.model)
            patch_musa_convolutions(speech.forced_aligner.model)
        speech.device = speech.model.device
        speech.forced_aligner.device = speech.forced_aligner.model.device
        self.runtime.assert_on_accelerator(speech.model)
        self.runtime.assert_on_accelerator(speech.forced_aligner.model)
        return speech

    def transcribe(self, audio_path: Path) -> Sequence[TranscriptSegment]:
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio does not exist: {path}")
        try:
            import soundfile
        except ImportError as exc:
            raise LocalModelError("Reading speech audio requires soundfile.") from exc
        duration = _finite_number(soundfile.info(str(path)).duration, "Audio duration")
        if duration <= 0:
            raise LocalModelError("Speech audio is empty.")
        key = ("qwen3-speech", self.model, self.aligner_model)
        with self.runtime.use(key, self._load) as speech:
            results = speech.transcribe(
                audio=str(path), language=self.language, return_time_stamps=True,
            )
            if not isinstance(results, list) or len(results) != 1:
                raise LocalModelError("Qwen3-ASR must return exactly one transcription.")
            result = results[0]
            stamps = getattr(result, "time_stamps", None)
            items = getattr(stamps, "items", None)
            return aligned_segments(getattr(result, "text", None), items, duration=duration)


def shield_verdict(
    probabilities: Sequence[Sequence[float]], timestamps: Sequence[float], *, threshold: float,
) -> SafetyResult:
    """Validate all policy rows and classify with the YES (violation) column."""
    if len(probabilities) != len(timestamps) * len(SHIELD_POLICIES) or not timestamps:
        raise LocalModelError("ShieldGemma returned an unexpected number of policy scores.")
    violations: list[str] = []
    for index, row in enumerate(probabilities):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise LocalModelError("ShieldGemma scores must contain YES and NO probabilities.")
        yes, no = (_finite_number(value, "ShieldGemma probability") for value in row)
        if not (0 <= yes <= 1 and 0 <= no <= 1) or abs(yes + no - 1) > 0.02:
            raise LocalModelError("ShieldGemma returned invalid probabilities.")
        if yes >= threshold:
            frame_index, policy_index = divmod(index, len(SHIELD_POLICIES))
            violations.append(
                f"{SHIELD_POLICIES[policy_index]} at {timestamps[frame_index]:.2f}s ({yes:.3f})"
            )
    return SafetyResult(not violations, "; ".join(violations))


class ShieldGemma2VideoSafety(VideoSafetyModel):
    """Check fixed-rate samples throughout a video's visual track.

    ShieldGemma 2 is an image classifier. Sampling is explicit and bounded; a
    video exceeding the frame budget raises instead of silently skipping its end.
    Transcript safety is supplied separately by Qwen3Guard.
    """

    def __init__(
        self, runtime: Any, model: str = SHIELD_MODEL, *, sample_fps: float = 2.0,
        max_frames: int = 600, threshold: float = 0.5,
        ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
    ) -> None:
        self.runtime = runtime
        self.model = _model_name(model)
        if isinstance(sample_fps, bool) or not isinstance(sample_fps, Real) or not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("Safety sample FPS must be finite and positive.")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
            raise ValueError("Safety max_frames must be a positive integer.")
        if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Safety threshold must be strictly between zero and one.")
        self.sample_fps = float(sample_fps)
        self.max_frames = max_frames
        self.threshold = float(threshold)
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def _load(self) -> tuple[Any, Any]:
        try:
            from transformers import AutoProcessor, ShieldGemma2ForImageClassification
        except ImportError as exc:
            raise LocalModelError("ShieldGemma 2 requires a compatible transformers installation.") from exc
        model, loading = ShieldGemma2ForImageClassification.from_pretrained(
            self.model, dtype=self.runtime.dtype,
            device_map=None, attn_implementation="eager",
            output_loading_info=True,
        )
        # Transformers 4.57's Shield wrapper delegates tie_weights to Gemma's
        # decoder instead of its causal-LM owner, leaving a random output head.
        # The released checkpoint stores that head only as the tied embeddings.
        model.model.tie_weights()
        tied = bool(getattr(model.config.text_config, "tie_word_embeddings", False))
        if tied and model.model.get_output_embeddings().weight is not model.model.get_input_embeddings().weight:
            raise LocalModelError("ShieldGemma's output head was not tied to its trained embeddings.")
        missing = set(loading.get("missing_keys", ()))
        if tied:
            missing.discard("model.lm_head.weight")
        if missing:
            raise LocalModelError(f"ShieldGemma checkpoint is missing trained weights: {sorted(missing)}")
        model.to(self.runtime.device).eval()
        if self.runtime.device.startswith("musa:"):
            from .gpu_ops import patch_musa_convolutions
            patch_musa_convolutions(model)
        self.runtime.assert_on_accelerator(model)
        return model, AutoProcessor.from_pretrained(self.model)

    def check(self, video_path: Path) -> SafetyResult:
        from .video_frames import sample_video_frames

        frames = sample_video_frames(
            video_path, sample_fps=self.sample_fps, max_frames=self.max_frames,
            max_side=896, ffmpeg=self.ffmpeg, ffprobe=self.ffprobe,
        )
        if not frames:
            raise LocalModelError("No video frames were available for safety review.")
        try:
            from PIL import Image
            import torch
        except ImportError as exc:
            raise LocalModelError("ShieldGemma inference requires Pillow and accelerator PyTorch.") from exc
        with self.runtime.use(("shieldgemma2", self.model), self._load) as resource:
            model, processor = resource
            for frame in frames:
                with Image.open(BytesIO(frame.jpeg)) as source:
                    picture = source.convert("RGB")
                try:
                    inputs = processor(
                        images=[picture], policies=list(SHIELD_POLICIES), return_tensors="pt",
                    ).to(device=self.runtime.device, dtype=self.runtime.dtype)
                finally:
                    picture.close()
                with torch.inference_mode():
                    outputs = model(**inputs)
                scores = outputs.probabilities.detach().float().cpu().tolist()
                verdict = shield_verdict(scores, [frame.timestamp], threshold=self.threshold)
                if not verdict.safe:
                    return verdict
        return SafetyResult(True)
