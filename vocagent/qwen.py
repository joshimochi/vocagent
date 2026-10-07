"""Qwen3-VL generation and selection, plus Qwen3Guard text/audio safety.

Qwen3Guard uses its own tokenizer's native response-moderation template:
https://huggingface.co/Qwen/Qwen3Guard-Gen-8B
Only a complete, valid Safe verdict passes; errors stop the workflow.
"""

import json
import math
from pathlib import Path
import re
from typing import Protocol, Sequence

from .models import SafetyResult
from .prompts.generation import H3_REPAIR_PROMPT, outline_prompt, storyboard_prompt
from .prompts.quality import (
    QUALITY_PROMPT, VIDEO_CANDIDATE_TEMPLATE, VIDEO_FRAME_TEMPLATE, VIDEO_QUALITY_PROMPT,
)
from .prompts.safety import AUDIO_SAFETY_PROMPT, SAFETY_FEEDBACK_PROMPT, TEXT_SAFETY_PROMPT
from .providers import (
    AudioSafetyModel, LLMGenerator, TextQualityAgent, TextSafetyAgent,
    VideoQualityModel,
)
from .video_frames import sample_video_frames


class QwenError(RuntimeError):
    """A local model failed or returned an unusable response."""


class ChatClient(Protocol):
    """Chat interface for the on-demand local GPU runtime."""

    def complete(
        self, messages: Sequence[dict[str, object]], *, max_tokens: int,
        temperature: float, non_thinking: bool = False,
    ) -> str: ...


def _validate_h3_storyboard(result: str) -> None:
    """Reject malformed native prompts before expensive video inference."""
    def invalid(reason: str) -> None:
        preview = result[:1024]
        suffix = " [truncated]" if len(result) > 1024 else ""
        raise QwenError(
            "H3 storyboard must use its native fields, speaker ID and tagged speech; "
            f"{reason}. Response preview: {preview!r}{suffix}"
        )

    required_fields = (
        "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music",
    )
    missing = [name for name in required_fields if f"{name}:" not in result]
    if missing:
        invalid(f"missing required field(s): {', '.join(missing)}")
    fields = re.fullmatch(
        r"integrated_multimodal_description:\s*(.+?)\s*"
        r"overall_soundscape:\s*(.+?)\s*non_diegetic_music:\s*(.+)",
        result, re.DOTALL,
    )
    if fields is None:
        invalid("required fields must appear in order with nonempty contents")
    for name, content in zip(required_fields, fields.groups()):
        if not content.strip():
            invalid(f"required field {name} is empty")
    description = fields[1]
    if any(re.search(r"</?d\b", content, re.IGNORECASE) for content in fields.groups()[1:]):
        invalid("all dialogue must be inside integrated_multimodal_description")
    for marker in ("[Shot 1]", "(S1)"):
        if marker not in description:
            invalid(f"integrated_multimodal_description is missing {marker}")
    dialogue = list(re.finditer(r"<d>(.*?)</d>", description, re.DOTALL))
    if not dialogue:
        invalid("integrated_multimodal_description is missing tagged speech")
    for index, block in enumerate(dialogue, 1):
        speech = re.fullmatch(r"\s*\[([^]\r\n]+)\]\s*([^<>]+)", block[1], re.DOTALL)
        if speech is None or not speech[1].strip() or not speech[2].strip():
            invalid(f"dialogue block {index} must contain a nonempty language tag and speech")
    outside_dialogue = re.sub(r"<d>.*?</d>", "", description, flags=re.DOTALL)
    if re.search(r"</?d\b", outside_dialogue, re.IGNORECASE):
        invalid("integrated_multimodal_description contains an unmatched or malformed dialogue tag")


class QwenGenerator(LLMGenerator):
    def __init__(self, client: ChatClient, *, storyboard_seconds: float):
        if (
            isinstance(storyboard_seconds, bool) or not isinstance(storyboard_seconds, (int, float))
            or not math.isfinite(storyboard_seconds) or storyboard_seconds <= 0
        ):
            raise ValueError("Storyboard duration must be a finite positive number of seconds.")
        self.client = client
        self.storyboard_seconds = storyboard_seconds

    def generate_outline(self, vocabulary: Sequence[str], feedback: str | None) -> str:
        prompt = outline_prompt(self.storyboard_seconds)
        inputs = {"vocabulary": list(vocabulary), "duration_seconds": self.storyboard_seconds}
        return self._generate(prompt, inputs, feedback, 1024)

    def generate_storyboard(
        self, vocabulary: Sequence[str], outline: str, feedback: str | None,
    ) -> str:
        inputs = {
            "vocabulary": list(vocabulary), "outline": outline,
            "duration_seconds": self.storyboard_seconds,
        }
        prompt = storyboard_prompt(self.storyboard_seconds)
        result = self._generate(prompt, inputs, feedback, 2048)
        # Missing speech tags caused H3 to improvise unrelated dialogue in a
        # real run. Permit one format repair, then reject any unusable prompt
        # before expensive video inference. Backend errors are never retried.
        try:
            _validate_h3_storyboard(result)
        except QwenError as exc:
            repair_inputs = {
                **inputs, "previous_storyboard": result, "format_error": str(exc),
            }
            repair_prompt = prompt + H3_REPAIR_PROMPT
            result = self._generate(repair_prompt, repair_inputs, feedback, 2048)
            _validate_h3_storyboard(result)
        return result

    def _generate(self, prompt: str, inputs: dict, feedback: str | None, max_tokens: int) -> str:
        if feedback:
            inputs["safety_feedback"] = feedback
            prompt += SAFETY_FEEDBACK_PROMPT
        return self.client.complete(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)}],
            max_tokens=max_tokens, temperature=0.7, non_thinking=True,
        )


_CATEGORIES = frozenset({
    "Violent", "Non-violent Illegal Acts", "Sexual Content or Sexual Acts", "PII",
    "Suicide & Self-Harm", "Unethical Acts", "Politically Sensitive Topics", "Copyright Violation",
})
_GUARD_OUTPUT = re.compile(
    r"Safety: (Safe|Unsafe|Controversial)\s*\n"
    r"Categories: ([^\r\n]+)\s*\n"
    r"Refusal: (Yes|No)"
)


def _validate_stage(stage: str) -> None:
    if stage not in ("outline", "storyboard"):
        raise ValueError("Qwen text stage must be outline or storyboard.")


class QwenGuard(TextSafetyAgent):
    def __init__(self, client: ChatClient):
        self.client = client

    def check(self, stage: str, text: str) -> SafetyResult:
        _validate_stage(stage)
        return self._check_response(TEXT_SAFETY_PROMPT.format(stage=stage), text)

    def _check_response(self, prompt: str, text: str) -> SafetyResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Qwen3Guard requires nonempty candidate text.")
        # The model's native template classifies the final assistant response.
        # A generic system safety prompt would replace its trained task format.
        content = self.client.complete(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": text}],
            max_tokens=128, temperature=0,
        )
        match = _GUARD_OUTPUT.fullmatch(content)
        if match is None:
            raise QwenError("Qwen3Guard returned an invalid Safety/Categories/Refusal verdict.")
        label, category_text, _refusal = match.groups()
        categories = [item.strip() for item in category_text.split(",")]
        if (not categories or len(set(categories)) != len(categories)
                or any(item not in _CATEGORIES and item != "None" for item in categories)
                or ("None" in categories and len(categories) != 1)
                or (label == "Safe") != (categories == ["None"])):
            raise QwenError("Qwen3Guard returned invalid or inconsistent safety categories.")
        if label == "Safe":
            return SafetyResult(True)
        return SafetyResult(False, f"Qwen3Guard: {label}; categories: {', '.join(categories)}.")


class QwenAudioSafety(AudioSafetyModel):
    """Moderate the complete spoken transcript with the native response template."""

    def __init__(self, client: ChatClient):
        self.client = client
        self._guard = QwenGuard(client)

    def check(self, transcript: str) -> SafetyResult:
        return self._guard._check_response(
            AUDIO_SAFETY_PROMPT,
            transcript,
        )


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


class QwenQuality(TextQualityAgent):
    def __init__(self, client: ChatClient):
        self.client = client

    def choose(self, stage: str, vocabulary: Sequence[str], candidates: Sequence[str]) -> int:
        _validate_stage(stage)
        if (isinstance(candidates, str) or not candidates
                or any(not isinstance(item, str) or not item.strip() for item in candidates)):
            raise ValueError("Qwen quality selection requires nonempty text candidates.")
        content = self.client.complete(
            [{"role": "system", "content": QUALITY_PROMPT},
             {"role": "user", "content": json.dumps({
                 "stage": stage, "vocabulary": list(vocabulary),
                 "candidates": [{"index": index, "text": text}
                                for index, text in enumerate(candidates)],
             }, ensure_ascii=False)}],
            max_tokens=128, temperature=0, non_thinking=True,
        )
        return _selection_index(content, len(candidates))


def _selection_index(content: str, candidate_count: int) -> int:
    try:
        decision = json.loads(content, object_pairs_hook=_unique_object)
    except ValueError as exc:
        raise QwenError("Qwen quality judge must return JSON with one integer index.") from exc
    if (not isinstance(decision, dict) or set(decision) != {"index"}
            or type(decision["index"]) is not int
            or not 0 <= decision["index"] < candidate_count):
        raise QwenError("Qwen quality judge returned an invalid or out-of-range index.")
    return decision["index"]


class QwenVideoQuality(VideoQualityModel):
    """Judge actual pixels using bounded, timestamped frame sequences.

    Sampling provides visual evidence, but does not inspect every video frame.
    """

    def __init__(
        self, client: ChatClient, *, max_frames: int = 16, max_side: int = 768,
        max_total_frames: int = 32,
    ):
        if type(max_frames) is not int or not 1 <= max_frames <= 64:
            raise ValueError("Video quality max_frames must be an integer from 1 to 64.")
        if type(max_side) is not int or not 32 <= max_side <= 2048:
            raise ValueError("Video quality max_side must be an integer from 32 to 2048.")
        if type(max_total_frames) is not int or not max_frames <= max_total_frames <= 64:
            raise ValueError("Video quality total frame limit must fit one candidate and be at most 64.")
        self.client = client
        self.max_frames = max_frames
        self.max_side = max_side
        self.max_total_frames = max_total_frames

    def validate_candidate_count(self, count: int) -> None:
        if type(count) is not int or count < 1:
            raise ValueError("Video quality candidate count must be a positive integer.")
        if count * self.max_frames > self.max_total_frames:
            raise ValueError(
                f"Qwen video selection supports at most {self.max_total_frames // self.max_frames} "
                f"candidate slots with {self.max_frames} frames each "
                f"({self.max_total_frames} total sampled frames)."
            )

    def choose(self, storyboard: str, candidates: Sequence[Path]) -> int:
        if not isinstance(storyboard, str) or not storyboard.strip():
            raise ValueError("Qwen video selection requires a nonempty storyboard.")
        if (isinstance(candidates, (str, bytes, Path)) or not candidates
                or any(not isinstance(path, (str, Path)) for path in candidates)):
            raise ValueError("Qwen video selection requires nonempty video candidates.")
        # The runtime bounds image tokens per frame. Bound the total frame count
        # too so increasing candidate slots cannot silently exhaust GPU memory.
        self.validate_candidate_count(len(candidates))
        content: list[dict[str, object]] = [
            {"type": "text", "text": json.dumps({"storyboard": storyboard}, ensure_ascii=False)}
        ]
        for index, path in enumerate(candidates):
            frames = sample_video_frames(
                Path(path), max_frames=self.max_frames, max_side=self.max_side,
            )
            content.append({"type": "text", "text": VIDEO_CANDIDATE_TEMPLATE.format(
                index=index, frame_count=len(frames),
            )})
            for frame in frames:
                content.extend([
                    {"type": "text", "text": VIDEO_FRAME_TEMPLATE.format(
                        index=index, timestamp=frame.timestamp,
                    )},
                    {"type": "image", "image": frame.jpeg},
                ])
        result = self.client.complete(
            [{"role": "system", "content": VIDEO_QUALITY_PROMPT},
             {"role": "user", "content": content}],
            max_tokens=128, temperature=0, non_thinking=True,
        )
        return _selection_index(result, len(candidates))
