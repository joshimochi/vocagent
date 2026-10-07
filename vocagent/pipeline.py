"""Generate candidates, check visual and spoken safety, then subtitle the winner."""

import json
import logging
import shutil
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

from .benchmark import Benchmark
from .media import FFmpegMedia
from .models import SafetyResult, StoryboardResult, TranscriptSegment, WorkflowResult
from .providers import Providers
from .selection import select_candidate, validate_safety_result, validate_selection_index
from .subtitles import validate_segments, write_ass
from .vocabulary import normalize_target_word


log = logging.getLogger(__name__)
_CANDIDATE_METADATA: ContextVar[dict] = ContextVar("candidate_metadata", default={})


@dataclass(frozen=True)
class _VideoCandidate:
    video: Path
    audio: Path
    transcript: Path
    segments: tuple[TranscriptSegment, ...]


class Workflow:
    def __init__(
        self,
        providers: Providers,
        media: FFmpegMedia | None = None,
        *,
        outline_candidate_count: int = 2,
        storyboard_candidate_count: int = 2,
        video_candidate_count: int = 2,
        concurrent_candidates: bool = False,
    ):
        for name, count in (
            ("outline_candidate_count", outline_candidate_count),
            ("storyboard_candidate_count", storyboard_candidate_count),
            ("video_candidate_count", video_candidate_count),
        ):
            if type(count) is not int or count < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.providers = providers
        self.media = media
        self.outline_candidate_count = outline_candidate_count
        self.storyboard_candidate_count = storyboard_candidate_count
        self.video_candidate_count = video_candidate_count
        self.concurrent_candidates = concurrent_candidates

    def run(
        self,
        vocabulary: str | Sequence[str],
        output_dir: str | Path,
        *,
        stop_after: Literal["storyboard", "video"] = "video",
    ) -> StoryboardResult | WorkflowResult:
        """Teach one English word, optionally continuing through final video."""
        if stop_after not in ("storyboard", "video"):
            raise ValueError("stop_after must be 'storyboard' or 'video'.")
        words = (normalize_target_word(vocabulary),)
        if stop_after == "video":
            self.providers.video_quality.validate_candidate_count(self.video_candidate_count)
        output = Path(output_dir).resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Output directory must be new or empty to protect existing files.")
        output.mkdir(parents=True, exist_ok=True)
        benchmark = Benchmark(output / "benchmark.json", metadata={
            "stop_after": stop_after,
            "vocabulary_count": len(words),
            "outline_candidate_count": self.outline_candidate_count,
            "storyboard_candidate_count": self.storyboard_candidate_count,
            "video_candidate_count": self.video_candidate_count,
            "concurrent_candidates": self.concurrent_candidates,
        })
        candidate_token = _CANDIDATE_METADATA.set({})
        try:
            result = self._run(words, output, stop_after, benchmark)
        except BaseException as exc:
            try:
                benchmark.finish(error=exc)
            except OSError:
                log.exception("Could not save the failed workflow benchmark")
            raise
        finally:
            _CANDIDATE_METADATA.reset(candidate_token)
        benchmark.finish()
        return result

    def _run(
        self,
        words: tuple[str, ...],
        output: Path,
        stop_after: Literal["storyboard", "video"],
        benchmark: Benchmark,
    ) -> StoryboardResult | WorkflowResult:
        (output / "vocabulary.json").write_text(
            json.dumps(words, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        providers = self.providers
        # Each candidate's callbacks execute in the same thread. Context-local
        # metadata keeps parallel slots and their retries attributed correctly.
        candidate_metadata = _CANDIDATE_METADATA

        def checked_text(text: str) -> str:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Text generators must return nonempty strings.")
            return text

        def save_text_candidate(directory: Path, slot: int, attempt: int, text: str) -> None:
            path = directory / f"candidate-{slot}-attempt-{attempt}.txt"
            path.write_text(text, encoding="utf-8")

        def generate_outline(slot: int, attempt: int, feedback: str | None) -> str:
            candidate_metadata.set({"slot": slot, "attempt": attempt})
            with benchmark.step("outline.generation", **candidate_metadata.get()):
                return checked_text(providers.llm.generate_outline(words, feedback))

        def check_text(stage: str, text: str) -> SafetyResult:
            with benchmark.step(f"{stage}.safety", **candidate_metadata.get()) as step:
                result = validate_safety_result(providers.text_safety.check(stage, text), stage)
                step.outcome("passed" if result.safe else "rejected")
                return result

        def choose_text(stage: str, candidates: Sequence[str]) -> int:
            with benchmark.step(f"{stage}.selection", candidate_count=len(candidates)):
                return validate_selection_index(
                    providers.text_quality.choose(stage, words, candidates), len(candidates), stage,
                )

        outline_candidates_dir = output / "outline_candidates"
        outline_candidates_dir.mkdir()
        outline = select_candidate(
            stage="outline",
            candidate_count=self.outline_candidate_count,
            generate=generate_outline,
            check_safety=lambda text: check_text("outline", text),
            choose_best=lambda candidates: choose_text("outline", candidates),
            pass_feedback=True,
            concurrent_candidates=self.concurrent_candidates,
            on_accept=lambda slot, attempt, text: save_text_candidate(
                outline_candidates_dir, slot, attempt, text
            ),
        )
        outline_path = output / "outline.txt"
        outline_path.write_text(outline, encoding="utf-8")

        storyboard_candidates_dir = output / "storyboard_candidates"
        storyboard_candidates_dir.mkdir()

        def generate_storyboard(slot: int, attempt: int, feedback: str | None) -> str:
            candidate_metadata.set({"slot": slot, "attempt": attempt})
            with benchmark.step("storyboard.generation", **candidate_metadata.get()):
                return checked_text(providers.llm.generate_storyboard(words, outline, feedback))

        storyboard = select_candidate(
            stage="storyboard",
            candidate_count=self.storyboard_candidate_count,
            generate=generate_storyboard,
            check_safety=lambda text: check_text("storyboard", text),
            choose_best=lambda candidates: choose_text("storyboard", candidates),
            pass_feedback=True,
            concurrent_candidates=self.concurrent_candidates,
            on_accept=lambda slot, attempt, text: save_text_candidate(
                storyboard_candidates_dir, slot, attempt, text
            ),
        )
        storyboard_path = output / "storyboard.txt"
        storyboard_path.write_text(storyboard, encoding="utf-8")
        if stop_after == "storyboard":
            return StoryboardResult(
                outline=outline_path, storyboard=storyboard_path,
                benchmarks=output / "benchmark.json",
            )

        media = self.media if self.media is not None else FFmpegMedia()
        candidates_dir = output / "video_candidates"
        candidates_dir.mkdir()

        def generate_video(slot: int, attempt: int, feedback: str | None) -> _VideoCandidate:
            candidate_metadata.set({"slot": slot, "attempt": attempt})
            path = candidates_dir / f"candidate-{slot}-attempt-{attempt}.mp4"
            with benchmark.step("video.generation", **candidate_metadata.get()):
                providers.video.generate(storyboard, path)
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Video generator did not write a nonempty video: {path}")
            audio = path.with_suffix(".wav")
            with benchmark.step("audio.extraction", **candidate_metadata.get()):
                media.extract_audio(path, audio)
            with benchmark.step("speech.transcription", **candidate_metadata.get()) as step:
                segments = tuple(providers.speech_to_text.transcribe(audio))
                validate_segments(segments)
                step.add_work(transcript_segments=len(segments))
            transcript = path.with_suffix(".json")
            transcript.write_text(
                json.dumps([asdict(segment) for segment in segments], ensure_ascii=False,
                           indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            return _VideoCandidate(path, audio, transcript, segments)

        def check_video(candidate: _VideoCandidate) -> SafetyResult:
            with benchmark.step("video.safety", **candidate_metadata.get()) as step:
                visual = validate_safety_result(
                    providers.video_safety.check(candidate.video), "video visuals"
                )
                step.outcome("passed" if visual.safe else "rejected")
            with benchmark.step("audio.safety", **candidate_metadata.get()) as step:
                audio = validate_safety_result(
                    providers.audio_safety.check("\n".join(
                        segment.text for segment in candidate.segments
                    )),
                    "video audio",
                )
                step.outcome("passed" if audio.safe else "rejected")
            if not visual.safe or not audio.safe:
                log.warning(
                    "Rejected %s: visual_safe=%s visual_reason=%r "
                    "transcript_safe=%s transcript_reason=%r",
                    candidate.video.name, visual.safe, visual.reason,
                    audio.safe, audio.reason,
                )
                for path in (candidate.video, candidate.audio, candidate.transcript):
                    path.unlink()
                return visual if not visual.safe else audio
            return SafetyResult(True)

        def choose_video(candidates: Sequence[_VideoCandidate]) -> int:
            with benchmark.step("video.selection", candidate_count=len(candidates)):
                return validate_selection_index(
                    providers.video_quality.choose(
                        storyboard, tuple(candidate.video for candidate in candidates)
                    ),
                    len(candidates), "video",
                )

        selected = select_candidate(
            stage="video",
            candidate_count=self.video_candidate_count,
            generate=generate_video,
            check_safety=check_video,
            choose_best=choose_video,
            pass_feedback=False,
            concurrent_candidates=self.concurrent_candidates,
        )

        audio_path = output / "audio.wav"
        transcript_path = output / "transcript.json"
        shutil.copyfile(selected.audio, audio_path)
        shutil.copyfile(selected.transcript, transcript_path)
        with benchmark.step("subtitles.write") as step:
            subtitles_path = write_ass(selected.segments, output / "subtitles.ass")
            step.add_work(subtitle_cues=len(selected.segments))
        final_path = output / "final.mp4"
        with benchmark.step("subtitles.burn"):
            media.burn_subtitles(selected.video, subtitles_path, final_path)
        return WorkflowResult(
            outline=outline_path,
            storyboard=storyboard_path,
            selected_video=selected.video,
            audio=audio_path,
            transcript=transcript_path,
            subtitles=subtitles_path,
            final_video=final_path,
            benchmarks=output / "benchmark.json",
        )
