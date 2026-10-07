"""Generate candidates, check visual and spoken safety, then subtitle the winner."""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

from .media import FFmpegMedia
from .models import SafetyResult, StoryboardResult, TranscriptSegment, WorkflowResult
from .providers import Providers
from .selection import select_candidate, validate_safety_result
from .subtitles import validate_segments, write_ass
from .vocabulary import missing_vocabulary


log = logging.getLogger(__name__)


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
        vocabulary: Sequence[str],
        output_dir: str | Path,
        *,
        stop_after: Literal["storyboard", "video"] = "video",
    ) -> StoryboardResult | WorkflowResult:
        """Write selected text artifacts, optionally continuing through final video."""
        if stop_after not in ("storyboard", "video"):
            raise ValueError("stop_after must be 'storyboard' or 'video'.")
        if isinstance(vocabulary, str) or not vocabulary:
            raise ValueError("Provide a nonempty sequence of vocabulary words.")
        if any(not isinstance(word, str) or not word.strip() for word in vocabulary):
            raise ValueError("Every vocabulary item must be a nonempty string.")
        words = tuple(word.strip() for word in vocabulary)
        if stop_after == "video":
            self.providers.video_quality.validate_candidate_count(self.video_candidate_count)
        output = Path(output_dir).resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Output directory must be new or empty to protect existing files.")
        output.mkdir(parents=True, exist_ok=True)
        (output / "vocabulary.json").write_text(
            json.dumps(words, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        providers = self.providers

        def checked_text(text: str) -> str:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Text generators must return nonempty strings.")
            return text

        outline = select_candidate(
            stage="outline",
            candidate_count=self.outline_candidate_count,
            generate=lambda slot, attempt, feedback: checked_text(
                providers.llm.generate_outline(words, feedback)
            ),
            check_safety=lambda text: providers.text_safety.check("outline", text),
            choose_best=lambda candidates: providers.text_quality.choose(
                "outline", words, candidates
            ),
            pass_feedback=True,
            concurrent_candidates=self.concurrent_candidates,
        )
        outline_path = output / "outline.txt"
        outline_path.write_text(outline, encoding="utf-8")

        storyboard = select_candidate(
            stage="storyboard",
            candidate_count=self.storyboard_candidate_count,
            generate=lambda slot, attempt, feedback: checked_text(
                providers.llm.generate_storyboard(words, outline, feedback)
            ),
            check_safety=lambda text: providers.text_safety.check("storyboard", text),
            choose_best=lambda candidates: providers.text_quality.choose(
                "storyboard", words, candidates
            ),
            pass_feedback=True,
            concurrent_candidates=self.concurrent_candidates,
        )
        storyboard_path = output / "storyboard.txt"
        storyboard_path.write_text(storyboard, encoding="utf-8")
        if stop_after == "storyboard":
            return StoryboardResult(outline=outline_path, storyboard=storyboard_path)

        media = self.media if self.media is not None else FFmpegMedia()
        candidates_dir = output / "video_candidates"
        candidates_dir.mkdir()

        def generate_video(slot: int, attempt: int, feedback: str | None) -> _VideoCandidate:
            path = candidates_dir / f"candidate-{slot}-attempt-{attempt}.mp4"
            providers.video.generate(storyboard, path)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Video generator did not write a nonempty video: {path}")
            audio = path.with_suffix(".wav")
            media.extract_audio(path, audio)
            segments = tuple(providers.speech_to_text.transcribe(audio))
            validate_segments(segments)
            transcript = path.with_suffix(".json")
            transcript.write_text(
                json.dumps([asdict(segment) for segment in segments], ensure_ascii=False,
                           indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            return _VideoCandidate(path, audio, transcript, segments)

        def check_video(candidate: _VideoCandidate) -> SafetyResult:
            visual = validate_safety_result(
                providers.video_safety.check(candidate.video), "video visuals"
            )
            audio = validate_safety_result(
                providers.audio_safety.check("\n".join(
                    segment.text for segment in candidate.segments
                )),
                "video audio",
            )
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

        def check_spoken_vocabulary(candidate: _VideoCandidate) -> str | None:
            missing = missing_vocabulary(words, "\n".join(
                segment.text for segment in candidate.segments
            ))
            if not missing:
                return None
            reason = "Recognized speech is missing requested vocabulary: " + ", ".join(
                repr(word) for word in missing
            )
            log.warning("Rejected %s for content: %s", candidate.video.name, reason)
            for path in (candidate.video, candidate.audio, candidate.transcript):
                path.unlink()
            return reason

        selected = select_candidate(
            stage="video",
            candidate_count=self.video_candidate_count,
            generate=generate_video,
            check_safety=check_video,
            check_content=check_spoken_vocabulary,
            choose_best=lambda candidates: providers.video_quality.choose(
                storyboard, tuple(candidate.video for candidate in candidates)
            ),
            pass_feedback=False,
            concurrent_candidates=self.concurrent_candidates,
        )

        audio_path = output / "audio.wav"
        transcript_path = output / "transcript.json"
        shutil.copyfile(selected.audio, audio_path)
        shutil.copyfile(selected.transcript, transcript_path)
        subtitles_path = write_ass(selected.segments, output / "subtitles.ass")
        final_path = output / "final.mp4"
        media.burn_subtitles(selected.video, subtitles_path, final_path)
        return WorkflowResult(
            outline=outline_path,
            storyboard=storyboard_path,
            selected_video=selected.video,
            audio=audio_path,
            transcript=transcript_path,
            subtitles=subtitles_path,
            final_video=final_path,
        )
