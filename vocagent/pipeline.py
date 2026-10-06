"""Run the eight steps described in workflow.md."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .media import FFmpegMedia
from .models import SafetyResult, WorkflowResult
from .providers import Providers
from .selection import select_candidate
from .subtitles import validate_segments, write_ass


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
        self.media = media if media is not None else FFmpegMedia()
        self.outline_candidate_count = outline_candidate_count
        self.storyboard_candidate_count = storyboard_candidate_count
        self.video_candidate_count = video_candidate_count
        self.concurrent_candidates = concurrent_candidates

    def run(self, vocabulary: Sequence[str], output_dir: str | Path) -> WorkflowResult:
        if isinstance(vocabulary, str) or not vocabulary:
            raise ValueError("Provide a nonempty sequence of vocabulary words.")
        if any(not isinstance(word, str) or not word.strip() for word in vocabulary):
            raise ValueError("Every vocabulary item must be a nonempty string.")
        words = tuple(word.strip() for word in vocabulary)
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
        candidates_dir = output / "video_candidates"
        candidates_dir.mkdir()

        def generate_video(slot: int, attempt: int, feedback: str | None) -> Path:
            path = candidates_dir / f"candidate-{slot}-attempt-{attempt}.mp4"
            providers.video.generate(storyboard, path)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Video generator did not write a nonempty video: {path}")
            return path

        def check_video(path: Path) -> SafetyResult:
            assessment = providers.video_safety.check(path)
            if isinstance(assessment, SafetyResult) and assessment.safe is False:
                path.unlink()
            return assessment

        selected_video = select_candidate(
            stage="video",
            candidate_count=self.video_candidate_count,
            generate=generate_video,
            check_safety=check_video,
            choose_best=lambda candidates: providers.video_quality.choose(storyboard, candidates),
            pass_feedback=False,
            concurrent_candidates=self.concurrent_candidates,
        )

        audio_path = output / "audio.wav"
        self.media.extract_audio(selected_video, audio_path)
        segments = list(providers.speech_to_text.transcribe(audio_path))
        validate_segments(segments)
        transcript_path = output / "transcript.json"
        transcript_path.write_text(
            json.dumps([asdict(segment) for segment in segments], ensure_ascii=False,
                       indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        subtitles_path = write_ass(segments, output / "subtitles.ass")
        final_path = output / "final.mp4"
        self.media.burn_subtitles(selected_video, subtitles_path, final_path)
        return WorkflowResult(
            outline=outline_path,
            storyboard=storyboard_path,
            selected_video=selected_video,
            audio=audio_path,
            transcript=transcript_path,
            subtitles=subtitles_path,
            final_video=final_path,
        )
