"""Provider interfaces and the complete set of workflow dependencies.

Generation, safety assessment, and quality selection are separate dependencies.
Text safety and quality must use agents separate from the generating agent;
visual safety and video quality must use specialized models. Audio safety uses
a separate model to assess the transcript of each generated video.
Candidate slots run sequentially by default. When concurrent candidate generation
is enabled, generation, transcription, and safety methods must support concurrent
calls on the same instance, up to the stage's configured candidate count (two by default).
Quality judging runs after all candidate slots finish.
Vocabulary sequences contain exactly one English target word per workflow.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .models import SafetyResult, TranscriptSegment


class LLMGenerator:
    def generate_outline(
        self, vocabulary: Sequence[str], feedback: str | None
    ) -> str:
        raise NotImplementedError("Implement LLMGenerator.generate_outline")

    def generate_video_prompt(
        self, vocabulary: Sequence[str], outline: str, feedback: str | None
    ) -> str:
        raise NotImplementedError("Implement LLMGenerator.generate_video_prompt")


class TextSafetyAgent:
    def check(self, stage: str, text: str) -> SafetyResult:
        raise NotImplementedError("Implement a separate TextSafetyAgent.check")


class TextQualityAgent:
    def choose(
        self, stage: str, vocabulary: Sequence[str], candidates: Sequence[str]
    ) -> int:
        """Return the zero-based index of the best safe candidate."""
        raise NotImplementedError("Implement a separate TextQualityAgent.choose")


class VideoGenerator:
    def generate(self, video_prompt: str, output_path: Path) -> None:
        """Write a video with an audio track to output_path."""
        raise NotImplementedError("Implement VideoGenerator.generate")


class VideoSafetyModel:
    def check(self, video_path: Path) -> SafetyResult:
        """Assess visual content; spoken content is checked by AudioSafetyModel."""
        raise NotImplementedError("Implement a specialized VideoSafetyModel.check")


class AudioSafetyModel:
    def check(self, transcript: str) -> SafetyResult:
        """Assess spoken content from the complete, newline-joined transcript."""
        raise NotImplementedError("Implement a separate AudioSafetyModel.check")


class VideoQualityModel:
    def validate_candidate_count(self, count: int) -> None:
        """Validate configured slots before generating any video candidates."""
        if type(count) is not int or count < 1:
            raise ValueError("Video candidate count must be a positive integer.")

    def choose(self, video_prompt: str, candidates: Sequence[Path]) -> int:
        """Return the zero-based index of the best safe video."""
        raise NotImplementedError("Implement a specialized VideoQualityModel.choose")


class SpeechToText:
    def transcribe(self, audio_path: Path) -> Sequence[TranscriptSegment]:
        """Return timestamped segments, with times measured in seconds."""
        raise NotImplementedError("Implement SpeechToText.transcribe")


@dataclass
class Providers:
    llm: LLMGenerator
    text_safety: TextSafetyAgent
    text_quality: TextQualityAgent
    video: VideoGenerator
    video_safety: VideoSafetyModel
    video_quality: VideoQualityModel
    speech_to_text: SpeechToText
    audio_safety: AudioSafetyModel

    def close(self) -> None:
        """Release provider resources; resource-owning bundles override this hook."""
