"""Replace these stubs with model integrations; no model SDKs are required.

Generation, safety assessment, and quality selection are separate dependencies.
Text safety and quality must use agents separate from the generating agent;
visual safety and video quality must use specialized models. Audio safety uses
a separate model to assess the transcript of each generated video.
Candidate slots run sequentially by default. When concurrent candidate generation
is enabled, generation, transcription, and safety methods must support concurrent
calls on the same instance, up to the stage's configured candidate count (two by default).
Quality judging runs after all candidate slots finish.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .models import SafetyResult, TranscriptSegment


class LLMGenerator:
    def generate_outline(
        self, vocabulary: Sequence[str], feedback: str | None
    ) -> str:
        raise NotImplementedError("Implement LLMGenerator.generate_outline")

    def generate_storyboard(
        self, vocabulary: Sequence[str], outline: str, feedback: str | None
    ) -> str:
        raise NotImplementedError("Implement LLMGenerator.generate_storyboard")


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
    def generate(self, storyboard: str, output_path: Path) -> None:
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
    def choose(self, storyboard: str, candidates: Sequence[Path]) -> int:
        """Return the zero-based index of the best safe video."""
        raise NotImplementedError("Implement a specialized VideoQualityModel.choose")


class SpeechToText:
    def transcribe(self, audio_path: Path) -> Sequence[TranscriptSegment]:
        """Return timestamped segments, with times measured in seconds."""
        raise NotImplementedError("Implement SpeechToText.transcribe")


@dataclass
class Providers:
    llm: LLMGenerator = field(default_factory=LLMGenerator)
    text_safety: TextSafetyAgent = field(default_factory=TextSafetyAgent)
    text_quality: TextQualityAgent = field(default_factory=TextQualityAgent)
    video: VideoGenerator = field(default_factory=VideoGenerator)
    video_safety: VideoSafetyModel = field(default_factory=VideoSafetyModel)
    video_quality: VideoQualityModel = field(default_factory=VideoQualityModel)
    speech_to_text: SpeechToText = field(default_factory=SpeechToText)
    audio_safety: AudioSafetyModel = field(default_factory=AudioSafetyModel)
