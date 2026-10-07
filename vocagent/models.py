"""Small data objects shared by the workflow and provider implementations."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    reason: str = ""


@dataclass(frozen=True)
class TranscriptSegment:
    """A subtitle cue; start and end are seconds from the start of the audio."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class VideoPromptResult:
    """Selected, safety-checked outline and video prompt artifacts."""

    outline: Path
    video_prompt: Path
    benchmarks: Path | None = None


@dataclass(frozen=True)
class WorkflowResult:
    outline: Path
    video_prompt: Path
    selected_video: Path
    audio: Path
    transcript: Path
    subtitles: Path
    final_video: Path
    benchmarks: Path | None = None
