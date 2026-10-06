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
class StoryboardResult:
    """Selected, safety-checked outline and storyboard artifacts."""

    outline: Path
    storyboard: Path


@dataclass(frozen=True)
class WorkflowResult:
    outline: Path
    storyboard: Path
    selected_video: Path
    audio: Path
    transcript: Path
    subtitles: Path
    final_video: Path
