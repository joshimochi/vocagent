"""Convert timestamped plain text into subtitles accepted by FFmpeg/libass."""

from collections.abc import Sequence
from decimal import Decimal, ROUND_HALF_UP
import math
from numbers import Real
from pathlib import Path
import unicodedata

from .models import TranscriptSegment


_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,40,40,35,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def validate_segments(segments: Sequence[TranscriptSegment]) -> None:
    """Require usable cues ordered by start time; overlapping speech is allowed."""
    if not segments:
        raise ValueError("The transcript must contain at least one segment.")
    previous_start = -1.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, TranscriptSegment):
            raise ValueError(f"Transcript segment {index} must be a TranscriptSegment.")
        for name, value in (("start", segment.start), ("end", segment.end)):
            try:
                valid = (
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                )
            except (OverflowError, TypeError):
                valid = False
            if not valid:
                raise ValueError(f"Segment {index} {name} must be finite, nonnegative seconds.")
        if segment.end <= segment.start:
            raise ValueError(f"Segment {index} must end after it starts.")
        if segment.start < previous_start:
            raise ValueError("Transcript segments must be ordered by start time.")
        previous_start = segment.start
        if not isinstance(segment.text, str) or not any(
            char.isprintable() and not char.isspace() for char in segment.text
        ):
            raise ValueError(f"Segment {index} must contain visible text.")
        if any(
            unicodedata.category(char) in {"Cc", "Cs"} and char not in "\t\r\n"
            for char in segment.text
        ):
            raise ValueError(f"Segment {index} contains unsupported control characters.")


def _centiseconds(seconds: float) -> int:
    # Decimal prevents 1.005-style binary rounding errors, before carrying units.
    return int((Decimal(str(seconds)) * 100).to_integral_value(rounding=ROUND_HALF_UP))


def _timestamp(centiseconds: int) -> str:
    seconds, fraction = divmod(centiseconds, 100)
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour}:{minute:02d}:{second:02d}.{fraction:02d}"


def _escape_text(text: str) -> str:
    # libass has no \\ escape: doubling a slash still activates a following N/n/h.
    # A zero-width WORD JOINER after literal slashes preserves their appearance.
    text = text.replace("\\", "\\\u2060").replace("{", r"\{").replace("}", r"\}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\u2028", "\n").replace("\u2029", "\n").replace("\n", r"\N").replace("\t", "    ")


def write_ass(segments: Sequence[TranscriptSegment], output_path: Path) -> Path:
    """Write UTF-8 ASS subtitles, rounding times to the nearest centisecond."""
    validate_segments(segments)
    lines = [_ASS_HEADER]
    for segment in segments:
        start = _centiseconds(segment.start)
        # ASS cannot represent a duration below 10 ms; keep short cues visible.
        end = max(start + 1, _centiseconds(segment.end))
        lines.append(
            f"Dialogue: 0,{_timestamp(start)},{_timestamp(end)},Default,,0,0,0,,"
            f"{_escape_text(segment.text)}\n"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
