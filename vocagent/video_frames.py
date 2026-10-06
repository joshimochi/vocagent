"""Bounded local video sampling shared by visual quality and safety providers."""

import base64
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

from .media import MediaError, _input_file


@dataclass(frozen=True)
class VideoFrame:
    timestamp: float
    jpeg: bytes

    @property
    def data_url(self) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.jpeg).decode("ascii")


def _run(arguments: list[str], timeout: float = 60) -> bytes:
    try:
        return subprocess.run(
            arguments, check=True, capture_output=True, timeout=timeout,
        ).stdout
    except FileNotFoundError as exc:
        raise MediaError(f"Video sampling executable was not found: {arguments[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError("Video frame sampling timed out.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", errors="replace").strip()[:2000]
        raise MediaError(f"Video frame sampling failed: {detail}") from exc


def sample_video_frames(
    video_path: Path, *, max_frames: int = 16, max_side: int = 768,
    sample_fps: float | None = None, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
) -> tuple[VideoFrame, ...]:
    """Sample the whole timeline, including its beginning and last nominal frame.

    Quality mode spreads at most max_frames evenly across the clip. Fixed-fps
    safety mode rejects clips exceeding max_frames rather than skipping their
    tail or silently reducing coverage. Both modes are image-based sampling;
    brief events between samples are not guaranteed to be observed.
    """
    if type(max_frames) is not int or not 1 <= max_frames <= 10000:
        raise ValueError("Video sampling max_frames must be an integer from 1 to 10000.")
    if type(max_side) is not int or not 32 <= max_side <= 4096:
        raise ValueError("Video sampling max_side must be an integer from 32 to 4096.")
    if sample_fps is not None and (
        isinstance(sample_fps, bool) or not isinstance(sample_fps, (int, float))
        or not math.isfinite(sample_fps) or not 0 < sample_fps <= 120
    ):
        raise ValueError("Video sampling fps must be finite and between 0 and 120.")
    path = _input_file(video_path)
    raw = _run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=duration,avg_frame_rate:format=duration", "-of", "json", str(path),
    ])
    try:
        data = json.loads(raw)
        stream = data["streams"][0]
        duration = float(stream.get("duration") or data.get("format", {}).get("duration"))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid duration")
        try:
            frame_rate = float(Fraction(stream.get("avg_frame_rate", "0/1")))
        except (ValueError, ZeroDivisionError):
            frame_rate = 0
        # Avoid seeking past the last decodable frame at the container duration.
        frame_interval = 1 / frame_rate if frame_rate > 0 else min(0.1, duration)
        final_time = max(0.0, duration - frame_interval)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
        raise MediaError("Video sampling requires a video stream with a finite positive duration.") from exc
    if sample_fps is None:
        count = min(max_frames, max(1, math.ceil(duration * (frame_rate or 1))))
        timestamps = ([final_time / 2] if count == 1 and final_time > 0 else
                      [final_time * index / max(1, count - 1) for index in range(count)])
    else:
        count = math.floor(final_time * sample_fps) + 1
        add_final = final_time - (count - 1) / sample_fps > 1e-6
        if count + int(add_final) > max_frames:
            raise MediaError(
                f"Video requires {count + int(add_final)} frames at {sample_fps:g} fps; "
                f"the configured limit is {max_frames}."
            )
        timestamps = [index / sample_fps for index in range(count)]
        if add_final:
            timestamps.append(final_time)
    frames = []
    for timestamp in timestamps:
        jpeg = _run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{timestamp:.9f}", "-i", str(path), "-map", "0:v:0",
            "-frames:v", "1", "-an", "-sn", "-vf",
            f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease",
            "-threads", "1", "-c:v", "mjpeg", "-q:v", "3", "-f", "image2pipe", "pipe:1",
        ])
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise MediaError(f"Video sampling produced no complete JPEG at {timestamp:.3f} seconds.")
        if len(jpeg) > 8 * 1024 * 1024:
            raise MediaError("A sampled video frame exceeded 8 MiB.")
        frames.append(VideoFrame(timestamp, jpeg))
    return tuple(frames)
