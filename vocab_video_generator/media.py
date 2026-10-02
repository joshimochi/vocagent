"""Local FFmpeg operations used by the vocabulary video workflow."""

from pathlib import Path
import shutil
import subprocess
import tempfile


class MediaError(RuntimeError):
    """FFmpeg could not complete a media operation."""


def _input_file(path: Path) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return path


def _output_file(path: Path, *inputs: Path) -> Path:
    path = Path(path).resolve()
    for source in inputs:
        if path == source or (path.exists() and path.samefile(source)):
            raise ValueError("Input and output files must be different.")
    if path.is_dir():
        raise ValueError(f"Output path is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class FFmpegMedia:
    """Extract mono speech audio and burn ASS subtitles into an MP4 video."""

    def __init__(self, ffmpeg: str = "ffmpeg") -> None:
        # burn_subtitles changes cwd, so retain relative executable paths too.
        executable = Path(ffmpeg)
        self.ffmpeg = (
            str(executable.resolve())
            if "/" in ffmpeg or "\\" in ffmpeg or executable.exists()
            else ffmpeg
        )

    def _run(self, arguments: list[str], *, cwd: Path | None = None) -> None:
        try:
            subprocess.run(
                [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *arguments],
                check=True,
                capture_output=True,
                text=True,
                cwd=cwd,
            )
        except FileNotFoundError as exc:
            raise MediaError(f"FFmpeg executable was not found: {self.ffmpeg}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "No error details were returned.").strip()
            raise MediaError(f"FFmpeg failed (exit {exc.returncode}): {detail}") from exc

    def extract_audio(self, video_path: Path, audio_path: Path) -> Path:
        """Extract the first audio stream as mono, 16 kHz, signed 16-bit WAV."""
        video_path = _input_file(video_path)
        audio_path = _output_file(audio_path, video_path)
        self._run([
            "-i", str(video_path), "-map", "0:a:0", "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(audio_path),
        ])
        return audio_path

    def burn_subtitles(self, video_path: Path, subtitle_path: Path, output_path: Path) -> Path:
        """Burn subtitles with libass and encode H.264/AAC MP4 output."""
        video_path = _input_file(video_path)
        subtitle_path = _input_file(subtitle_path)
        output_path = _output_file(output_path, video_path, subtitle_path)
        # The filter parser has several escaping layers. A fixed local filename
        # avoids interpreting any part of the caller's path as filter syntax.
        with tempfile.TemporaryDirectory(prefix="vocab-subtitles-") as directory:
            workdir = Path(directory)
            shutil.copyfile(subtitle_path, workdir / "subtitles.ass")
            try:
                self._run([
                    "-i", str(video_path), "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", "ass=subtitles.ass", "-c:v", "libx264", "-c:a", "aac",
                    "-movflags", "+faststart", "-f", "mp4", str(output_path),
                ], cwd=workdir)
            except MediaError as exc:
                if "No such filter" in str(exc) or "No option name near 'subtitles.ass'" in str(exc):
                    raise MediaError(
                        f"{exc}\nSubtitle burning requires FFmpeg built with libass (the 'ass' filter)."
                    ) from exc
                raise
        return output_path
