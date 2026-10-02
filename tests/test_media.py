from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from vocab_video_generator.media import FFmpegMedia, MediaError
from vocab_video_generator.models import TranscriptSegment
from vocab_video_generator.subtitles import write_ass


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.video = self.root / "video ' [scene],take:1\\clip.mp4"
        self.video.write_bytes(b"video")
        self.subtitles = self.root / "subtitles ' [scene],take:1\\clip.ass"
        self.subtitles.write_text("subtitle content", encoding="utf-8")
        self.media = FFmpegMedia()

    @patch("vocab_video_generator.media.subprocess.run")
    def test_audio_command_preserves_paths_and_requests_speech_format(self, run):
        output = self.root / "nested" / "audio.wav"
        self.assertEqual(self.media.extract_audio(self.video, output), output)
        arguments = run.call_args.args[0]
        self.assertEqual(arguments[arguments.index("-i") + 1], str(self.video))
        for flag, value in (("-ac", "1"), ("-ar", "16000"), ("-c:a", "pcm_s16le")):
            self.assertEqual(arguments[arguments.index(flag) + 1], value)
        self.assertEqual(arguments[-1], str(output))
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertNotIn("shell", run.call_args.kwargs)

    @patch("vocab_video_generator.media.subprocess.run")
    def test_burn_uses_safe_fixed_subtitle_name_in_temporary_directory(self, run):
        copied_directories = []

        def check_copy(arguments, **kwargs):
            directory = kwargs["cwd"]
            copied_directories.append(directory)
            self.assertEqual((directory / "subtitles.ass").read_text(), "subtitle content")
            self.assertEqual(arguments[arguments.index("-vf") + 1], "ass=subtitles.ass")
            self.assertEqual(arguments[arguments.index("-i") + 1], str(self.video))
            self.assertEqual(arguments[arguments.index("-c:v") + 1], "libx264")
            self.assertEqual(arguments[arguments.index("-c:a") + 1], "aac")
            self.assertEqual(arguments[-1], str(output))

        run.side_effect = check_copy
        output = self.root / "final ' [scene],take:1\\clip.mp4"
        self.assertEqual(self.media.burn_subtitles(self.video, self.subtitles, output), output)
        self.assertFalse(copied_directories[0].exists())

    @patch("vocab_video_generator.media.subprocess.run")
    def test_rejects_overwriting_inputs_including_aliases(self, run):
        alias = self.root / "alias.mp4"
        alias.symlink_to(self.video)
        hardlink = self.root / "hardlink.mp4"
        hardlink.hardlink_to(self.video)
        for output in (self.video, alias, hardlink):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.media.extract_audio(self.video, output)
        with self.assertRaises(ValueError):
            self.media.burn_subtitles(self.video, self.subtitles, self.subtitles)
        run.assert_not_called()

    @patch("vocab_video_generator.media.subprocess.run")
    def test_missing_input_is_reported_before_running_ffmpeg(self, run):
        with self.assertRaises(FileNotFoundError):
            self.media.extract_audio(self.root / "missing.mp4", self.root / "audio.wav")
        run.assert_not_called()

    @patch("vocab_video_generator.media.subprocess.run")
    def test_ffmpeg_errors_include_diagnostic(self, run):
        run.side_effect = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="No audio stream")
        with self.assertRaisesRegex(MediaError, "No audio stream"):
            self.media.extract_audio(self.video, self.root / "audio.wav")
        run.side_effect = FileNotFoundError("missing executable")
        with self.assertRaisesRegex(MediaError, "executable was not found"):
            self.media.extract_audio(self.video, self.root / "audio.wav")

    @patch("vocab_video_generator.media.subprocess.run")
    def test_missing_libass_error_explains_required_ffmpeg_support(self, run):
        run.side_effect = subprocess.CalledProcessError(
            1, ["ffmpeg"], stderr="No option name near 'subtitles.ass'"
        )
        with self.assertRaisesRegex(MediaError, "requires FFmpeg built with libass"):
            self.media.burn_subtitles(self.video, self.subtitles, self.root / "final.mp4")


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
class FFmpegIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_real_audio_extraction(self):
        source = self.root / "source ' [audio],take:1\\clip.wav"
        with wave.open(str(source), "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(48000)
            stream.writeframes(b"\0" * 48000 * 2 * 2)
        output = FFmpegMedia().extract_audio(source, self.root / "speech.wav")
        with wave.open(str(output), "rb") as stream:
            self.assertEqual(stream.getnchannels(), 1)
            self.assertEqual(stream.getframerate(), 16000)
            self.assertEqual(stream.getsampwidth(), 2)
            self.assertEqual(stream.getnframes(), 16000)

    def test_real_subtitle_burn_when_libass_is_available(self):
        filters = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], check=True, capture_output=True, text=True
        ).stdout
        if not any(line.split()[1:2] == ["ass"] for line in filters.splitlines()):
            self.skipTest("Installed FFmpeg does not include the libass filter")
        source = self.root / "video ' [scene],take:1\\clip.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x180:d=0.3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
        ], check=True, capture_output=True)
        subtitles = write_ass(
            [TranscriptSegment(0, 0.3, "Hello {world}\\Name")],
            self.root / "subtitles ' [scene],take:1\\clip.ass",
        )
        output = FFmpegMedia().burn_subtitles(source, subtitles, self.root / "final.mp4")
        self.assertGreater(output.stat().st_size, 0)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(output),
            "-f", "null", "-",
        ], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
