"""Actual-pixel sampling contracts and FFmpeg integration for visual models."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from vocagent.media import MediaError
from vocagent.video_frames import sample_video_frames


class VideoFrameTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "video.mp4"
        self.path.write_bytes(b"video")

    def probe(self, duration=10, rate="30/1"):
        return json.dumps({"streams": [{"duration": duration, "avg_frame_rate": rate}]}).encode()

    def test_uniform_mode_covers_both_ends_and_returns_actual_jpeg_bytes(self):
        jpeg = b"\xff\xd8frame\xff\xd9"
        with patch("vocagent.video_frames._run", side_effect=[self.probe(), jpeg, jpeg, jpeg]) as run:
            frames = sample_video_frames(self.path, max_frames=3, max_side=448)
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0].timestamp, 0)
        self.assertAlmostEqual(frames[-1].timestamp, 10 - 1 / 30)
        self.assertEqual(frames[0].jpeg, jpeg)
        args = run.call_args_list[-1].args[0]
        self.assertIn("scale=448:448:force_original_aspect_ratio=decrease", args)

    def test_fixed_fps_adds_final_frame_and_never_silently_truncates(self):
        jpeg = b"\xff\xd8frame\xff\xd9"
        with patch("vocagent.video_frames._run", side_effect=[self.probe(2), *([jpeg] * 5)]):
            frames = sample_video_frames(self.path, sample_fps=2, max_frames=5)
        self.assertEqual([frame.timestamp for frame in frames[:4]], [0, 0.5, 1, 1.5])
        self.assertAlmostEqual(frames[-1].timestamp, 2 - 1 / 30)
        with patch("vocagent.video_frames._run", return_value=self.probe(2)) as run:
            with self.assertRaisesRegex(MediaError, "requires 5 frames"):
                sample_video_frames(self.path, sample_fps=2, max_frames=4)
        run.assert_called_once()

    def test_bad_duration_absent_stream_and_missing_frames_fail_closed(self):
        for raw in (b"[]", b"{}", b"bad", self.probe(0), self.probe("nan"), b'{"streams": [null]}'):
            with self.subTest(raw=raw), patch("vocagent.video_frames._run", return_value=raw):
                with self.assertRaises(MediaError):
                    sample_video_frames(self.path)
        for jpeg in (b"", b"partial", b"\xff\xd8truncated"):
            with patch("vocagent.video_frames._run", side_effect=[self.probe(), jpeg]):
                with self.assertRaisesRegex(MediaError, "no complete JPEG"):
                    sample_video_frames(self.path, max_frames=1)

    def test_bad_sampling_options_rejected_before_media_processing(self):
        for options in ({"max_frames": True}, {"max_frames": 0}, {"max_side": 2},
                        {"sample_fps": 0}, {"sample_fps": float("inf")}, {"sample_fps": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                sample_video_frames(self.path, **options)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_real_video_has_decodable_first_and_last_samples(self):
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=1",
            "-c:v", "mpeg4", str(self.path),
        ], check=True, capture_output=True)
        uniform = sample_video_frames(self.path, max_frames=3, max_side=64)
        regular = sample_video_frames(self.path, sample_fps=2, max_frames=4, max_side=64)
        self.assertEqual(len(uniform), 3)
        self.assertEqual(len(regular), 3)
        self.assertEqual(uniform[0].timestamp, 0)
        self.assertAlmostEqual(uniform[-1].timestamp, 0.9)
        self.assertNotEqual(uniform[0].jpeg, uniform[-1].jpeg)


if __name__ == "__main__":
    unittest.main()
