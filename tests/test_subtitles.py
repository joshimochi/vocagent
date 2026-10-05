from pathlib import Path
import tempfile
import unittest

from vocagent.models import TranscriptSegment
from vocagent.subtitles import validate_segments, write_ass


class SubtitleTests(unittest.TestCase):
    def test_valid_overlapping_segments(self):
        validate_segments([
            TranscriptSegment(0, 3, "Hello"),
            TranscriptSegment(1, 2, "world"),
            TranscriptSegment(1, 4, "你好"),
        ])

    def test_rejects_empty_and_unusable_transcripts(self):
        invalid = [
            [],
            [TranscriptSegment(-1, 1, "word")],
            [TranscriptSegment(0, 0, "word")],
            [TranscriptSegment(2, 1, "word")],
            [TranscriptSegment(float("nan"), 1, "word")],
            [TranscriptSegment(0, float("inf"), "word")],
            [TranscriptSegment(True, 2, "word")],
            [TranscriptSegment("0", 2, "word")],
            [TranscriptSegment(0, 1, " \n\t")],
            [TranscriptSegment(0, 1, "\u2060\u200b")],
            [TranscriptSegment(0, 1, "word\0hidden")],
            [TranscriptSegment(0, 1, "word\ud800")],
            [TranscriptSegment(0, 1, None)],
            [TranscriptSegment(1, 2, "late"), TranscriptSegment(0, 1, "early")],
        ]
        for segments in invalid:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                validate_segments(segments)

    def test_centisecond_rounding_carries_into_minutes_and_hours(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "subtitles.ass"
            result = write_ass([
                TranscriptSegment(1.005, 2.005, "First"),
                TranscriptSegment(59.999, 60.016, "Second"),
                TranscriptSegment(3599.999, 3600.001, "Short cue"),
            ], path)
            self.assertEqual(result, path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("[V4+ Styles]", text)
            self.assertIn("Dialogue: 0,0:00:01.01,0:00:02.01,", text)
            self.assertIn("Dialogue: 0,0:01:00.00,0:01:00.02,", text)
            self.assertIn("Dialogue: 0,1:00:00.00,1:00:00.01,", text)

    def test_plain_text_cannot_add_ass_tags_or_dialogue_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subtitles.ass"
            write_ass([TranscriptSegment(
                0, 1, "你好, {\\b1}path\\Name\\here\r\nDialogue: harmless\rnext\nlast"
            )], path)
            text = path.read_text(encoding="utf-8")
            rows = [line for line in text.splitlines() if line.startswith("Dialogue:")]
            self.assertEqual(len(rows), 1)
            cue = rows[0].split(",", 9)[9]
            self.assertEqual(
                cue,
                "你好, \\{\\\u2060b1\\}path\\\u2060Name\\\u2060here"
                "\\NDialogue: harmless\\Nnext\\Nlast",
            )

    def test_invalid_transcript_does_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subtitles.ass"
            path.write_text("keep me", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_ass([], path)
            self.assertEqual(path.read_text(encoding="utf-8"), "keep me")


if __name__ == "__main__":
    unittest.main()
