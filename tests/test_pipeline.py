"""Exercise the whole workflow with deterministic providers and local fake media."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, call

from vocab_video_generator.media import FFmpegMedia
from vocab_video_generator.models import SafetyResult, TranscriptSegment
from vocab_video_generator.pipeline import Workflow
from vocab_video_generator.providers import (
    LLMGenerator, Providers, SpeechToText, TextQualityAgent, TextSafetyAgent,
    VideoGenerator, VideoQualityModel, VideoSafetyModel,
)
from vocab_video_generator.selection import HumanInterventionRequired


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name).resolve() / "result"
        self.llm = Mock(spec=LLMGenerator)
        self.llm.generate_outline.side_effect = (
            lambda vocabulary, feedback: f"outline {self.llm.generate_outline.call_count}"
        )
        self.llm.generate_storyboard.side_effect = (
            lambda vocabulary, outline, feedback:
            f"storyboard {self.llm.generate_storyboard.call_count}"
        )
        self.text_safety = Mock(spec=TextSafetyAgent)
        self.text_safety.check.return_value = SafetyResult(True)
        self.text_quality = Mock(spec=TextQualityAgent)
        self.text_quality.choose.side_effect = (
            lambda stage, vocabulary, candidates: 2 if stage == "outline" else 1
        )
        self.video = Mock(spec=VideoGenerator)
        self.video.generate.side_effect = (
            lambda storyboard, output_path: output_path.write_bytes(b"fake video")
        )
        self.video_safety = Mock(spec=VideoSafetyModel)
        self.video_safety.check.return_value = SafetyResult(True)
        self.video_quality = Mock(spec=VideoQualityModel)
        self.video_quality.choose.return_value = 3
        self.speech_to_text = Mock(spec=SpeechToText)
        self.segments = [
            TranscriptSegment(0.25, 1.5, "Hello, apple."),
            TranscriptSegment(2.0, 3.75, "A green pear. 梨"),
        ]
        self.speech_to_text.transcribe.return_value = self.segments
        self.providers = Providers(
            llm=self.llm, text_safety=self.text_safety, text_quality=self.text_quality,
            video=self.video, video_safety=self.video_safety,
            video_quality=self.video_quality, speech_to_text=self.speech_to_text,
        )
        self.media = Mock(spec=FFmpegMedia)
        self.media.extract_audio.side_effect = (
            lambda video_path, audio_path: audio_path.write_bytes(b"fake audio")
        )
        self.media.burn_subtitles.side_effect = (
            lambda video_path, subtitle_path, output_path:
            output_path.write_bytes(b"fake subtitled video")
        )
        self.workflow = Workflow(self.providers, self.media)

    def test_chosen_candidates_flow_through_all_eight_steps(self):
        result = self.workflow.run([" apple ", "pear"], self.output)

        words = ("apple", "pear")
        self.assertEqual(self.llm.generate_outline.call_args_list, [call(words, None)] * 4)
        self.assertEqual(
            self.llm.generate_storyboard.call_args_list,
            [call(words, "outline 3", None)] * 4,
        )
        self.assertEqual(self.text_quality.choose.call_args_list, [
            call("outline", words, ("outline 1", "outline 2", "outline 3", "outline 4")),
            call("storyboard", words,
                 ("storyboard 1", "storyboard 2", "storyboard 3", "storyboard 4")),
        ])
        videos = tuple(
            self.output / "video_candidates" / f"candidate-{slot}-attempt-1.mp4"
            for slot in range(1, 5)
        )
        self.assertEqual(
            self.video.generate.call_args_list,
            [call("storyboard 2", path) for path in videos],
        )
        self.video_quality.choose.assert_called_once_with("storyboard 2", videos)
        self.assertEqual(result.selected_video, videos[3])
        self.media.extract_audio.assert_called_once_with(videos[3], result.audio)
        self.speech_to_text.transcribe.assert_called_once_with(result.audio)
        self.media.burn_subtitles.assert_called_once_with(
            videos[3], result.subtitles, result.final_video,
        )
        self.assertEqual(result.outline.read_text(), "outline 3")
        self.assertEqual(result.storyboard.read_text(), "storyboard 2")
        self.assertEqual(json.loads((self.output / "vocabulary.json").read_text()), list(words))
        self.assertEqual(json.loads(result.transcript.read_text()), [
            {"start": 0.25, "end": 1.5, "text": "Hello, apple."},
            {"start": 2.0, "end": 3.75, "text": "A green pear. 梨"},
        ])
        subtitles = result.subtitles.read_text(encoding="utf-8")
        self.assertIn("[Events]", subtitles)
        self.assertIn("0:00:00.25,0:00:01.50", subtitles)
        self.assertIn("0:00:02.00,0:00:03.75", subtitles)
        self.assertIn("A green pear. 梨", subtitles)
        self.assertEqual(result.audio.read_bytes(), b"fake audio")
        self.assertEqual(result.final_video.read_bytes(), b"fake subtitled video")

    def test_text_safety_feedback_reaches_only_the_retried_stage(self):
        self.text_safety.check.side_effect = (
            lambda stage, text: SafetyResult(text != "outline 1", "remove unsafe content")
        )

        result = self.workflow.run(["apple"], self.output)

        self.assertEqual(self.llm.generate_outline.call_args_list, [
            call(("apple",), None), call(("apple",), "remove unsafe content"),
            call(("apple",), None), call(("apple",), None), call(("apple",), None),
        ])
        self.assertEqual(result.outline.read_text(), "outline 4")
        self.assertEqual(
            self.llm.generate_storyboard.call_args_list,
            [call(("apple",), "outline 4", None)] * 4,
        )

    def test_unsafe_video_is_deleted_before_regeneration_without_feedback(self):
        rejected = self.output / "video_candidates" / "candidate-1-attempt-1.mp4"

        def generate(storyboard, output_path):
            if output_path.name == "candidate-1-attempt-2.mp4":
                self.assertFalse(rejected.exists())
            output_path.write_bytes(b"fake video")

        self.video.generate.side_effect = generate
        self.video_safety.check.side_effect = (
            lambda path: SafetyResult(path != rejected, "do not pass this reason to video")
        )

        self.workflow.run(["apple"], self.output)

        self.assertFalse(rejected.exists())
        self.assertEqual(self.video.generate.call_count, 5)
        self.assertTrue(all(len(item.args) == 2 for item in self.video.generate.call_args_list))
        self.assertTrue(all(item.args[0] == "storyboard 2"
                            for item in self.video.generate.call_args_list))
        survivors = self.video_quality.choose.call_args.args[1]
        self.assertEqual(len(survivors), 4)
        self.assertNotIn(rejected, survivors)
        self.assertEqual(survivors[0].name, "candidate-1-attempt-2.mp4")

    def test_all_unsafe_outline_stops_before_storyboard(self):
        self.text_safety.check.return_value = SafetyResult(False, "unsafe outline")

        with self.assertRaisesRegex(HumanInterventionRequired, "outline"):
            self.workflow.run(["apple"], self.output)

        self.assertEqual(self.llm.generate_outline.call_count, 16)
        self.llm.generate_storyboard.assert_not_called()
        self.text_quality.choose.assert_not_called()
        self.video.generate.assert_not_called()
        self.media.extract_audio.assert_not_called()

    def test_all_unsafe_storyboards_stop_before_video(self):
        self.text_safety.check.side_effect = (
            lambda stage, text: SafetyResult(stage != "storyboard", "unsafe storyboard")
        )

        with self.assertRaisesRegex(HumanInterventionRequired, "storyboard"):
            self.workflow.run(["apple"], self.output)

        self.assertEqual(self.llm.generate_storyboard.call_count, 16)
        self.assertEqual(self.text_quality.choose.call_count, 1)
        self.video.generate.assert_not_called()
        self.media.extract_audio.assert_not_called()

    def test_all_unsafe_videos_are_removed_and_stop_before_audio(self):
        self.video_safety.check.return_value = SafetyResult(False, "unsafe video")

        with self.assertRaisesRegex(HumanInterventionRequired, "video"):
            self.workflow.run(["apple"], self.output)

        self.assertEqual(self.video.generate.call_count, 16)
        self.assertEqual(list((self.output / "video_candidates").iterdir()), [])
        self.video_quality.choose.assert_not_called()
        self.media.extract_audio.assert_not_called()
        self.speech_to_text.transcribe.assert_not_called()
        self.media.burn_subtitles.assert_not_called()

    def test_invalid_vocabulary_does_not_create_output_or_call_providers(self):
        for vocabulary in (None, [], "apple", [""], ["  "], [1], ["apple", None]):
            with self.subTest(vocabulary=vocabulary):
                with self.assertRaises(ValueError):
                    self.workflow.run(vocabulary, self.output)
                self.assertFalse(self.output.exists())
        self.llm.generate_outline.assert_not_called()

    def test_nonempty_output_is_rejected_without_touching_existing_files(self):
        self.output.mkdir()
        original = self.output / "outline.txt"
        original.write_bytes(b"existing work")

        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.workflow.run(["apple"], self.output)

        self.assertEqual(list(self.output.iterdir()), [original])
        self.assertEqual(original.read_bytes(), b"existing work")
        self.llm.generate_outline.assert_not_called()

    def test_output_file_is_rejected_without_overwriting_it(self):
        self.output.write_bytes(b"existing file")

        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.workflow.run(["apple"], self.output)

        self.assertEqual(self.output.read_bytes(), b"existing file")
        self.llm.generate_outline.assert_not_called()

    def test_missing_or_empty_video_stops_before_safety_or_media(self):
        for write_empty in (False, True):
            with self.subTest(write_empty=write_empty):
                self.video.generate.side_effect = (
                    lambda storyboard, path: path.write_bytes(b"") if write_empty else None
                )
                output = self.output / str(write_empty)
                with self.assertRaisesRegex(ValueError, "nonempty video"):
                    self.workflow.run(["apple"], output)
        self.video_safety.check.assert_not_called()
        self.video_quality.choose.assert_not_called()
        self.media.extract_audio.assert_not_called()

    def test_invalid_transcript_is_rejected_before_serialization_or_subtitle_burn(self):
        transcripts = (
            [], [TranscriptSegment(-1, 1, "word")],
            [TranscriptSegment(0, float("nan"), "word")],
            [TranscriptSegment(2, 1, "word")], [{"start": 0, "end": 1, "text": "word"}],
        )
        for index, transcript in enumerate(transcripts):
            with self.subTest(transcript=transcript):
                self.speech_to_text.transcribe.return_value = transcript
                output = self.output / str(index)
                with self.assertRaises(ValueError):
                    self.workflow.run(["apple"], output)
                self.assertFalse((output / "transcript.json").exists())
                self.assertFalse((output / "subtitles.ass").exists())
        self.media.burn_subtitles.assert_not_called()


if __name__ == "__main__":
    unittest.main()
