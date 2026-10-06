"""Exercise the whole workflow with deterministic providers and local fake media."""

import json
from pathlib import Path
import tempfile
from threading import Barrier, Lock, current_thread
import unittest
from unittest.mock import Mock, call

from vocagent.media import FFmpegMedia
from vocagent.models import SafetyResult, TranscriptSegment
from vocagent.pipeline import Workflow
from vocagent.providers import (
    LLMGenerator, Providers, SpeechToText, TextQualityAgent, TextSafetyAgent,
    VideoGenerator, VideoQualityModel, VideoSafetyModel,
)
from vocagent.selection import HumanInterventionRequired


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name).resolve() / "result"
        text_counts = {"outline": 0, "storyboard": 0}
        text_lock = Lock()

        def generate_text(stage):
            with text_lock:
                text_counts[stage] += 1
                return f"{stage} {text_counts[stage]}"

        self.llm = Mock(spec=LLMGenerator)
        self.llm.generate_outline.side_effect = (
            lambda vocabulary, feedback: generate_text("outline")
        )
        self.llm.generate_storyboard.side_effect = (
            lambda vocabulary, outline, feedback: generate_text("storyboard")
        )
        self.text_safety = Mock(spec=TextSafetyAgent)
        self.text_safety.check.return_value = SafetyResult(True)
        self.text_quality = Mock(spec=TextQualityAgent)
        self.text_quality.choose.side_effect = (
            lambda stage, vocabulary, candidates: 1 if stage == "outline" else 0
        )
        self.video = Mock(spec=VideoGenerator)
        self.video.generate.side_effect = (
            lambda storyboard, output_path: output_path.write_bytes(b"fake video")
        )
        self.video_safety = Mock(spec=VideoSafetyModel)
        self.video_safety.check.return_value = SafetyResult(True)
        self.video_quality = Mock(spec=VideoQualityModel)
        self.video_quality.choose.return_value = 1
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
        quality_calls = self.text_quality.choose.call_args_list
        self.assertEqual(len(quality_calls), 2)
        outline_call, storyboard_call = quality_calls
        self.assertEqual(outline_call.args[:2], ("outline", words))
        self.assertEqual(storyboard_call.args[:2], ("storyboard", words))
        outlines = outline_call.args[2]
        storyboards = storyboard_call.args[2]
        self.assertCountEqual(outlines, [f"outline {number}" for number in range(1, 3)])
        self.assertCountEqual(storyboards, [f"storyboard {number}" for number in range(1, 3)])
        chosen_outline = outlines[1]
        chosen_storyboard = storyboards[0]
        self.assertEqual(self.llm.generate_outline.call_args_list, [call(words, None)] * 2)
        self.assertEqual(
            self.llm.generate_storyboard.call_args_list,
            [call(words, chosen_outline, None)] * 2,
        )
        videos = tuple(
            self.output / "video_candidates" / f"candidate-{slot}-attempt-1.mp4"
            for slot in range(1, 3)
        )
        self.assertCountEqual(
            self.video.generate.call_args_list,
            [call(chosen_storyboard, path) for path in videos],
        )
        self.video_quality.choose.assert_called_once_with(chosen_storyboard, videos)
        self.assertEqual(result.selected_video, videos[1])
        self.media.extract_audio.assert_called_once_with(videos[1], result.audio)
        self.speech_to_text.transcribe.assert_called_once_with(result.audio)
        self.media.burn_subtitles.assert_called_once_with(
            videos[1], result.subtitles, result.final_video,
        )
        self.assertEqual(result.outline.read_text(), chosen_outline)
        self.assertEqual(result.storyboard.read_text(), chosen_storyboard)
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

    def test_generation_and_safety_run_sequentially_by_default(self):
        caller = current_thread()
        events = []
        for label, method in (
            ("outline", self.llm.generate_outline),
            ("storyboard", self.llm.generate_storyboard),
            ("video", self.video.generate),
            ("text safety", self.text_safety.check),
            ("video safety", self.video_safety.check),
        ):
            original = method.side_effect
            result = method.return_value

            def record(*args, label=label, original=original, result=result):
                self.assertIs(current_thread(), caller)
                events.append(label)
                return original(*args) if original is not None else result

            method.side_effect = record

        self.text_quality.choose.side_effect = (
            lambda stage, vocabulary, candidates: len(candidates) - 1
        )
        self.video_quality.choose.side_effect = (
            lambda storyboard, candidates: len(candidates) - 1
        )
        for counts in ({}, {
            "outline_candidate_count": 1,
            "storyboard_candidate_count": 3,
            "video_candidate_count": 5,
        }):
            with self.subTest(counts=counts):
                events.clear()
                workflow = Workflow(self.providers, self.media, **counts)
                workflow.run(["apple"], self.output / ("configured" if counts else "default"))

                self.assertEqual(events, (
                    ["outline", "text safety"] * counts.get("outline_candidate_count", 2)
                    + ["storyboard", "text safety"] * counts.get("storyboard_candidate_count", 2)
                    + ["video", "video safety"] * counts.get("video_candidate_count", 2)
                ))

    def test_both_default_generations_overlap_in_each_stage_when_enabled(self):
        for generator in (
            self.llm.generate_outline, self.llm.generate_storyboard, self.video.generate,
        ):
            barrier = Barrier(2, timeout=5)
            original = generator.side_effect

            def concurrent_generate(*args, barrier=barrier, original=original):
                barrier.wait()
                return original(*args)

            generator.side_effect = concurrent_generate

        workflow = Workflow(self.providers, self.media, concurrent_candidates=True)
        result = workflow.run(["apple"], self.output)

        self.assertTrue(result.final_video.is_file())
        self.assertEqual(self.llm.generate_outline.call_count, 2)
        self.assertEqual(self.llm.generate_storyboard.call_count, 2)
        self.assertEqual(self.video.generate.call_count, 2)

    def test_candidate_counts_are_independent_and_control_stage_concurrency(self):
        self.text_quality.choose.side_effect = (
            lambda stage, vocabulary, candidates: len(candidates) - 1
        )
        self.video_quality.choose.side_effect = (
            lambda storyboard, candidates: len(candidates) - 1
        )
        for generator, count in (
            (self.llm.generate_outline, 1),
            (self.llm.generate_storyboard, 3),
            (self.video.generate, 5),
        ):
            barrier = Barrier(count, timeout=5)
            original = generator.side_effect

            def concurrent_generate(*args, barrier=barrier, original=original):
                barrier.wait()
                return original(*args)

            generator.side_effect = concurrent_generate

        workflow = Workflow(
            self.providers, self.media,
            outline_candidate_count=1,
            storyboard_candidate_count=3,
            video_candidate_count=5,
            concurrent_candidates=True,
        )
        result = workflow.run(["apple"], self.output)

        self.assertEqual(self.llm.generate_outline.call_count, 1)
        self.assertEqual(self.llm.generate_storyboard.call_count, 3)
        self.assertEqual(self.video.generate.call_count, 5)
        outline_call, storyboard_call = self.text_quality.choose.call_args_list
        self.assertEqual(len(outline_call.args[2]), 1)
        self.assertEqual(len(storyboard_call.args[2]), 3)
        self.assertEqual(result.outline.read_text(), outline_call.args[2][-1])
        self.assertEqual(result.storyboard.read_text(), storyboard_call.args[2][-1])
        videos = self.video_quality.choose.call_args.args[1]
        self.assertEqual(
            [video.name for video in videos],
            [f"candidate-{slot}-attempt-1.mp4" for slot in range(1, 6)],
        )
        self.assertEqual(result.selected_video, videos[-1])
        self.assertTrue(result.final_video.is_file())

    def test_invalid_candidate_counts_are_rejected_before_starting_work(self):
        for name in (
            "outline_candidate_count", "storyboard_candidate_count", "video_candidate_count",
        ):
            for count in (0, -1, True, False, 1.5, "4", None):
                with self.subTest(name=name, count=count):
                    with self.assertRaisesRegex(ValueError, f"{name}.*positive integer"):
                        Workflow(self.providers, self.media, **{name: count})
        self.assertFalse(self.output.exists())
        self.llm.generate_outline.assert_not_called()
        self.llm.generate_storyboard.assert_not_called()
        self.video.generate.assert_not_called()

    def test_text_safety_feedback_reaches_only_the_retried_stage(self):
        self.text_safety.check.side_effect = (
            lambda stage, text: SafetyResult(text != "outline 1", "remove unsafe content")
        )

        result = self.workflow.run(["apple"], self.output)

        self.assertCountEqual(
            self.llm.generate_outline.call_args_list,
            [call(("apple",), None)] * 2 + [call(("apple",), "remove unsafe content")],
        )
        outlines = self.text_quality.choose.call_args_list[0].args[2]
        self.assertCountEqual(outlines, [f"outline {number}" for number in range(2, 4)])
        chosen_outline = outlines[1]
        self.assertEqual(result.outline.read_text(), chosen_outline)
        self.assertEqual(
            self.llm.generate_storyboard.call_args_list,
            [call(("apple",), chosen_outline, None)] * 2,
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

        result = self.workflow.run(["apple"], self.output)

        self.assertFalse(rejected.exists())
        self.assertEqual(self.video.generate.call_count, 3)
        self.assertTrue(all(len(item.args) == 2 for item in self.video.generate.call_args_list))
        self.assertTrue(all(item.args[0] == result.storyboard.read_text()
                            for item in self.video.generate.call_args_list))
        survivors = self.video_quality.choose.call_args.args[1]
        self.assertEqual(len(survivors), 2)
        self.assertNotIn(rejected, survivors)
        self.assertEqual(survivors[0].name, "candidate-1-attempt-2.mp4")

    def test_all_unsafe_outline_stops_before_storyboard(self):
        self.text_safety.check.return_value = SafetyResult(False, "unsafe outline")

        with self.assertRaisesRegex(HumanInterventionRequired, "outline"):
            self.workflow.run(["apple"], self.output)

        self.assertEqual(self.llm.generate_outline.call_count, 8)
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

        self.assertEqual(self.llm.generate_storyboard.call_count, 8)
        self.assertEqual(self.text_quality.choose.call_count, 1)
        self.video.generate.assert_not_called()
        self.media.extract_audio.assert_not_called()

    def test_all_unsafe_videos_are_removed_and_stop_before_audio(self):
        self.video_safety.check.return_value = SafetyResult(False, "unsafe video")

        with self.assertRaisesRegex(HumanInterventionRequired, "video"):
            self.workflow.run(["apple"], self.output)

        self.assertEqual(self.video.generate.call_count, 8)
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
