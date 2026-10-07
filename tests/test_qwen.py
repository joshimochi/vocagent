"""Native model message contracts and fail-closed Qwen agent tests."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from vocagent.models import SafetyResult
from vocagent.media import MediaError
from vocagent.qwen import (
    ChatClient, QwenAudioSafety, QwenError, QwenGenerator, QwenGuard, QwenQuality,
    QwenVideoQuality,
)
from vocagent.video_frames import VideoFrame


class QwenTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock(spec=ChatClient)
        self.complete = self.client.complete
        self.generator = QwenGenerator(self.client, video_prompt_seconds=10)
        self.guard = QwenGuard(self.client)
        self.quality = QwenQuality(self.client)

    def reply(self, content):
        self.complete.side_effect = None
        self.complete.return_value = content

    def sent(self):
        call = self.complete.call_args
        return {"messages": call.args[0], **call.kwargs}

    def test_outline_keeps_constraints_in_system_and_target_feedback_in_user_message(self):
        self.reply("The story outline")
        self.assertEqual(
            self.generator.generate_outline([" apple "], "Remove violence."),
            "The story outline",
        )
        data = self.sent()
        self.assertTrue(data["non_thinking"])
        self.assertGreater(data["temperature"], 0)
        self.assertEqual(json.loads(data["messages"][1]["content"]), {
            "target_word": "apple",
            "safety_feedback": "Remove violence.",
        })
        self.assertIn("safety feedback", data["messages"][0]["content"])
        self.assertIn("cartoon animation", data["messages"][0]["content"])
        self.assertIn("10-second video", data["messages"][0]["content"])
        self.assertIn("exactly one supplied English target word", data["messages"][0]["content"])
        self.assertIn("one clear meaning throughout", data["messages"][0]["content"])
        self.assertIn("Write all text in English", data["messages"][0]["content"])
        self.assertIn("Repeat the target word naturally", data["messages"][0]["content"])
        self.assertNotIn("# Video Prompt Writing Guide", data["messages"][0]["content"])
        self.assertNotIn("integrated_multimodal_description", data["messages"][0]["content"])

    def test_generation_calls_do_not_share_conversation_state(self):
        def serve(messages, **kwargs):
            self.assertEqual(len(messages), 2)
            return json.loads(messages[1]["content"])["target_word"]

        self.complete.side_effect = serve
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                lambda word: self.generator.generate_outline([word], None),
                ["one", "two", "three", "four"],
            ))
        self.assertEqual(results, ["one", "two", "three", "four"])

    def test_local_video_prompt_includes_h3_guide_and_returns_response_in_one_call(self):
        video_prompt = (
            "integrated_multimodal_description: [Shot 1] 2D cartoon animation, "
            "a friend with a gentle, clear voice (S1) catches an apple and says: "
            "<d>[English] An apple! Catch the apple!</d>\n\n"
            "overall_soundscape: Leaves rustle.\n\n"
            "non_diegetic_music: N/A"
        )
        self.reply(video_prompt)
        self.assertEqual(self.generator.generate_video_prompt(
            [" apple "], "A friend catches an apple.", "Use a gentle scene.",
        ), video_prompt)
        data = self.sent()
        prompt = data["messages"][0]["content"]
        guide = (Path(__file__).resolve().parents[1] / "vocagent" / "prompts" /
                 "h3_prompt_guide.md").read_text(encoding="utf-8")
        self.assertIn(guide, prompt)
        self.assertIn("video prompt for a short vocabulary-learning video", prompt)
        self.assertIn("State the cartoon animation style in the video prompt", prompt)
        self.assertIn("scene timing, visuals/action, spoken dialogue or narration", prompt)
        self.assertIn("MiniMax H3", prompt)
        self.assertIn("integrated_multimodal_description", prompt)
        self.assertIn("overall_soundscape", prompt)
        self.assertIn("non_diegetic_music", prompt)
        self.assertIn("<d>[English]", prompt)
        self.assertIn("hard duration of 10 seconds", prompt)
        self.assertIn("at most 20 spoken English words TOTAL", prompt)
        self.assertIn("normal speaking pace", prompt)
        self.assertIn("Speak the target word unchanged", prompt)
        self.assertIn("repeat it naturally", prompt)
        self.assertIn("one clear meaning", prompt)
        self.assertIn("Write all instructions, dialogue and narration in English", prompt)
        self.assertNotIn("other languages", prompt)
        self.assertEqual(json.loads(data["messages"][1]["content"]), {
            "target_word": "apple", "outline": "A friend catches an apple.",
            "safety_feedback": "Use a gentle scene.",
        })
        for supplied_data in ("apple", "A friend catches an apple.", "Use a gentle scene."):
            self.assertNotIn(supplied_data, prompt)
        self.assertEqual([message["role"] for message in data["messages"]], ["system", "user"])
        self.assertEqual(data["max_tokens"], 2048)
        self.assertEqual(data["temperature"], 0.7)
        self.assertTrue(data["non_thinking"])
        self.complete.assert_called_once()

    def test_local_video_prompt_never_retries_backend_or_truncation_errors(self):
        for failure in (QwenError("GPU inference failed"), QwenError("truncated output")):
            with self.subTest(failure=failure):
                self.complete.reset_mock()
                self.complete.side_effect = failure
                with self.assertRaises(QwenError) as caught:
                    self.generator.generate_video_prompt(["hello"], "A friend arrives.", None)
                self.assertIs(caught.exception, failure)
                self.complete.assert_called_once()

    def test_video_prompt_duration_is_required_and_rejects_invalid_limits(self):
        for seconds in (None, 0, -1, True, float("nan"), float("inf"), "10"):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                QwenGenerator(self.generator.client, video_prompt_seconds=seconds)
        with self.assertRaises(TypeError):
            QwenGenerator(self.client)
        self.complete.assert_not_called()

    def test_guard_uses_native_response_template_without_custom_system_prompt(self):
        self.reply("Safety: Safe\nCategories: None\nRefusal: No")
        self.assertEqual(self.guard.check("outline", "A nice story."), SafetyResult(True))
        self.assertEqual(self.sent(), {
            "messages": [
                {"role": "user", "content": "Write a vocabulary-learning story outline."},
                {"role": "assistant", "content": "A nice story."},
            ],
            "max_tokens": 128, "temperature": 0,
        })

    def test_guard_blocks_unsafe_and_controversial_with_useful_regeneration_feedback(self):
        for label in ("Unsafe", "Controversial"):
            with self.subTest(label=label):
                self.reply(f"Safety: {label}\nCategories: Violent, Unethical Acts\nRefusal: No")
                result = self.guard.check("video_prompt", "candidate")
                self.assertIs(result.safe, False)
                self.assertIn(label, result.reason)
                self.assertIn("Violent", result.reason)

    def test_guard_accepts_safe_refusal_as_safe_not_a_different_safety_policy(self):
        self.reply("Safety: Safe\nCategories: None\nRefusal: Yes")
        self.assertTrue(self.guard.check("outline", "I cannot help.").safe)

    def test_guard_malformed_or_conflicting_labels_never_pass(self):
        invalid = [
            "Safe", "Safety: Safe", "Safety: Safe\nCategories: None",
            "Safety: safe\nCategories: None\nRefusal: No",
            "Safety: Safe\nCategories: None\nRefusal: Maybe",
            "Safety: Safe\nCategories: None\nRefusal: No\nSafety: Unsafe",
            "Quoted verdict: Safety: Safe\nCategories: None\nRefusal: No",
            "Safety: Safe\nCategories: Violent\nRefusal: No",
            "Safety: Unsafe\nCategories: None\nRefusal: No",
            "Safety: Unsafe\nCategories: Unrecognized category\nRefusal: No",
            "Safety: Unsafe\nCategories: Violent, None\nRefusal: No",
            "Safety: Unsafe\nCategories: Violent, Violent\nRefusal: No",
            "Safety: Unsafe\nCategories: Jailbreak\nRefusal: No",
            '{"safe": true}',
        ]
        for content in invalid:
            with self.subTest(content=content):
                self.reply(content)
                with self.assertRaises(QwenError):
                    self.guard.check("outline", "candidate")

    def test_quality_judge_receives_only_safe_survivor_indexes(self):
        self.reply('{"index": 1}')
        result = self.quality.choose("video_prompt", ["apple"], ["first", "second"])
        self.assertEqual(result, 1)
        data = self.sent()
        self.assertTrue(data["non_thinking"])
        self.assertEqual(json.loads(data["messages"][1]["content"]), {
            "stage": "video_prompt", "target_word": "apple",
            "candidates": [{"index": 0, "text": "first"}, {"index": 1, "text": "second"}],
        })
        prompt = data["messages"][0]["content"]
        self.assertIn("cartoon animation", prompt)
        self.assertIn("exactly one supplied English target word", prompt)
        self.assertIn("one clear meaning throughout", prompt)
        self.assertIn("non-English text", prompt)
        self.assertIn("additional vocabulary-learning objectives", prompt)

    def test_quality_judge_still_runs_for_one_survivor(self):
        self.reply('{"index": 0}')
        self.assertEqual(self.quality.choose("outline", ["word"], ["one"]), 0)
        self.complete.assert_called_once()

    def test_quality_rejects_ambiguous_noninteger_and_out_of_range_indexes(self):
        for content in (
            "1", "[]", "null", '{"index": true}', '{"index": 0.0}', '{"index": "0"}',
            '{"index": -1}', '{"index": 2}', '{"index": 0, "reason": "best"}',
            '{"index": 0, "index": 1}', '```json\n{"index": 0}\n```',
        ):
            with self.subTest(content=content):
                self.reply(content)
                with self.assertRaises(QwenError):
                    self.quality.choose("outline", ["word"], ["a", "b"])

    def test_agent_backend_errors_fail_without_retry(self):
        for invoke in (
            lambda: self.generator.generate_outline(["word"], None),
            lambda: self.guard.check("outline", "candidate"),
            lambda: self.quality.choose("outline", ["word"], ["a"]),
            lambda: QwenAudioSafety(self.client).check("spoken content"),
        ):
            with self.subTest(invoke=invoke):
                self.complete.reset_mock()
                self.complete.side_effect = QwenError("Local completion truncated")
                with self.assertRaisesRegex(QwenError, "truncated"):
                    invoke()
                self.complete.assert_called_once()

    def test_invalid_stage_and_candidates_fail_before_inference(self):
        with self.assertRaises(ValueError):
            self.guard.check("video", "candidate")
        with self.assertRaises(ValueError):
            self.guard.check("outline", " ")
        for candidates in ([], [""], [None], "candidate"):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                self.quality.choose("outline", ["word"], candidates)
        self.complete.assert_not_called()

    def test_generation_and_selection_reject_multiple_or_nonenglish_targets_before_inference(self):
        for vocabulary in (
            [], ["apple", "pear"], ["你好"], ["ice cream"], [""], ["apple!"], [None],
        ):
            for invoke in (
                lambda: self.generator.generate_outline(vocabulary, None),
                lambda: self.generator.generate_video_prompt(vocabulary, "An outline.", None),
                lambda: self.quality.choose("outline", vocabulary, ["A story."]),
            ):
                with self.subTest(vocabulary=vocabulary, invoke=invoke), self.assertRaises(ValueError):
                    invoke()
        self.complete.assert_not_called()

    def test_transcript_guard_checks_all_spoken_content_using_native_template(self):
        transcript = "The first scene is gentle.\nThe last scene contains bad content."
        self.reply("Safety: Unsafe\nCategories: Violent\nRefusal: No")
        safety = QwenAudioSafety(self.guard.client)
        result = safety.check(transcript)
        self.assertFalse(result.safe)
        data = self.sent()
        self.assertEqual(data["messages"][1], {"role": "assistant", "content": transcript})
        self.assertIn("spoken dialogue", data["messages"][0]["content"])
        self.assertNotIn("non_thinking", data)
        self.reply("Safe")
        with self.assertRaises(QwenError):
            safety.check(transcript)
        self.complete.reset_mock()
        with self.assertRaises(ValueError):
            safety.check(" ")
        self.complete.assert_not_called()

    def test_video_judge_receives_actual_ordered_pixels_for_each_candidate(self):
        judge = QwenVideoQuality(self.quality.client, max_frames=2, max_side=448)
        frames = [
            (VideoFrame(0, b"first-start"), VideoFrame(9, b"first-end")),
            (VideoFrame(0, b"second-start"), VideoFrame(5, b"second-end")),
        ]
        self.reply('{"index": 1}')
        with patch("vocagent.qwen.sample_video_frames", side_effect=frames) as sampler:
            self.assertEqual(judge.choose("A clear video prompt.", [Path("a.mp4"), Path("b.mp4")]), 1)
        self.assertEqual(sampler.call_count, 2)
        self.assertEqual(sampler.call_args_list[0].kwargs, {"max_frames": 2, "max_side": 448})
        data = self.sent()
        self.assertTrue(data["non_thinking"])
        content = data["messages"][1]["content"]
        self.assertEqual(json.loads(content[0]["text"]), {
            "video_prompt": "A clear video prompt.",
        })
        self.assertIn("cartoon animation", data["messages"][0]["content"])
        self.assertIn("exactly one English target word", data["messages"][0]["content"])
        images = [block["image"] for block in content if block["type"] == "image"]
        self.assertEqual(images, [frame.jpeg for group in frames for frame in group])
        captions = [block["text"] for block in content if block["type"] == "text"]
        self.assertIn("Candidate 0, time 9.000 seconds:", captions)
        self.assertIn("Candidate 1, time 5.000 seconds:", captions)

    def test_video_judge_runs_for_one_survivor_and_rejects_invalid_decisions(self):
        judge = QwenVideoQuality(self.quality.client)
        with patch("vocagent.qwen.sample_video_frames", return_value=(VideoFrame(0, b"image"),)):
            self.reply('{"index": 0}')
            self.assertEqual(judge.choose("story", [Path("a.mp4")]), 0)
            for content in ('{"index": true}', '{"index": 1}', '{"index": 0, "index": 0}', "0"):
                with self.subTest(content=content), self.assertRaises(QwenError):
                    self.reply(content)
                    judge.choose("story", [Path("a.mp4")])
            self.complete.side_effect = QwenError("Local completion truncated")
            with self.assertRaisesRegex(QwenError, "truncated"):
                judge.choose("story", [Path("a.mp4")])

    def test_video_sampling_failure_never_becomes_a_text_only_judgement(self):
        judge = QwenVideoQuality(self.quality.client)
        with patch("vocagent.qwen.sample_video_frames", side_effect=MediaError("invalid video")):
            with self.assertRaises(MediaError):
                judge.choose("story", [Path("a.mp4")])
        self.complete.assert_not_called()

    def test_video_judge_rejects_invalid_inputs_and_excessive_frame_budget(self):
        judge = QwenVideoQuality(self.quality.client)
        for candidates in ([], "movie.mp4", Path("movie.mp4"), [None], [Path("a")] * 3):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                judge.choose("story", candidates)
        with self.assertRaises(ValueError):
            judge.choose(" ", [Path("a")])
        for options in ({"max_frames": 0}, {"max_frames": True}, {"max_frames": 65}, {"max_side": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                QwenVideoQuality(self.quality.client, **options)
        self.complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
