"""HTTP contract and fail-closed tests for the local Qwen text providers."""

from concurrent.futures import ThreadPoolExecutor
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from vocagent.models import SafetyResult
from vocagent.providers import SpeechToText, VideoGenerator
from vocagent.qwen import QwenChatClient, QwenEndpoint, QwenError, create_providers


def completion(content, *, finish_reason="stop", **message_fields):
    return {"choices": [{
        "finish_reason": finish_reason,
        "message": {"role": "assistant", "content": content, **message_fields},
    }]}


def response(data):
    return io.BytesIO(data if isinstance(data, bytes) else json.dumps(data).encode("utf-8"))


class QwenTests(unittest.TestCase):
    def setUp(self):
        self.providers = create_providers({})
        self.transport = patch("vocagent.qwen.urlopen").start()
        self.addCleanup(patch.stopall)

    def reply(self, content, **kwargs):
        self.transport.return_value = response(completion(content, **kwargs))

    def sent(self):
        request = self.transport.call_args.args[0]
        return request, json.loads(request.data)

    def test_defaults_and_independent_clients_leave_later_stages_explicit(self):
        providers = self.providers
        self.assertEqual(providers.llm.client.endpoint.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(providers.llm.client.endpoint.model, "qwen3")
        self.assertEqual(providers.text_safety.client.endpoint.base_url, "http://127.0.0.1:8001/v1")
        self.assertEqual(providers.text_safety.client.endpoint.model, "qwen3-guard")
        self.assertIsNot(providers.llm.client, providers.text_quality.client)
        self.assertIs(type(providers.video), VideoGenerator)
        self.assertIs(type(providers.speech_to_text), SpeechToText)
        with self.assertRaises(NotImplementedError):
            providers.video.generate("scenes", Path("unused.mp4"))

    def test_outline_http_request_carries_unicode_feedback_and_no_thinking(self):
        providers = create_providers({
            "QWEN_BASE_URL": "http://model.test:9000/v1/", "QWEN_MODEL": "my-qwen3",
            "QWEN_API_KEY": "test-key", "QWEN_TIMEOUT_SECONDS": "15.5",
        })
        self.reply("  The story outline  ")
        self.assertEqual(
            providers.llm.generate_outline(["apple", "你好"], "Remove violence."),
            "The story outline",
        )
        request, data = self.sent()
        self.assertEqual(request.full_url, "http://model.test:9000/v1/chat/completions")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(self.transport.call_args.kwargs["timeout"], 15.5)
        self.assertEqual(data["model"], "my-qwen3")
        self.assertEqual(data["chat_template_kwargs"], {"enable_thinking": False})
        self.assertFalse(data["stream"])
        self.assertGreater(data["temperature"], 0)
        self.assertEqual(json.loads(data["messages"][1]["content"]), {
            "vocabulary": ["apple", "你好"], "safety_feedback": "Remove violence.",
        })
        self.assertIn("safety feedback", data["messages"][0]["content"])
        self.assertNotIn("test-key", repr(providers.llm.client.endpoint))
        self.assertNotIn("stop", data)

    def test_storyboard_contains_selected_outline_and_spoken_word_instruction(self):
        self.reply("Scene 1: Hello")
        self.providers.llm.generate_storyboard(["hello"], "A friend arrives.", None)
        request, data = self.sent()
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(json.loads(data["messages"][1]["content"]), {
            "vocabulary": ["hello"], "outline": "A friend arrives.",
        })
        self.assertIn("Speak every vocabulary item", data["messages"][0]["content"])

    def test_generation_calls_do_not_share_conversation_state(self):
        def serve(request, **kwargs):
            messages = json.loads(request.data)["messages"]
            self.assertEqual(len(messages), 2)
            vocabulary = json.loads(messages[1]["content"])["vocabulary"]
            return response(completion(vocabulary[0]))

        self.transport.side_effect = serve
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                lambda word: self.providers.llm.generate_outline([word], None),
                ["one", "two", "three", "four"],
            ))
        self.assertEqual(results, ["one", "two", "three", "four"])

    def test_guard_uses_native_response_template_without_custom_system_prompt(self):
        self.reply("Safety: Safe\nCategories: None\nRefusal: No")
        self.assertEqual(self.providers.text_safety.check("outline", "A nice story."), SafetyResult(True))
        request, data = self.sent()
        self.assertEqual(request.full_url, "http://127.0.0.1:8001/v1/chat/completions")
        self.assertEqual(data["model"], "qwen3-guard")
        self.assertEqual(data["messages"], [
            {"role": "user", "content": "Write a vocabulary-learning story outline."},
            {"role": "assistant", "content": "A nice story."},
        ])
        self.assertNotIn("chat_template_kwargs", data)
        self.assertNotIn("response_format", data)
        self.assertEqual(data["temperature"], 0)
        self.assertNotIn("stop", data)

    def test_guard_blocks_unsafe_and_controversial_with_useful_regeneration_feedback(self):
        for label in ("Unsafe", "Controversial"):
            with self.subTest(label=label):
                self.reply(f"Safety: {label}\nCategories: Violent, Unethical Acts\nRefusal: No")
                result = self.providers.text_safety.check("storyboard", "candidate")
                self.assertIs(result.safe, False)
                self.assertIn(label, result.reason)
                self.assertIn("Violent", result.reason)

    def test_guard_accepts_safe_refusal_as_safe_not_a_different_safety_policy(self):
        self.reply("Safety: Safe\nCategories: None\nRefusal: Yes")
        self.assertTrue(self.providers.text_safety.check("outline", "I cannot help.").safe)

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
                    self.providers.text_safety.check("outline", "candidate")

    def test_quality_uses_independent_json_judge_and_only_safe_survivor_indexes(self):
        self.reply('{"index": 1}')
        result = self.providers.text_quality.choose("storyboard", ["apple"], ["first", "second"])
        self.assertEqual(result, 1)
        _, data = self.sent()
        self.assertEqual(data["response_format"], {"type": "json_object"})
        self.assertEqual(data["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(json.loads(data["messages"][1]["content"]), {
            "stage": "storyboard", "vocabulary": ["apple"],
            "candidates": [{"index": 0, "text": "first"}, {"index": 1, "text": "second"}],
        })

    def test_quality_judge_still_runs_for_one_survivor(self):
        self.reply('{"index": 0}')
        self.assertEqual(self.providers.text_quality.choose("outline", ["word"], ["one"]), 0)
        self.transport.assert_called_once()

    def test_quality_rejects_ambiguous_noninteger_and_out_of_range_indexes(self):
        for content in (
            "1", "[]", "null", '{"index": true}', '{"index": 0.0}', '{"index": "0"}',
            '{"index": -1}', '{"index": 2}', '{"index": 0, "reason": "best"}',
            '{"index": 0, "index": 1}', '```json\n{"index": 0}\n```',
        ):
            with self.subTest(content=content):
                self.reply(content)
                with self.assertRaises(QwenError):
                    self.providers.text_quality.choose("outline", ["word"], ["a", "b"])

    def test_no_provider_accepts_truncated_output(self):
        for invoke, content in (
            (lambda: self.providers.llm.generate_outline(["word"], None), "unfinished"),
            (lambda: self.providers.text_safety.check("outline", "candidate"),
             "Safety: Safe\nCategories: None\nRefusal: No"),
            (lambda: self.providers.text_quality.choose("outline", ["word"], ["a"]), '{"index": 0}'),
        ):
            with self.subTest(content=content):
                self.reply(content, finish_reason="length")
                with self.assertRaisesRegex(QwenError, "truncated"):
                    invoke()

    def test_invalid_http_responses_fail_closed(self):
        for data in (
            b"not json", b"\xff", [], {"error": {"message": "failed"}}, {},
            {"choices": []}, {"choices": [None]},
            {"choices": [completion("a")["choices"][0]] * 2},
            completion(None), completion(""), completion("   "), completion([{"text": "hello"}]),
            completion("hello", role="user"), completion("hello", tool_calls=[{}]),
            completion("hello", refusal="refused"), completion("hello", finish_reason="tool_calls"),
        ):
            with self.subTest(data=data):
                self.transport.return_value = response(data)
                with self.assertRaises(QwenError):
                    self.providers.text_safety.check("outline", "candidate")

    def test_transport_errors_raise_without_fabricated_verdict_or_retry(self):
        for error in (
            HTTPError("http://model.test", 503, "Unavailable", {}, None),
            URLError("connection refused"), TimeoutError("timed out"),
            IncompleteRead(b"partial"),
        ):
            with self.subTest(error=error):
                self.transport.reset_mock()
                self.transport.side_effect = error
                with self.assertRaises(QwenError):
                    self.providers.text_safety.check("outline", "candidate")
                self.transport.assert_called_once()

    def test_guard_credentials_and_timeout_do_not_inherit_generation_secrets(self):
        providers = create_providers({
            "QWEN_API_KEY": "generation-secret", "QWEN_TIMEOUT_SECONDS": "1",
            "QWEN_GUARD_API_KEY": "guard-secret", "QWEN_GUARD_TIMEOUT_SECONDS": "9",
        })
        self.reply("Safety: Safe\nCategories: None\nRefusal: No")
        providers.text_safety.check("outline", "candidate")
        request, _ = self.sent()
        self.assertEqual(request.get_header("Authorization"), "Bearer guard-secret")
        self.assertEqual(self.transport.call_args.kwargs["timeout"], 9)

    def test_bad_configuration_fails_before_request(self):
        for env in (
            {"QWEN_BASE_URL": "file:///tmp/model"},
            {"QWEN_BASE_URL": "http://user:password@model.test/v1"},
            {"QWEN_BASE_URL": "http://model.test/v1?key=secret"},
            {"QWEN_MODEL": " "},
            *({"QWEN_TIMEOUT_SECONDS": value} for value in ("0", "-1", "NaN", "inf", "bad")),
            {"QWEN_GUARD_TIMEOUT_SECONDS": "0"},
        ):
            with self.subTest(env=env), self.assertRaises(ValueError):
                create_providers(env)
        self.transport.assert_not_called()

    def test_invalid_stage_and_candidates_fail_before_request(self):
        with self.assertRaises(ValueError):
            self.providers.text_safety.check("video", "candidate")
        with self.assertRaises(ValueError):
            self.providers.text_safety.check("outline", " ")
        for candidates in ([], [""], [None], "candidate"):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                self.providers.text_quality.choose("outline", ["word"], candidates)
        self.transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
