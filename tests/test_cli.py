"""Verify the CLI boundary without calling external models or FFmpeg."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vocagent.__main__ import load_providers, main
from vocagent.models import SafetyResult, VideoPromptResult
from vocagent.providers import (
    AudioSafetyModel, LLMGenerator, Providers, SpeechToText, TextQualityAgent,
    TextSafetyAgent, VideoGenerator, VideoQualityModel, VideoSafetyModel,
)


def providers_for_test(**overrides):
    dependencies = {
        "llm": Mock(spec=LLMGenerator),
        "text_safety": Mock(spec=TextSafetyAgent),
        "text_quality": Mock(spec=TextQualityAgent),
        "video": Mock(spec=VideoGenerator),
        "video_safety": Mock(spec=VideoSafetyModel),
        "video_quality": Mock(spec=VideoQualityModel),
        "speech_to_text": Mock(spec=SpeechToText),
        "audio_safety": Mock(spec=AudioSafetyModel),
    }
    dependencies.update(overrides)
    return Providers(**dependencies)


class CLITests(unittest.TestCase):
    def test_invalid_word_is_rejected_before_providers_or_output(self):
        for words in (
            [], ["apple", "pear"], ["apple", "apple"], ["green pear"],
            ["你好"], [""], ["123"], ["apple!"],
        ):
            with self.subTest(words=words), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "result"
                with patch("vocagent.__main__.load_providers") as load, \
                        patch("vocagent.__main__.Workflow") as workflow, \
                        redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as raised:
                    main([*words, "--output-dir", str(output)])
                self.assertEqual(raised.exception.code, 2)
                load.assert_not_called()
                workflow.assert_not_called()
                self.assertFalse(output.exists())

    def test_default_provider_configuration_failure_is_a_clear_cli_error(self):
        stderr = io.StringIO()
        stdout = io.StringIO()
        with patch("vocagent.__main__.load_providers",
                   side_effect=ValueError("Configure the local model checkpoints")) as load, \
                redirect_stderr(stderr), redirect_stdout(stdout):
            status = main(["apple", "--stop-after", "video-prompt"])

        load.assert_called_once_with("vocagent.local:create_providers")
        self.assertEqual(status, 1)
        self.assertIn("Workflow stopped: Configure the local model checkpoints", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")

    def test_local_factory_explicit_stop_at_video_prompt(self):
        llm = Mock(spec=LLMGenerator)
        llm.generate_outline.return_value = "A story about an apple."
        video_prompt = (
            'Scene 1 (0–10 seconds): A friend catches an apple and says "Apple!"\n'
            "Audio: Leaves rustle."
        )
        llm.generate_video_prompt.return_value = video_prompt
        safety = Mock(spec=TextSafetyAgent)
        safety.check.return_value = SafetyResult(True)
        quality = Mock()
        quality.choose.return_value = 0
        providers = providers_for_test(llm=llm, text_safety=safety, text_quality=quality)
        with tempfile.TemporaryDirectory() as temporary:
            stdout = io.StringIO()
            with patch("vocagent.__main__.load_providers", return_value=providers) as load, \
                    redirect_stdout(stdout):
                status = main(["apple", "--output-dir", temporary, "--stop-after", "video-prompt"])

            self.assertEqual(status, 0)
            load.assert_called_once_with("vocagent.local:create_providers")
            output = Path(temporary)
            self.assertEqual(stdout.getvalue(), f"{output / 'video_prompt.txt'}\n")
            self.assertEqual((output / "video_prompt.txt").read_text(), video_prompt)
            self.assertEqual({path.name for path in output.iterdir()}, {
                "vocabulary.json", "outline.txt", "video_prompt.txt",
                "outline_candidates", "video_prompt_candidates", "benchmark.json",
            })
            report = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["metadata"]["stop_after"], "video_prompt")
            for stage, text in (("outline", "A story about an apple."),
                                ("video_prompt", video_prompt)):
                self.assertEqual({
                    path.name: path.read_text(encoding="utf-8")
                    for path in (output / f"{stage}_candidates").iterdir()
                }, {
                    f"candidate-{slot}-attempt-1.txt": text for slot in (1, 2)
                })

    def test_default_local_factory_runs_full_video_and_releases_gpu_runtime(self):
        providers = providers_for_test()
        providers.close = Mock()
        final_path = Path("/tmp/local-final.mp4")
        stdout = io.StringIO()
        with patch("vocagent.__main__.load_providers", return_value=providers) as load, \
                patch("vocagent.__main__.Workflow") as workflow, redirect_stdout(stdout):
            workflow.return_value.run.return_value = SimpleNamespace(final_video=final_path)
            status = main(["apple", "--output-dir", "chosen-output"])
        self.assertEqual(status, 0)
        load.assert_called_once_with("vocagent.local:create_providers")
        workflow.return_value.run.assert_called_once_with(
            "apple", Path("chosen-output"), stop_after="video",
        )
        providers.close.assert_called_once_with()
        self.assertEqual(stdout.getvalue(), f"{final_path}\n")

    def test_local_runtime_is_released_when_gpu_inference_fails(self):
        providers = providers_for_test()
        providers.close = Mock()
        stderr = io.StringIO()
        with patch("vocagent.__main__.load_providers", return_value=providers), \
                patch("vocagent.__main__.Workflow") as workflow, redirect_stderr(stderr):
            workflow.return_value.run.side_effect = RuntimeError("GPU allocation failed")
            status = main(["apple"])
        self.assertEqual(status, 1)
        self.assertIn("GPU allocation failed", stderr.getvalue())
        providers.close.assert_called_once_with()

    def test_explicit_stop_stage_applies_to_each_provider(self):
        for factory_args in ([], ["--providers", "example:build"]):
            for stage in ("video_prompt", "video"):
                with self.subTest(factory_args=factory_args, stage=stage), \
                        patch("vocagent.__main__.load_providers", return_value=providers_for_test()), \
                        patch("vocagent.__main__.Workflow") as workflow, \
                        redirect_stdout(io.StringIO()):
                    workflow.return_value.run.return_value = (
                        VideoPromptResult(Path("outline.txt"), Path("video_prompt.txt"))
                        if stage == "video_prompt" else SimpleNamespace(final_video=Path("final.mp4"))
                    )
                    status = main(["apple", *factory_args, "--stop-after", stage.replace("_", "-")])

                self.assertEqual(status, 0)
                workflow.return_value.run.assert_called_once_with(
                    "apple", Path("output"), stop_after=stage,
                )

    def test_safety_exhaustion_reports_human_intervention_and_exit_two(self):
        llm = Mock(spec=LLMGenerator)
        llm.generate_outline.return_value = "unsafe outline"
        safety = Mock(spec=TextSafetyAgent)
        safety.check.return_value = SafetyResult(False, "unsafe content")
        providers = providers_for_test(llm=llm, text_safety=safety)
        with tempfile.TemporaryDirectory() as temporary:
            stderr = io.StringIO()
            with patch("vocagent.__main__.load_providers", return_value=providers), \
                    redirect_stderr(stderr):
                status = main(["apple", "--providers", "example:build", "--output-dir", temporary])
            report = json.loads((Path(temporary) / "benchmark.json").read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertIn("Human intervention required:", stderr.getvalue())
        self.assertIn("outline", stderr.getvalue())
        self.assertEqual(llm.generate_outline.call_count, 8)
        self.assertEqual(report["status"], "failed")
        self.assertIn("human intervention", report["error"]["message"])

    def test_custom_provider_defaults_to_video_and_prints_final_path(self):
        providers = providers_for_test()
        final_path = Path("/tmp/example-final.mp4")
        stdout = io.StringIO()
        with patch("vocagent.__main__.load_providers", return_value=providers) as load, \
                patch("vocagent.__main__.Workflow") as workflow, \
                redirect_stdout(stdout):
            workflow.return_value.run.return_value = SimpleNamespace(final_video=final_path)
            status = main(["apple", "--providers", "example:build",
                           "--output-dir", "chosen-output"])

        self.assertEqual(status, 0)
        load.assert_called_once_with("example:build")
        workflow.assert_called_once_with(
            providers, outline_candidate_count=2, video_prompt_candidate_count=2,
            video_candidate_count=2, concurrent_candidates=False,
        )
        workflow.return_value.run.assert_called_once_with(
            "apple", Path("chosen-output"), stop_after="video",
        )
        self.assertEqual(stdout.getvalue(), f"{final_path}\n")

    def test_candidate_counts_are_forwarded_independently(self):
        for concurrent_candidates in (False, True):
            with self.subTest(concurrent_candidates=concurrent_candidates):
                providers = providers_for_test()
                with patch("vocagent.__main__.load_providers", return_value=providers), \
                        patch("vocagent.__main__.Workflow") as workflow, \
                        redirect_stdout(io.StringIO()):
                    workflow.return_value.run.return_value = SimpleNamespace(
                        final_video=Path("chosen-output/final.mp4"),
                    )
                    status = main([
                        "apple", "--providers", "example:build",
                        *(["--concurrent-candidates"] if concurrent_candidates else []),
                        "--outline-candidate-count", "1",
                        "--video-prompt-candidate-count", "3",
                        "--video-candidate-count", "6",
                    ])

                self.assertEqual(status, 0)
                workflow.assert_called_once_with(
                    providers, outline_candidate_count=1, video_prompt_candidate_count=3,
                    video_candidate_count=6, concurrent_candidates=concurrent_candidates,
                )

    def test_invalid_candidate_counts_stop_before_providers_or_output(self):
        for stage in ("outline", "video-prompt", "video"):
            flag = f"--{stage}-candidate-count"
            for value in ("0", "-1", "1.5", "invalid"):
                with self.subTest(stage=stage, value=value), \
                        tempfile.TemporaryDirectory() as temporary:
                    output = Path(temporary) / "result"
                    stderr = io.StringIO()
                    stdout = io.StringIO()
                    with patch("vocagent.__main__.load_providers") as load, \
                            patch("vocagent.__main__.Workflow") as workflow, \
                            redirect_stderr(stderr), redirect_stdout(stdout), \
                            self.assertRaises(SystemExit) as raised:
                        main(["apple", "--providers", "example:build",
                              "--output-dir", str(output), flag, value])

                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(flag, stderr.getvalue())
                    self.assertIn("must be a positive integer", stderr.getvalue())
                    self.assertEqual(stdout.getvalue(), "")
                    load.assert_not_called()
                    workflow.assert_not_called()
                    self.assertFalse(output.exists())

    def test_concurrent_candidates_flag_enables_concurrency(self):
        providers = providers_for_test()
        with patch("vocagent.__main__.load_providers", return_value=providers), \
                patch("vocagent.__main__.Workflow") as workflow, \
                redirect_stdout(io.StringIO()):
            workflow.return_value.run.return_value = SimpleNamespace(
                final_video=Path("/tmp/example-final.mp4"),
            )
            status = main(["apple", "--providers", "example:build", "--concurrent-candidates"])

        self.assertEqual(status, 0)
        workflow.assert_called_once_with(
            providers, outline_candidate_count=2, video_prompt_candidate_count=2,
            video_candidate_count=2, concurrent_candidates=True,
        )

    def test_invalid_stop_stage_is_rejected_before_loading_providers(self):
        stderr = io.StringIO()
        with patch("vocagent.__main__.load_providers") as load, \
                redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["apple", "--stop-after", "outline"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--stop-after", stderr.getvalue())
        load.assert_not_called()

    def test_provider_load_failure_is_a_clear_cli_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = main(["apple", "--providers", "missing_separator"])
        self.assertEqual(status, 1)
        self.assertIn("module_name:function_name", stderr.getvalue())


class ProviderFactoryTests(unittest.TestCase):
    def test_factory_is_loaded_and_called_without_arguments(self):
        module = ModuleType("test_workflow_providers")
        providers = providers_for_test()
        module.build = Mock(return_value=providers)
        with patch.dict("sys.modules", {module.__name__: module}):
            actual = load_providers(f"{module.__name__}:build")
        self.assertIs(actual, providers)
        module.build.assert_called_once_with()

    def test_factory_must_return_a_providers_instance(self):
        module = ModuleType("test_workflow_providers")
        module.build = lambda: {"llm": object()}
        with patch.dict("sys.modules", {module.__name__: module}):
            with self.assertRaisesRegex(TypeError, "Providers instance"):
                load_providers(f"{module.__name__}:build")

    def test_factory_spec_requires_both_module_and_attribute(self):
        for spec in ("", "example", "example:", ":build"):
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(ValueError, "module_name:function_name"):
                    load_providers(spec)

    def test_missing_factory_attribute_propagates(self):
        module = ModuleType("test_workflow_providers")
        with patch.dict("sys.modules", {module.__name__: module}):
            with self.assertRaises(AttributeError):
                load_providers(f"{module.__name__}:missing")


if __name__ == "__main__":
    unittest.main()
