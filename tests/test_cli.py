"""Verify the CLI boundary without calling external models or FFmpeg."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vocab_video_generator.__main__ import load_providers, main
from vocab_video_generator.models import SafetyResult
from vocab_video_generator.providers import LLMGenerator, Providers, TextSafetyAgent


class CLITests(unittest.TestCase):
    def test_default_stubs_stop_with_clear_error_and_exit_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            stderr = io.StringIO()
            stdout = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(stdout):
                status = main(["apple", "--output-dir", str(output)])

        self.assertEqual(status, 1)
        self.assertIn("Workflow stopped: Implement LLMGenerator.generate_outline", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")

    def test_safety_exhaustion_reports_human_intervention_and_exit_two(self):
        llm = Mock(spec=LLMGenerator)
        llm.generate_outline.return_value = "unsafe outline"
        safety = Mock(spec=TextSafetyAgent)
        safety.check.return_value = SafetyResult(False, "unsafe content")
        providers = Providers(llm=llm, text_safety=safety)
        with tempfile.TemporaryDirectory() as temporary:
            stderr = io.StringIO()
            with patch("vocab_video_generator.__main__.load_providers", return_value=providers), \
                    redirect_stderr(stderr):
                status = main(["apple", "--providers", "example:build", "--output-dir", temporary])

        self.assertEqual(status, 2)
        self.assertIn("Human intervention required:", stderr.getvalue())
        self.assertIn("outline", stderr.getvalue())
        self.assertEqual(llm.generate_outline.call_count, 16)

    def test_success_prints_final_path_and_passes_cli_arguments(self):
        providers = Providers()
        final_path = Path("/tmp/example-final.mp4")
        stdout = io.StringIO()
        with patch("vocab_video_generator.__main__.load_providers", return_value=providers) as load, \
                patch("vocab_video_generator.__main__.Workflow") as workflow, \
                redirect_stdout(stdout):
            workflow.return_value.run.return_value = SimpleNamespace(final_video=final_path)
            status = main(["apple", "green pear", "--providers", "example:build",
                           "--output-dir", "chosen-output"])

        self.assertEqual(status, 0)
        load.assert_called_once_with("example:build")
        workflow.assert_called_once_with(providers)
        workflow.return_value.run.assert_called_once_with(
            ["apple", "green pear"], Path("chosen-output"),
        )
        self.assertEqual(stdout.getvalue(), f"{final_path}\n")

    def test_provider_load_failure_is_a_clear_cli_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = main(["apple", "--providers", "missing_separator"])
        self.assertEqual(status, 1)
        self.assertIn("module_name:function_name", stderr.getvalue())


class ProviderFactoryTests(unittest.TestCase):
    def test_factory_is_loaded_and_called_without_arguments(self):
        module = ModuleType("test_workflow_providers")
        providers = Providers()
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
