"""H3 subprocess failures must not bypass GPU serialization or artifact checks."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vocagent.isolated_h3 import IsolatedMiniMaxH3VideoGenerator
from vocagent.h3_worker import main as worker_main


class IsolatedH3Tests(unittest.TestCase):
    def runtime(self):
        state = {"active": False}

        @contextmanager
        def use(key, loader):
            state["active"] = True
            try:
                yield loader()
            finally:
                state["active"] = False

        return SimpleNamespace(backend="musa", use=use), state

    def test_native_worker_runs_inside_lock_with_literal_unicode_request(self):
        runtime, state = self.runtime()
        provider = IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable="/env/bin/python")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'clip.mp4'

            def run(command, **kwargs):
                self.assertTrue(state["active"])
                self.assertEqual(command[:3], ["/env/bin/python", "-m", "vocagent.h3_worker"])
                request = json.loads(Path(command[3]).read_text())
                self.assertEqual(request['storyboard'], '说 hello; $(not-a-command)')
                self.assertEqual(request['backend'], 'musa')
                self.assertNotIn('shell', kwargs)
                self.assertNotEqual(Path(request['output']), output)
                Path(request['output']).write_bytes(b'video')

            with patch('vocagent.isolated_h3.subprocess.run', side_effect=run):
                provider.generate('说 hello; $(not-a-command)', output)
            self.assertFalse(state['active'])
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['clip.mp4'])

    def test_worker_failure_or_absent_output_never_passes(self):
        runtime, state = self.runtime()
        provider = IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable="python")
        for error in (subprocess.CalledProcessError(1, ['python']), subprocess.TimeoutExpired(['python'], 1), None):
            with tempfile.TemporaryDirectory() as directory, self.subTest(error=error):
                with patch('vocagent.isolated_h3.subprocess.run', side_effect=error), self.assertRaises(RuntimeError):
                    provider.generate('hello', Path(directory) / 'clip.mp4')
                self.assertFalse(state['active'])
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_stale_output_cannot_mask_missing_or_failed_generation(self):
        runtime, state = self.runtime()
        provider = IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable="python")
        for error in (None, subprocess.TimeoutExpired(['python'], 1)):
            with tempfile.TemporaryDirectory() as directory, self.subTest(error=error):
                output = Path(directory) / "clip.mp4"
                output.write_bytes(b"previous video")

                def run(command, **kwargs):
                    self.assertEqual(output.read_bytes(), b"previous video")
                    if error:
                        request = json.loads(Path(command[3]).read_text())
                        Path(request['output']).write_bytes(b"partial video")
                        raise error

                with patch('vocagent.isolated_h3.subprocess.run', side_effect=run), self.assertRaises(RuntimeError):
                    provider.generate('hello', output)
                self.assertEqual(output.read_bytes(), b"previous video")
                self.assertEqual([path.name for path in Path(directory).iterdir()], ["clip.mp4"])
                self.assertFalse(state['active'])

    def test_worker_closes_runtime_on_success_and_model_failure(self):
        for error in (None, RuntimeError("GPU allocation failed")):
            with tempfile.TemporaryDirectory() as directory, self.subTest(error=error):
                path = Path(directory) / "request.json"
                path.write_text(json.dumps({
                    "backend": "musa", "model": "/models/H3", "settings": {},
                    "storyboard": "hello", "output": str(Path(directory) / "video.mp4"),
                }))
                runtime = Mock()
                generator = Mock()
                generator.generate.side_effect = error
                with patch('vocagent.h3_worker.GPURuntime', return_value=runtime), \
                        patch('vocagent.h3_worker.MiniMaxH3VideoGenerator', return_value=generator):
                    if error is None:
                        worker_main([str(path)])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "GPU allocation failed"):
                            worker_main([str(path)])
                runtime.close.assert_called_once_with()

    def test_invalid_worker_configuration_fails_before_launch(self):
        runtime, _ = self.runtime()
        for timeout in (0, -1, True, float('nan'), float('inf')):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable="python", timeout=timeout)
        with self.assertRaises(ValueError):
            IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable=" ")
