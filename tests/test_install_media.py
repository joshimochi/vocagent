"""Exercise the isolated media installer without APT/network/system mutation."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "install_media.sh"


class MediaInstallerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fake_bin = self.root / "tools"
        self.fake_bin.mkdir()
        self.prefix = self.root / "private media"
        self.log = self.root / "commands.jsonl"
        self.env = {
            **os.environ, "PATH": str(self.fake_bin) + os.pathsep + os.environ["PATH"],
            "INSTALLER_TEST_LOG": str(self.log),
        }
        self.tool("apt-get", r'''
import json, os, pathlib, re, sys
config = pathlib.Path(os.environ["APT_CONFIG"]).read_text()
with open(os.environ["INSTALLER_TEST_LOG"], "a") as log:
    log.write(json.dumps({"tool": "apt-get", "args": sys.argv[1:], "config": config}) + "\n")
if "install" in sys.argv:
    assert "--download-only" in sys.argv
    archives = pathlib.Path(re.search(r'Dir::Cache::archives "([^"]+)";', config)[1])
    (archives / "media_1_amd64.deb").write_bytes(b"test package")
''')
        self.tool("dpkg-deb", r'''
import json, os, pathlib, sys
with open(os.environ["INSTALLER_TEST_LOG"], "a") as log:
    log.write(json.dumps({"tool": "dpkg-deb", "args": sys.argv[1:]}) + "\n")
if sys.argv[1] == "--field":
    print("Package: media\nVersion: 1\nArchitecture: amd64")
else:
    assert sys.argv[1] == "--extract"
    root = pathlib.Path(sys.argv[3])
    binaries = root / "usr/bin"
    libraries = root / "lib/test-linux-gnu"
    binaries.mkdir(parents=True, exist_ok=True)
    libraries.mkdir(parents=True, exist_ok=True)
    loader = libraries / "ld-linux-test.so.2"
    loader.write_text('#!/bin/sh\n[ "$1" = --library-path ] || exit 91\nshift 2\nexec "$@"\n')
    loader.chmod(0o755)
    (libraries / "libinstaller-test.so.1").write_text("marker")
    for name in ("ffmpeg", "ffprobe"):
        binary = binaries / name
        if name == "ffprobe":
            binary.write_text('#!/bin/sh\nprintf "video\\naudio\\n"\n')
        else:
            binary.write_text('#!/bin/sh\n[ "${INSTALLER_TEST_FAIL:-}" != 1 ] || exit 29\n'
                              'for argument in "$@"; do :; done\n'
                              '[ "$argument" != smoke.mp4 ] || : > smoke.mp4\n'
                              'printf "private ffmpeg\\n"\n')
        binary.chmod(0o755)
''')

    def tool(self, name, source):
        path = self.fake_bin / name
        path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(source))
        path.chmod(0o755)

    def run_installer(self, **env):
        return subprocess.run(
            ["bash", str(SCRIPT), str(self.prefix)], env={**self.env, **env},
            capture_output=True, text=True,
        )

    def test_private_apt_download_extract_and_relocated_wrappers(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in self.log.read_text().splitlines()]
        installs = [row for row in records if row["tool"] == "apt-get" and "install" in row["args"]]
        self.assertEqual(len(installs), 1)
        self.assertIn("--download-only", installs[0]["args"])
        self.assertIn('Dir::Etc::parts "-";', installs[0]["config"])
        self.assertIn(str(self.prefix / "apt/status"), installs[0]["config"])
        self.assertEqual((self.prefix / "apt/status").read_text(), "")
        self.assertTrue((self.prefix / "current").is_symlink())
        self.assertTrue((self.prefix / "current/packages.txt").is_file())
        self.assertTrue((self.prefix / "current/archive-sha256.txt").is_file())
        for row in records:
            if row["tool"] == "dpkg-deb":
                self.assertIn(row["args"][0], ("--extract", "--field"))
        # The wrapper must still find its matching loader/libs after promotion.
        probe = subprocess.run(
            [str(self.prefix / "current/bin/ffmpeg"), "-version"],
            capture_output=True, text=True, env=self.env,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertIn("private ffmpeg", probe.stdout)

    def test_failed_validation_preserves_previous_release(self):
        first = self.run_installer()
        self.assertEqual(first.returncode, 0, first.stderr)
        previous = (self.prefix / "current").readlink()
        failed = self.run_installer(INSTALLER_TEST_FAIL="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual((self.prefix / "current").readlink(), previous)
        self.assertEqual(list((self.prefix / "releases").glob(".build-*")), [])

    def test_offline_reuses_successful_archive_set_without_update(self):
        self.assertEqual(self.run_installer().returncode, 0)
        self.log.write_text("")
        result = self.run_installer(VOCAGENT_MEDIA_OFFLINE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        apt = [row for row in calls if row["tool"] == "apt-get"]
        self.assertEqual(len(apt), 1)
        self.assertIn("--no-download", apt[0]["args"])


if __name__ == "__main__":
    unittest.main()
