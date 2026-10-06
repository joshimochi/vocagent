"""Integrity and resumable-transfer tests for the standalone model downloader."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from deploy.download_model import download_file, download_model, model_files


class HTTPResponse(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}

    def getcode(self):
        return self.status


def entry(name="model.safetensors", body=b"weights"):
    return {"Type": "blob", "Path": name, "Size": len(body),
            "Sha256": hashlib.sha256(body).hexdigest(), "Revision": "a" * 40,
            "Url": "https://model.test/weights"}


def listing(entries):
    return HTTPResponse(json.dumps({"Code": 200, "Data": {"Files": entries}}).encode())


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.transport = patch("deploy.download_model.urlopen").start()
        self.sleep = patch("deploy.download_model.time.sleep").start()
        self.addCleanup(patch.stopall)

    def download(self, metadata, **kwargs):
        return download_file(metadata, self.root, retry_delay=0, **kwargs)

    def test_listing_selects_weights_tokenizer_and_configs_and_pins_every_file(self):
        kept = ["model.safetensors", "config.json", "tokenizer.model", "vocab.txt", "merges.txt",
                "chat_template.jinja", "LICENSE", "LICENSE.txt", "README.md"]
        excluded = ["modeling_qwen.py", "run.sh", "pytorch_model.bin", "image.png"]
        self.transport.return_value = listing([entry(name) for name in kept + excluded])
        url, files = model_files("Qwen/Test", "master")
        self.assertEqual([item["Path"] for item in files], kept)
        self.assertEqual(parse_qs(urlsplit(url).query), {"Revision": ["master"], "Recursive": ["true"]})
        for item in files:
            self.assertEqual(parse_qs(urlsplit(item["Url"]).query), {
                "Revision": ["a" * 40], "FilePath": [item["Path"]],
            })
            self.assertEqual(len(item["Sha256"]), 64)

    def test_listing_rejects_missing_integrity_metadata_or_duplicate_paths(self):
        cases = []
        for change in ({"Sha256": "bad"}, {"Size": -1}, {"Size": True}, {"Revision": ""}):
            cases.append([entry("config.json"), {**entry(), **change}])
        cases += [[entry("config.json"), entry(), entry()], [entry()], [entry("config.json")]]
        for entries in cases:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.transport.return_value = listing(entries)
                model_files("Qwen/Test")

    def test_untrusted_paths_cannot_escape_output_directory(self):
        for name in ("../escape.json", "/escape.json", "nested/../../escape.json", "nested\\escape.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.transport.return_value = listing([entry("config.json"), entry(), entry(name)])
                model_files("Qwen/Test")
        with self.assertRaises(ValueError):
            self.download(entry("../escape.safetensors"))

    def test_success_is_published_only_after_digest_validation(self):
        metadata = entry()
        target = self.root / metadata["Path"]

        def serve(request, **kwargs):
            self.assertFalse(target.exists())
            return HTTPResponse(b"weights", headers={"Content-Length": "7"})

        self.transport.side_effect = serve
        self.assertEqual(self.download(metadata), "downloaded and verified")
        self.assertEqual(target.read_bytes(), b"weights")
        self.assertFalse(target.with_suffix(target.suffix + ".part").exists())

    def test_existing_files_are_reused_only_after_hash_verification(self):
        metadata = entry()
        target = self.root / metadata["Path"]
        target.write_bytes(b"weights")
        self.assertEqual(self.download(metadata), "verified existing")
        self.transport.assert_not_called()
        target.write_bytes(b"corrupt")
        self.transport.return_value = HTTPResponse(b"weights")
        self.assertEqual(self.download(metadata), "downloaded and verified")
        self.assertEqual(target.read_bytes(), b"weights")

    def test_interrupted_download_resumes_only_from_validated_range(self):
        metadata = entry(body=b"abcdef")
        self.transport.side_effect = [
            HTTPResponse(b"abcd", headers={"Content-Length": "6"}),
            HTTPResponse(b"ef", 206, {"Content-Range": "bytes 4-5/6", "Content-Length": "2"}),
        ]
        self.download(metadata)
        first, second = [call.args[0] for call in self.transport.call_args_list]
        self.assertIsNone(first.get_header("Range"))
        self.assertEqual(second.get_header("Range"), "bytes=4-")
        self.assertEqual((self.root / metadata["Path"]).read_bytes(), b"abcdef")

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        metadata = entry(body=b"abcdef")
        (self.root / (metadata["Path"] + ".part")).write_bytes(b"abc")
        self.transport.return_value = HTTPResponse(b"abcdef", 200, {"Content-Length": "6"})
        self.download(metadata)
        self.assertEqual((self.root / metadata["Path"]).read_bytes(), b"abcdef")
        self.assertEqual(self.transport.call_args.args[0].get_header("Range"), "bytes=3-")

    def test_invalid_partial_responses_never_append_or_publish(self):
        metadata = entry(body=b"abcdef")
        partial = self.root / (metadata["Path"] + ".part")
        for headers in ({}, {"Content-Range": "bytes 0-2/6"}, {"Content-Range": "bytes 3-5/9"},
                        {"Content-Range": "bytes 3-6/6"}, {"Content-Range": "bytes 3-5/*"},
                        {"Content-Range": "bytes 3-5/6", "Content-Length": "4"}):
            with self.subTest(headers=headers):
                partial.write_bytes(b"abc")
                self.transport.return_value = HTTPResponse(b"def", 206, headers)
                with self.assertRaises(OSError):
                    self.download(metadata, attempts=1)
                self.assertEqual(partial.read_bytes(), b"abc")
                self.assertFalse((self.root / metadata["Path"]).exists())

    def test_checksum_failure_retries_from_zero_and_preserves_previous_target(self):
        metadata = entry(body=b"abcdef")
        target = self.root / metadata["Path"]
        target.write_bytes(b"old model")
        self.transport.side_effect = [HTTPResponse(b"xxxxxx"), HTTPResponse(b"abcdef")]
        self.download(metadata)
        self.assertEqual(target.read_bytes(), b"abcdef")
        self.assertTrue(all(call.args[0].get_header("Range") is None
                            for call in self.transport.call_args_list))

    def test_retries_are_bounded_and_corrupt_file_is_never_published(self):
        metadata = entry()
        target = self.root / metadata["Path"]
        target.write_bytes(b"previous")
        self.transport.side_effect = lambda *args, **kwargs: HTTPResponse(b"corrupt")
        with self.assertRaisesRegex(OSError, "after 3 attempts"):
            self.download(metadata, attempts=3)
        self.assertEqual(self.transport.call_count, 3)
        self.assertEqual(target.read_bytes(), b"previous")
        self.assertFalse((self.root / (metadata["Path"] + ".part")).exists())

    def test_complete_partial_is_verified_before_reuse(self):
        metadata = entry()
        (self.root / (metadata["Path"] + ".part")).write_bytes(b"weights")
        self.download(metadata)
        self.transport.assert_not_called()
        self.assertEqual((self.root / metadata["Path"]).read_bytes(), b"weights")

    def test_manifest_records_verified_source_hashes_and_commit_pins(self):
        entries = [entry("config.json", b"{}"), entry()]

        def serve(request, **kwargs):
            if isinstance(request, str):
                return listing(entries)
            path = parse_qs(urlsplit(request.full_url).query)["FilePath"][0]
            return HTTPResponse(b"{}" if path == "config.json" else b"weights")

        self.transport.side_effect = serve
        with patch("builtins.print"):
            manifest = download_model("Qwen/Test", self.root, workers=2)
        self.assertEqual(manifest["source"], "ModelScope")
        self.assertEqual(manifest["repository"], "Qwen/Test")
        self.assertEqual(manifest["requested_revision"], "master")
        self.assertEqual(json.loads((self.root / "modelscope-download-manifest.json").read_text()), manifest)
        self.assertTrue(all(item["Revision"] == "a" * 40 for item in manifest["files"]))
        self.assertEqual([item["Sha256"] for item in manifest["files"]], [item["Sha256"] for item in entries])


if __name__ == "__main__":
    unittest.main()
