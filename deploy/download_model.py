#!/usr/bin/env python3
"""Download original ModelScope model files with pinned SHA256 verification."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


API = "https://www.modelscope.cn/api/v1/models"
CHUNK_SIZE = 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(repo, revision="master"):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Repository must have the form owner/model.")
    listing_url = f"{API}/{quote(repo, safe='/')}/repo/files?" + urlencode({
        "Revision": revision, "Recursive": "true",
    })
    with urlopen(listing_url, timeout=60) as response:
        listing = json.load(response)
    if listing.get("Code") != 200:
        raise ValueError("ModelScope did not return a successful repository listing.")
    files = []
    paths = set()
    for entry in listing["Data"]["Files"]:
        if entry.get("Type") != "blob":
            continue
        name = entry["Path"]
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name
                or str(path) != name or not path.parts):
            raise ValueError("ModelScope returned an invalid file path.")
        basename = path.name.lower()
        if not (path.suffix.lower() in {".safetensors", ".json", ".jinja"}
                or basename in {"tokenizer.model", "vocab.txt", "merges.txt", "license", "readme.md"}
                or basename.startswith("license.")):
            continue
        if name in paths:
            raise ValueError("ModelScope returned duplicate file paths.")
        paths.add(name)
        size, digest, pin = entry["Size"], entry["Sha256"], entry["Revision"]
        if (type(size) is not int or size < 0 or not isinstance(digest, str)
                or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)
                or not isinstance(pin, str) or not pin):
            raise ValueError(f"Missing integrity metadata for {name}.")
        files.append({"Path": name, "Size": size, "Sha256": digest.lower(), "Revision": pin,
                      "Url": f"{API}/{quote(repo, safe='/')}/repo?" + urlencode({
                          "Revision": pin, "FilePath": name,
                      })})
    if not any(item["Path"].endswith(".safetensors") for item in files):
        raise ValueError("Repository contains no safetensors model files.")
    if "config.json" not in paths:
        raise ValueError("Repository contains no config.json.")
    return listing_url, files


class ChecksumError(OSError):
    pass


def _transfer(entry, partial):
    total = entry["Size"]
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        partial.unlink()
        offset = 0
    if offset == total:
        return
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urlopen(Request(entry["Url"], headers=headers), timeout=120) as response:
        status = response.getcode()
        if status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if match is None:
                raise OSError("Partial response omitted a valid Content-Range.")
            start, end, response_total = map(int, match.groups())
            if start != offset or response_total != total or not start <= end < total:
                raise OSError("Partial response Content-Range disagrees with the requested file.")
            expected_length = end - start + 1
            mode = "ab" if offset else "wb"
        elif status == 200:
            # A server may ignore Range; restart instead of appending a full file.
            offset = 0
            expected_length = total
            mode = "wb"
        else:
            raise OSError(f"Unexpected download HTTP status {status}.")
        declared_length = response.headers.get("Content-Length")
        if declared_length is not None and int(declared_length) != expected_length:
            raise OSError("Download Content-Length disagrees with file metadata.")
        received = 0
        with partial.open(mode) as output:
            while chunk := response.read(CHUNK_SIZE):
                received += len(chunk)
                if received > expected_length:
                    raise OSError("Download exceeded the expected response length.")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if received != expected_length or offset + received != total:
            raise OSError("Download interrupted before the complete file arrived.")


def download_file(entry, output_dir, *, attempts=4, retry_delay=2):
    root = Path(output_dir).resolve()
    target = root / entry["Path"]
    partial = target.with_name(target.name + ".part")
    for path in (target, partial):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Download path escapes its output directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if (target.is_file() and target.stat().st_size == entry["Size"]
            and sha256(target) == entry["Sha256"]):
        return "verified existing"
    for attempt in range(attempts):
        try:
            if entry["Size"] == 0:
                partial.write_bytes(b"")
            else:
                _transfer(entry, partial)
            if partial.stat().st_size != entry["Size"] or sha256(partial) != entry["Sha256"]:
                raise ChecksumError("SHA256 verification failed.")
            partial.replace(target)
            return "downloaded and verified"
        except (OSError, HTTPException, ValueError) as exc:
            if isinstance(exc, ChecksumError):
                partial.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise OSError(f"Failed to download {entry['Path']} after {attempts} attempts: {exc}") from exc
            time.sleep(min(retry_delay * (2 ** attempt), 30))


def download_model(repo, output_dir, revision="master", workers=4):
    if type(workers) is not int or workers < 1:
        raise ValueError("Workers must be a positive integer.")
    listing_url, files = model_files(repo, revision)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"{repo}: {len(files)} files, {sum(item['Size'] for item in files):,} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(download_file, entry, output): entry for entry in files}
        for future in as_completed(jobs):
            entry = jobs[future]
            print(f"{entry['Path']}: {future.result()}", flush=True)
    manifest = {
        "source": "ModelScope", "repository": repo, "requested_revision": revision,
        "listing_url": listing_url, "verified_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    temporary = output / "modelscope-download-manifest.json.part"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "modelscope-download-manifest.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="ModelScope repository, e.g. Qwen/Qwen3Guard-Gen-8B")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        download_model(args.repo, args.output_dir, args.revision, args.workers)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")


if __name__ == "__main__":
    main()
