# Deploy on hs

See the [project README](../README.md) for the workflow, models, CLI options, and
output files. This guide covers the shared GPU deployment and its validation
history.

## Recorded status — 2026-10-07

**GPU inference and native H3 capacity passed; final content validation remains
unverified.** These are historical results, not a fresh check of the host.

The latest recorded run, `full-workflow-3`, used 1344×768 video, five seconds,
and 50 sigma points. Its first clip completed rendering, ASR, and safety review,
but failed a safety check and was deleted. A second render started on
**2026-10-06 at 09:02:49 UTC**, the last confirmed event before SSH became
unavailable. Before restarting, inspect:

```text
/data/group6/vocagent/run-a406/full-workflow-3.exit
/data/group6/vocagent/run-a406/full-workflow-3.log
```

The [run record](validation/full-workflow-3-status.json) preserves that state.
An [earlier full run](validation/full-workflow-1.json) completed every stage but
omitted the requested words from its speech.

The current single-word input, English prompts, bundled H3 guide, `video_prompt`
rename, and removal of exact transcript word matching still need deployment and
validation on `hs`. Logging and early-capacity updates also await upload. The
remote records describe earlier source; pending work listed in those records
may differ from the current implementation. A complete 768p run must still
confirm safe output and English dialogue that teaches one meaning of the word.

Completed checks:

| Check | Recorded result |
| --- | --- |
| Model downloads | Qwen3-VL 32B verified by revision and SHA256; Guard 8B, ASR 1.7B, and aligner 0.6B verified. Existing H3 and ShieldGemma 2 directories reused. |
| [H3 decoders](validation/h3-decoders-musa.json) | Finite output from FP32 GPU decoder weights, with FP16 autocast for video; all seven regression tests passed. |
| [Speech and ShieldGemma 2](validation/hs-speech-shield.json) | Real ASR, alignment, and frame safety inference passed; GPU memory released after unload. |
| [H3 smoke test](validation/h3-full-gpu.json) | Full GPU-only audio/video generation at 448×256, five seconds, two sigma points. |
| [Native H3 capacity](validation/h3-native-gpu.json) | 1344×768, 243 frames (10.125 seconds), stereo 32 kHz AAC, and one denoise iteration. Peak allocation: 43.66 GiB; reserved: 44.68 GiB per GPU, below 47.91 GiB usable capacity. This checks capacity, not quality. |
| [Qwen3-VL](validation/qwen-vl-hs.json) | Known-answer generation and text/video selection passed, including 32 video frames. |
| [Remote test suite](validation/tests-hs.json) | 177 passed, no skips, on 2026-10-06, including numerical convolution and FFmpeg integration checks. |

The later local suite reported 205 tests: 201 passed, three failed because macOS
lacked `flock`, and one skipped (`TMPDIR=/private/tmp`). All 39 workflow and input
tests passed. That run predates the video prompt rename and H3 guide integration;
the [earlier local record](validation/tests-local-final.json) covers a different
revision. The current source still needs an unskipped test run on `hs`.

## Runtime

Use physical MTT S4000 GPUs **4–7**, with 48 GiB each. GPUs 0–3 are outside this
application's allocation. The container, `vocagent-local-a406`, exposes only
GPUs 4–7 and has no inference ports. Keep the old `vocagent-qwen3-guard`,
`group6-qwen3-vl`, and `vocagent-qwen3` services stopped during runs.

Setup uses the existing image:

```text
registry.mthreads.com/presale/devtech/vllm_musa@sha256:bec946b86d389888b8af8660537b786833fe41eb7121a3070561ddc648bc6237
```

The image supplies PyTorch 2.9.0/MUSA and driver 3.3.5-server. Setup preserves
vendor PyTorch and creates two environments under `run-a406`:

| Environment | Dependencies |
| --- | --- |
| `venv` | Speech, Guard, ShieldGemma 2, and Qwen3-VL; Transformers 4.57.6, Accelerate 1.12.0 |
| `h3-venv` | H3; Diffusers 0.40.0, Transformers 5.8.1, Accelerate 1.12.0 |

Separate environments resolve the incompatible Hugging Face Hub requirements
of Qwen-ASR and Diffusers. Both use torchvision 0.24.0 CPU wheels for
preprocessing only. Isolated FFmpeg/FFprobe wrappers provide H.264, AAC, and
libass without changing the vendor image's media libraries.

Inference stays on GPUs. The parent unloads its models before starting H3 and
waits for the worker to exit. Qwen3-VL uses uncached SDPA because cached decoding
failed correctness checks on MUSA. H3 uses the full BF16 encoder and 33B
transformer, FP32 decoder weights, and feed-forward chunks of at most 8,192
tokens; attention sees the full sequence. CPU/disk model placement is rejected.
The 235B Qwen3-VL model cannot fit at BF16 or FP8, and tested GPTQ kernels failed;
no working 235B alternative is claimed.

## Setup

From the checkout, upload the source without weights, environments, or
credentials:

```sh
tar --exclude='__pycache__' -czf /tmp/vocagent-a406-app.tar.gz vocagent tests deploy README.md prompts.md pyproject.toml
scp -F ~/.ssh/config /tmp/vocagent-a406-app.tar.gz hs:/data/group6/vocagent/run-a406/app.tar.gz
ssh -F ~/.ssh/config hs 'mkdir -p /data/group6/vocagent/run-a406/app && tar -xzf /data/group6/vocagent/run-a406/app.tar.gz -C /data/group6/vocagent/run-a406/app'
```

Then run on `hs`:

```sh
cd /data/group6/vocagent/run-a406/app
bash deploy/setup_hs.sh
```

Setup reuses or creates the container, installs the environments and media
wrappers, and downloads/verifies the Qwen checkpoints. It requires existing H3
and ShieldGemma 2 checkpoints; it does not accept licenses, obtain gated
credentials, or substitute models. Qwen download manifests record ModelScope
repositories, revisions, and SHA256 hashes. Check the existing H3/Shield
installation records before claiming the same verification for those models.

## Test and run

Run tests in both environments, using the media wrappers for the main suite:

```sh
docker exec -e PYTHONPATH=/data/group6/vocagent/run-a406/app \
  -e PATH=/data/group6/vocagent/run-a406/media/current/bin:/usr/local/bin:/usr/bin:/bin \
  -w /data/group6/vocagent/run-a406/app vocagent-local-a406 \
  /data/group6/vocagent/run-a406/venv/bin/python -m unittest discover -s tests -v

docker exec -e PYTHONPATH=/data/group6/vocagent/run-a406/app \
  -w /data/group6/vocagent/run-a406/app vocagent-local-a406 \
  /data/group6/vocagent/run-a406/h3-venv/bin/python -m unittest discover -s tests -p test_minimax.py -v
```

Test text generation first, then the full workflow. Use one English word and a
new output directory for each run:

```sh
bash deploy/run_hs.sh curious --stop-after video-prompt \
  --outline-candidate-count 1 --video-prompt-candidate-count 1 \
  --output-dir /data/group6/vocagent/outputs/a406-text-smoke

bash deploy/run_hs.sh curious \
  --output-dir /data/group6/vocagent/outputs/a406-video-smoke
```

`run_hs.sh` holds `/data/group6/vocagent/vocagent-gpu.lock` inside the container
until Python exits. An in-process lock also covers inference and the H3 worker.
Do not run independent GPU probes alongside the agent.

For long runs over intermittent SSH, detach the command and keep its log outside
the new output directory:

```sh
nohup bash deploy/run_hs.sh curious \
  --output-dir /data/group6/vocagent/outputs/detached-example \
  > /data/group6/vocagent/logs/detached-example.log 2>&1 < /dev/null &
```

Setup and download logs are under `/data/group6/vocagent/logs` and
`/data/group6/vocagent/run-a406`. The old probe source in
`run-a406/speech-probe-app` is outdated; do not use it to validate current code.
