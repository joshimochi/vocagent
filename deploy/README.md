# hs deployment

## Current status — 2026-10-06

**The technical pipeline and native H3 capacity checks passed; final spoken-vocabulary validation is in progress.** Source changes
for the model adapters have been uploaded to `/data/group6/vocagent/run-a406/app`.
The final logging, vocabulary-coverage, and early-capacity updates are verified
locally and awaiting synchronization while SSH is unavailable. Network connectivity
to `hs` is intermittent: SSH connections frequently time out, including an
established interactive session. Detached validation jobs use locks and log files
so connection loss does not duplicate model loads.

The latest full run is `full-workflow-3` (1344×768, five seconds, 50 sigma
points). The first native-resolution clip completed rendering, ASR and safety
review, then failed a safety check and was deleted. The bounded retry began a
second render at 09:02:49 UTC. That is the last confirmed remote event; the
subsequent TCP/SSH outage prevents inspection. Final acceptance and spoken vocabulary remain
**unverified**. Inspect `run-a406/full-workflow-3.exit`
and its `.log` before restarting anything.
[Latest run status](validation/full-workflow-3-status.json).

Completed on `hs`:

- Confirmed four allocated MTT S4000 GPUs, physical IDs **4,5,6,7**, 48 GiB each.
  GPUs 0–3 are outside this application's allocation.
- Downloaded and revision/SHA256-verified the official
  `Qwen/Qwen3-VL-32B-Instruct` checkpoint (66,726,517,854 bytes). The completed
  `modelscope-download-manifest.json` is present on hs.
- Verified existing Guard 8B, ASR 1.7B and aligner 0.6B downloads; reused existing
  ShieldGemma 2 and full H3 checkpoint directories.
- Created isolated container `vocagent-local-a406` exposing GPUs 4–7, with no
  inference ports. Stopped the old `vocagent-qwen3-guard` and `group6-qwen3-vl`
  services to release this allocation. `vocagent-qwen3` was already stopped.
  VL has restart policy `no`; Guard has `unless-stopped` and was explicitly
  stopped, so it remains stopped across reboots. Keep both stopped during runs.
- Installed separate speech/vision and H3 environments without replacing vendor
  MUSA PyTorch. Installed matching torchvision 0.24.0 CPU preprocessing wheels
  with dependency installation disabled, retaining GPU-only model inference.
- Installed an isolated FFmpeg/FFprobe runtime with working H.264, AAC and libass.
  Its private loader/libraries avoid the vendor image's OpenCL package conflict.
- Passed real GPU H3 decoder probes: FP32 video VAE weights (9.70 GiB) with FP16
  GPU autocast, and FP32 audio VAE weights (0.56 GiB), both producing finite output.
  The exact convolution adaptations passed all seven H3 regression tests on hs.
  [Decoder validation record](validation/h3-decoders-musa.json).
- Passed real ShieldGemma 2 inference on sampled video frames with its trained,
  tied language-model head. The model released all allocated memory afterward.
- Passed real Qwen3-ASR plus ForcedAligner inference on a known spoken sentence,
  with aligned output from 0.16 to 3.20 seconds and zero allocations after unload.
  Fifteen speech/vision contract and convolution tests passed on hs.
  [Speech and Shield validation record](validation/hs-speech-shield.json).
- Passed full H3 native audio/video generation at 448×256, 5 seconds, two sigma
  points. Its encoder used 62.13 GiB and diffusion transformer 61.72 GiB across
  GPUs 4–7, alongside both GPU decoders; all GPU memory released at process exit.
- Passed H3 native-resolution capacity: 1344×768, 243 frames (10.125 seconds),
  stereo 32 kHz AAC, one complete denoise iteration and both decoders. Peak GPU
  allocation was 43.66 GiB; peak reserved memory was 44.68 GiB, within each GPU's
  47.91 GiB usable capacity. Total runtime was 612 seconds, including 175 seconds
  for denoising. [Native capacity record](validation/h3-native-gpu.json).
- Passed Qwen3-VL known-answer text generation, text selection, and video
  selection with 32 actual frames (8,676 input tokens, 77 seconds for selection).
  The MUSA path uses uncached SDPA after cached decoding failed correctness tests.
  Explicit image-processor size bounds enforce the 262,144-pixel frame cap.
  [Qwen-VL validation record](validation/qwen-vl-hs.json).

Still required:

- The corrected complete 768p vocabulary-to-subtitled-video run using native H3 dialogue tags.
  The first complete run passed every stage but its speech omitted the requested
  words; this semantic failure is preserved in [its report](validation/full-workflow-1.json).

The previous `serve_hs.sh` starts legacy HTTP text services and is retained only
for compatibility. It is not used by the new local workflow.

The complete test suite passed on hs: **177 tests, no skips**, including numerical
convolution checks and real FFmpeg integration. [Test record](validation/tests-hs.json).
The final local source passes 188 tests (14 dependency/media checks skipped),
including spoken-vocabulary coverage and early candidate-capacity validation.
[Local test record](validation/tests-local-final.json). The unskipped test run
for those final updates is pending restored SSH access. These tests complement the live model probes. The first full workflow passed the
complete sequence; a corrected run verifies the corrected spoken-vocabulary prompt.

## Runtime and model choice

Base image already present on the host:

```
registry.mthreads.com/presale/devtech/vllm_musa@sha256:bec946b86d389888b8af8660537b786833fe41eb7121a3070561ddc648bc6237
```

It provides PyTorch 2.9.0/MUSA, driver 3.3.5-server. The new workflow does not use
vLLM serving. Speech, Guard, Shield, and Qwen3-VL use Transformers 4.57.6 and
Accelerate 1.12.0 in `run-a406/venv`. H3 uses Diffusers 0.40.0 and Transformers
5.8.1 in `run-a406/h3-venv`. These environments are intentionally separate:
Qwen-ASR 0.0.6 pins Transformers 4.57.6 and Hugging Face Hub <1, while Diffusers
0.40 needs Hub >=1.23. The parent releases its weights before starting the H3
worker and serializes access until that worker exits. Qwen3-VL uses uncached
SDPA on MUSA because cached decoding failed real correctness checks; this
recomputes token prefixes on GPU and does not offload weights or activations.

Qwen3-VL **32B BF16** is the largest dense release. The 235B model cannot fit in
192 GiB at BF16 or FP8. Previous numeric tests of the installed S4000 runtime's
GPTQ kernels failed; a 235B 4-bit configuration is therefore not claimed as a
working alternative. Guard **8B**, ASR **1.7B**, aligner **0.6B**, and ShieldGemma 2
**4B** are the largest released checkpoints in the requested families.

H3 uses the full released BF16 Qwen encoder and 33B diffusion transformer, plus
FP32 GPU decoder weights (MUSA video decoding uses FP16 autocast). Placement reserves activation headroom and rejects CPU/disk
maps. Weight sharding moves activations between GPUs, never model layers to CPU
between forwards. Token-independent feed-forward projections process at most
8,192 tokens at a time to reduce temporary GPU allocations; attention still sees
the complete sequence. The S4000 memory measurements above come from this
GPU-only implementation's live capacity test.

References: [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct),
[Qwen3Guard](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B),
[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR),
[ShieldGemma 2](https://huggingface.co/google/shieldgemma-2-4b-it),
[official H3 MUSA recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-MUSA.md).

## Upload and finish setup

From this checkout, create and upload a source archive. It contains no model
weights, virtual environments, or credentials:

```sh
tar --exclude='__pycache__' -czf /tmp/vocagent-a406-app.tar.gz vocagent tests deploy README.md workflow.md pyproject.toml
scp -F ~/.ssh/config /tmp/vocagent-a406-app.tar.gz hs:/data/group6/vocagent/run-a406/app.tar.gz
ssh -F ~/.ssh/config hs 'mkdir -p /data/group6/vocagent/run-a406/app && tar -xzf /data/group6/vocagent/run-a406/app.tar.gz -C /data/group6/vocagent/run-a406/app'
```

Then **on hs**:

```sh
cd /data/group6/vocagent/run-a406/app
bash deploy/setup_hs.sh
```

Setup reuses the existing container or creates it with the exact allocated GPUs.
It creates separate environments inheriting vendor PyTorch, installs dependency
requirements, prepares isolated media binaries, and downloads/verifies the Qwen
checkpoints. H3 and Shield use the existing model directories; setup fails if
those are missing. It does not accept licenses, acquire gated credentials, or
substitute models. Download manifests record the ModelScope source repository,
file revisions, and SHA256 hashes. H3/Shield provenance should be checked against
their existing installation records before claiming equivalent verification.

Run the tests in both environments. Set the media PATH to the installed wrappers:

```sh
docker exec -e PYTHONPATH=/data/group6/vocagent/run-a406/app \
  -e PATH=/data/group6/vocagent/run-a406/media/current/bin:/usr/local/bin:/usr/bin:/bin \
  -w /data/group6/vocagent/run-a406/app vocagent-local-a406 \
  /data/group6/vocagent/run-a406/venv/bin/python -m unittest discover -s tests -v

docker exec -e PYTHONPATH=/data/group6/vocagent/run-a406/app \
  -w /data/group6/vocagent/run-a406/app vocagent-local-a406 \
  /data/group6/vocagent/run-a406/h3-venv/bin/python -m unittest discover -s tests -p test_minimax.py -v
```

Validate text first, then the complete agent, using new output directories:

```sh
bash deploy/run_hs.sh sunrise curious journey --stop-after storyboard \
  --outline-candidate-count 1 --storyboard-candidate-count 1 \
  --output-dir /data/group6/vocagent/outputs/a406-text-smoke

bash deploy/run_hs.sh sunrise curious journey \
  --output-dir /data/group6/vocagent/outputs/a406-video-smoke
```

`run_hs.sh` holds `/data/group6/vocagent/vocagent-gpu.lock` inside the container to prevent two managed
runs from competing. Model residency also has an in-process lock covering
inference and the H3 worker. Do not run independent GPU probes alongside a full
agent run. The model loading log reports each family switch.

For long runs over the intermittent SSH connection, detach the command and keep
its log outside the new output directory:

```sh
nohup bash deploy/run_hs.sh sunrise curious journey \
  --output-dir /data/group6/vocagent/outputs/detached-example \
  > /data/group6/vocagent/logs/detached-example.log 2>&1 < /dev/null &
```

Check download/setup evidence in `/data/group6/vocagent/logs` and
`/data/group6/vocagent/run-a406`. Earlier speech probe source in
`run-a406/speech-probe-app` is **outdated** and must not be used to validate the
fixed implementation. Existing inference containers and checkpoint files have
been preserved.
