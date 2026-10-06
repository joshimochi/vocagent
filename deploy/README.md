# Models on hs

Deployment validation is incomplete. Both original checkpoints have been
downloaded and SHA256-verified on `hs`. Guard passed live safe/unsafe and
concurrent checks. The 80B text service has not passed startup with Guard loaded:
TP4 exhausted runtime headroom, and the initial PP4 reservation was too large.
The launch script below reduces that PP4 reservation to the measured available
memory; this retry and the complete storyboard smoke test still need to run.
On 2026-10-06, SSH and ping to `hs` stopped responding through the configured
jump host before that retry could be applied. Do not treat this configuration
as a verified deployment until both health checks and the workflow below pass.

The allocated devices are physical GPUs **4, 5, 6, 7** on `hs`: four Moore Threads
MTT S4000 GPUs with 48 GiB each. GPUs 0-3 belong to another workload. Docker's
`mthreads` runtime exposes the allocation as local devices 0-3.

The application calls two loopback-only vLLM services:

| Role | Checkpoint | Served name | Host port |
| --- | --- | --- | --- |
| Outline, storyboard, quality | Qwen/Qwen3-Next-80B-A3B-Instruct | `qwen3` | `18080` |
| Safety | Qwen/Qwen3Guard-Gen-8B | `qwen3-guard` | `18081` |

Generation and quality share the Qwen3 weights but use separate agents, prompts,
and requests. Both models use their original BF16 weights. Guard uses its native
chat template.
Model weights and KV caches reside on GPUs throughout inference. CPU weight
offloading is explicitly zero; KV offloading is left disabled (the runtime's
`None` default), and KV cache precision follows the model dtype.

The pinned runtime is the installed S4000 vLLM-MUSA image:

```
registry.mthreads.com/presale/devtech/vllm_musa@sha256:bec946b86d389888b8af8660537b786833fe41eb7121a3070561ddc648bc6237
```

It contains vLLM `0.21.0rc4.dev31+g50fe3fa1e.d20260702`, Transformers `5.8.1`,
and PyTorch `2.9.0`. Host driver: `3.3.5-server`. Qwen3 uses four-stage pipeline
parallelism; Guard uses tensor parallelism over the same four GPUs. Both allow
at most four simultaneous sequences. Qwen3 has a
16,384-token context; Guard has 8,192 tokens, comfortably above the generated
text limits. Prefill processes at most 512 tokens per batch. The launch script
sets memory budgets of 82% per GPU for Qwen3 and 10.5% for Guard. Their combined
runtime fit remains to be verified. `--enforce-eager` uses normal eager
GPU execution without graph capture.
The S4000 configuration uses 32-token cache blocks, disables prefix caching and
asynchronous scheduling, and preserves each checkpoint's generation config.
With the runtime defaults, live Guard tests produced repeated or malformed
verdicts. These settings restored complete native safe/unsafe verdicts, including
concurrent requests; no artificial verdict or forced classification is used.

The 80B checkpoint contains 151.49 GiB of weights and Guard adds 15.26 GiB.
Although the raw weights fit in the 192 GiB allocation, MUSA communication,
activations, and GPU KV caches require additional memory.
The larger Qwen3-235B cannot fit at BF16 or FP8. Its official GPTQ-Int4 checkpoint
fits by size, but this installed S4000 runtime failed numeric tests: GPTQ linear
layers returned zero outputs, and actual-shaped MoE prefill raised
`KeyError: BLOCK_SIZE_N`. It is therefore excluded. The original Next architecture
passed a GPU prefill/decode smoke test before downloading its full checkpoint.

## Reproduce

Run these commands on `hs` from the uploaded application directory
`/data/group6/vocagent/app`:

```sh
python3 deploy/download_model.py Qwen/Qwen3Guard-Gen-8B \
  /data/group6/vocagent/models/Qwen3Guard-Gen-8B --workers 4
python3 deploy/download_model.py Qwen/Qwen3-Next-80B-A3B-Instruct \
  /data/group6/vocagent/models/Qwen3-Next-80B-A3B-Instruct --workers 12
bash deploy/serve_hs.sh guard
bash deploy/serve_hs.sh text
```

The downloader uses the official Qwen repositories on ModelScope because direct
Hugging Face access is unavailable from this host. Every file is downloaded at
its recorded revision, checked against its SHA256, and atomically published.
`modelscope-download-manifest.json` records the source, revisions, and hashes
inside each model directory. Interrupted downloads can be resumed by running the
same command. Launching requires a completed verification manifest.

The scripts create only the `vocagent-qwen3` and `vocagent-qwen3-guard` containers
and files under `/data/group6/vocagent`. Existing containers are left in place.
Docker restarts the services after a reboot. Re-running the launch command with
an existing container name fails instead of replacing a running service.

Check readiness and logs:

```sh
curl --fail http://127.0.0.1:18080/health
curl --fail http://127.0.0.1:18081/health
docker logs --tail 40 vocagent-qwen3
docker logs --tail 40 vocagent-qwen3-guard
mthreads-gmi
```

Then exercise the complete text workflow:

```sh
QWEN_BASE_URL=http://127.0.0.1:18080/v1 \
QWEN_GUARD_BASE_URL=http://127.0.0.1:18081/v1 \
python3 -m vocagent sunrise curious journey \
  --stop-after storyboard --output-dir /data/group6/vocagent/outputs/example
```

For local use, open the SSH tunnel documented in the repository README. The
inference endpoints accept the OpenAI-compatible chat API; no OpenAI account,
cloud API, or Python SDK is involved.

References: [Qwen3 model](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct),
[Qwen3Guard format](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B),
[S4000 runtime documentation](https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/S4000/).
