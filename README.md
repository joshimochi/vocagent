# vocagent

Turn one English word into a subtitled cartoon that teaches one clear meaning.
The workflow generates candidates, checks safety, and selects the best result
using separate model calls. It runs locally on `hs`, using physical GPUs **4–7**
(four MTT S4000 cards, 48 GiB each).

**Validation:** the technical pipeline and native-resolution H3 capacity checks
passed on `hs`. The current single-word prompts still need a complete content
validation run. See [deployment status and setup](deploy/README.md).

## Run

After [setting up hs](deploy/README.md#setup):

```sh
cd /data/group6/vocagent/run-a406/app
bash deploy/run_hs.sh curious \
  --output-dir /data/group6/vocagent/outputs/example
```

Use one word and a new or empty output directory. Input accepts ASCII letters
with internal apostrophes or hyphens, such as `don't` or `well-known`; it does not
check a dictionary. Prompts request English text and speech throughout.

The default run produces a subtitled video with two candidate slots per stage.
Use `--stop-after video-prompt` to produce only the selected outline and video
prompt. Set candidate counts independently:

```sh
bash deploy/run_hs.sh curious --stop-after video-prompt \
  --outline-candidate-count 2 --video-prompt-candidate-count 2 \
  --output-dir /data/group6/vocagent/outputs/text-example
```

`--video-candidate-count` accepts 1 or 2 with the local provider. Text candidate
counts can be any positive integer. `--concurrent-candidates` lets slots run
concurrently within a stage; stages and local GPU inference remain sequential.

```python
from vocagent.local import create_providers
from vocagent.pipeline import Workflow

providers = create_providers()
try:
    result = Workflow(providers).run("curious", "output")
    print(result.final_video)
finally:
    providers.close()
```

The Python API also accepts `["curious"]`. Pass candidate counts and
`concurrent_candidates=True` to `Workflow`; pass `stop_after="video_prompt"` to
`run` to receive a `VideoPromptResult` with the selected text paths.

## Workflow

1. Generate outlines, check each with Qwen3Guard, and select a safe outline.
2. Generate video prompts, check each with Guard, and select a safe prompt.
3. Generate video candidates with native audio using MiniMax H3.
4. Extract mono 16 kHz PCM audio with FFmpeg, transcribe it in English, and align
   word timestamps. Save the recognized text and timestamps without rewriting them.
5. Check sampled frames with ShieldGemma 2 and the complete, newline-joined
   transcript with Guard. Both checks must pass.
6. Select the best safe video using Qwen3-VL and timestamped image frames.
7. Reuse its audio and transcript, write ASS subtitles, and burn them into the
   video with FFmpeg while preserving the original audio.

Video prompt generation includes the bundled [H3 guide](vocagent/prompts/h3_prompt_guide.md)
and task instructions for cartoon animation, one word meaning, English dialogue,
and a 15-second clip with at most 30 spoken words. The selected prompt goes to H3 unchanged.
See [prompt reference](prompts.md) for the format and source files.

Each slot gets up to four generation and safety attempts. Rejected text supplies
feedback for the next attempt. Rejected videos and their audio/transcript files
are deleted and retried without feedback. Exhausted slots are ignored. A quality
judge selects from safe survivors in slot order; a sole survivor is used directly.
If none survive, the workflow raises `HumanInterventionRequired`.

Provider, GPU, media, transcript, and malformed-response errors stop the run.
With concurrency enabled, new attempts stop while in-flight work finishes.
There is no automatic resume; resolve the failure and use a new output directory.

### Safety and selection limits

ShieldGemma checks `dangerous`, `sexual`, and `violence` policies at 2 frames per
second, including the final nominal frame. A violation probability of 0.5 or
higher rejects the video. Clips exceeding the sampling budget raise an error.
Sampling can miss brief events between frames.

The video judge sees up to 16 frames per candidate and does not hear audio. The
local 16K context budget supports two candidate slots (32 frames total), checked
before generation. ASR supplies transcript safety and subtitles; the workflow
**does not require an exact target-word match in recognized speech**.

## Models and runtime

| Task | Local checkpoint |
| --- | --- |
| Outline, video prompt, text and visual selection | Qwen/Qwen3-VL-32B-Instruct, BF16 |
| Text and transcript safety | Qwen/Qwen3Guard-Gen-8B, BF16 |
| Video with native audio | MiniMaxAI/MiniMax-H3, full released weights |
| Speech recognition | Qwen/Qwen3-ASR-1.7B, BF16 |
| Word timestamps | Qwen/Qwen3-ForcedAligner-0.6B, BF16 |
| Visual safety | google/shieldgemma-2-4b-it, BF16 |

The runtime loads one model family at a time and releases its weights before
loading another. It reserves GPU headroom and rejects CPU, disk, or meta weight
placement and offloading. There is no smaller-model or CPU inference fallback.
Tokenization, checkpoint loading, preprocessing, and media I/O may use the CPU.

Separate stateless generation and selection agents share the VL weights. On
MUSA, Qwen3-VL uses uncached SDPA because cached decoding failed correctness
checks. ASR and its aligner reside together. H3 runs in a separate Python
environment to avoid dependency conflicts, with the GPU lock held until it exits.
H3 uses BF16 encoder/diffusion weights and FP32 decoders, with FP16 autocast for
MUSA video decoding. [Deployment notes](deploy/README.md) cover versions and probes.

The default factory is `vocagent.local:create_providers`. Custom factories,
selected with `--providers module:factory`, must return a complete
[`Providers`](vocagent/providers.py) bundle, including candidate-count validation
and resource cleanup. When concurrency is enabled, generation, safety, media,
and speech implementations must support concurrent calls on the same instance.
Local runs need checkpoint directories, with no API keys or inference endpoints.

## Configuration

`VOCAGENT_DEPLOY_ROOT` defaults to `/data/group6/vocagent`. All model paths point
to local directories; the deployment container exposes only GPUs 4–7.

| Variable | Default |
| --- | --- |
| `QWEN_VL_MODEL` | `$VOCAGENT_DEPLOY_ROOT/models/Qwen3-VL-32B-Instruct` |
| `QWEN_GUARD_MODEL_PATH` | `$VOCAGENT_DEPLOY_ROOT/models/Qwen3Guard-Gen-8B` |
| `QWEN_ASR_MODEL` | `$VOCAGENT_DEPLOY_ROOT/models/Qwen3-ASR-1.7B` |
| `QWEN_ALIGNER_MODEL` | `$VOCAGENT_DEPLOY_ROOT/models/Qwen3-ForcedAligner-0.6B` |
| `SHIELDGEMMA_MODEL` | `$VOCAGENT_DEPLOY_ROOT/models/shieldgemma-2-4b-it` |
| `MINIMAX_H3_MODEL` | `/data/group6/minimax-h3/models/MiniMax-H3` |
| `MINIMAX_H3_PYTHON` | `$VOCAGENT_DEPLOY_ROOT/run-a406/h3-venv/bin/python` |
| `MINIMAX_H3_PROFILE` | `quality` (1344×768); `preview` uses 960×544 |
| `MINIMAX_H3_STEPS` | `50` for quality, `20` for preview |
| `MINIMAX_H3_TIMEOUT_SECONDS` | `14400` per video candidate |

Both landscape profiles request 15 seconds, rounded to H3's native frame grid:
345 frames at 24 fps, or 14.375 seconds. Outline and video prompt generation use
the same 15-second budget. A measured capacity probe suggests roughly 2.5 hours
for a 10-second, native-resolution, 50-point render on S4000; this is an estimate,
not a measured full render. The worker timeout defaults to four hours.

## Outputs

| Path | Contents |
| --- | --- |
| `vocabulary.json` | Target word |
| `outline.txt`, `video_prompt.txt` | Selected texts |
| `outline_candidates/`, `video_prompt_candidates/` | All accepted text candidates |
| `video_candidates/` | Accepted videos with `.wav` and `.json` sidecars |
| `audio.wav`, `transcript.json` | Selected audio and aligned transcript |
| `subtitles.ass`, `final.mp4` | Subtitles and finished video |
| `benchmark.json` | Timing, resource use, and run status |

Candidate files use `candidate-N-attempt-M` with one-based slot and attempt
numbers. Accepted text is saved as UTF-8 before selection, including alternatives.
Earlier artifacts remain after later failures. Stopping after `video-prompt`
keeps both text candidate directories and the selected texts.

Every started workflow writes a benchmark, including failed runs and text-only
runs; preflight failures do not. `result.benchmarks` points to the report.
It records per-call and per-step timings, retries, status, errors, and available
throughput and memory metrics.

Timing separates model loading, synchronized inference, and other work such as
preprocessing, media I/O, unloading, lock waits, and worker startup. Summed call
times may exceed elapsed time when slots overlap. Workflow timing excludes
provider creation/cleanup and input/output preflight. CPU metrics distinguish
call-thread time, total process time, and isolated worker time; GPU memory uses
maximum observed peaks, not sums. Throughput uses inference time. Custom providers
receive wall and CPU timing; model metrics require instrumentation. H3 worker
metrics join its generation step; a hard timeout without a worker report records
`worker_benchmark_status: unavailable`.

## Development

```sh
python3 -m unittest discover -s tests -v
```

Core tests need Python 3.10+. Media tests need FFmpeg/FFprobe with libass, H.264,
and AAC; numerical H3 tests need PyTorch. Real model tests need the `hs`
environments and checkpoints. Fake-provider tests do not verify model compatibility.

The former `storyboard` stage is now `video_prompt` in Python and filenames, or
`video-prompt` in CLI flags. Use `generate_video_prompt`, `video_prompt_seconds`,
and `video_prompt_candidate_count`. Old names have no aliases; existing outputs
are not renamed.
