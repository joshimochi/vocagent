# vocagent

A vocabulary-to-video agent that takes exactly one English word and makes a video
focused on one clear meaning of that word, with separate generation, safety, and
selection calls.
The default provider factory is `vocagent.local:create_providers`, intended to run
entirely on `hs`, using physical GPUs **4–7** (four MTT S4000, 48 GiB each).

Model prompts live in [`vocagent/prompts/`](vocagent/prompts/), grouped into
generation, safety, and quality selection modules. Edit the prompt text and
duration-specific builders there. The opt-in diagnostic prompt lives in
[`vocagent/diagnostics.py`](vocagent/diagnostics.py).

| Task | Local checkpoint |
| --- | --- |
| Outline, storyboard, text selection, visual video selection | Qwen/Qwen3-VL-32B-Instruct, BF16 |
| Text and complete transcript safety | Qwen/Qwen3Guard-Gen-8B, BF16 |
| Video with native audio | MiniMaxAI/MiniMax-H3, full released weights |
| Speech recognition | Qwen/Qwen3-ASR-1.7B, BF16 |
| Word timestamps | Qwen/Qwen3-ForcedAligner-0.6B, BF16 |
| Visual safety | google/shieldgemma-2-4b-it, BF16 |

**Final content validation is in progress.** The complete technical pipeline and
native-resolution H3 capacity checks passed on `hs`. The latest full validation run
uses an earlier prompt with H3's native dialogue format; the single-word changes
have not yet been validated on `hs`. See [deployment status](deploy/README.md)
for the exact completed checks and remaining work.

## Run on hs

After deploying and setting up the environment as described in
[deploy/README.md](deploy/README.md):

```sh
cd /data/group6/vocagent/run-a406/app
bash deploy/run_hs.sh curious \
  --output-dir /data/group6/vocagent/outputs/example
```

Each run accepts exactly one English target word. Multiple words and phrases are rejected.
The input permits ASCII letters with internal apostrophes or hyphens, such as
`don't` or `well-known`; it is treated as English without a dictionary lookup.
Generation and selection prompts require cartoon animation, English text and
spoken dialogue, with the outline, storyboard, and video focused on one clear
meaning of the target word. The storyboard explicitly states the cartoon style.
Language and clip duration are specified in system prompts. Speech recognition
also defaults to English.

The CLI defaults to a complete subtitled video. `--stop-after storyboard`
stops after the selected text artifacts. The output directory must be new or
empty. Candidate counts default to two per stage:

```sh
bash deploy/run_hs.sh curious \
  --outline-candidate-count 2 --storyboard-candidate-count 2 \
  --video-candidate-count 2 --output-dir /data/group6/vocagent/outputs/example-2
```

The Python API:

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

The Python API also accepts a singleton sequence such as `["curious"]`.

## Memory and model choice

The runtime loads a model family on demand and destroys the previous weights
before loading its replacement. It checks every model parameter and placement
hook, rejecting CPU, disk, and meta placement or weight offloading. GPU memory
budgets reserve runtime headroom; insufficient memory stops the run. There is no
smaller-model or CPU inference fallback. CPU tokenization, checkpoint reading,
media decoding/encoding, and input preprocessing are ordinary supporting work.

The 32B model is the largest dense Qwen3-VL release. The 235B checkpoint exceeds
this allocation in BF16 and FP8; the installed S4000 stack has not demonstrated a
working GPU-only 4-bit route. The other Qwen checkpoints are their largest
released family sizes; ShieldGemma 2 provides a 4B checkpoint. H3 uses the full
released encoder and diffusion weights in BF16 and its decoders in FP32 (video decoding uses FP16 autocast on MUSA).
Full H3 GPU residency and low-resolution native audio/video generation passed on hs.
Normal-resolution memory capacity also passed. The latest full validation run uses
the earlier native H3 prompt format to verify spoken vocabulary.

Separate stateless text generation and selection agents share the same VL
weights. Qwen3-VL uses uncached SDPA on MUSA because cached decoding failed
correctness checks on the installed stack. ASR and its small aligner reside together. H3 runs in a separate Python
process because its dependency versions conflict with the speech package;
the shared GPU lock covers that process through its exit. Concurrent candidate
threads can preprocess in parallel, but model inference remains serialized.

## Workflow and safety behavior

1. Generate outlines; check each with Qwen3Guard; select the best safe outline.
2. Generate storyboards with scene durations, visuals, dialogue and audio cues,
   using a speaking budget matched to the clip duration. Check each with Guard
   and select the best safe storyboard.
3. Generate a video with native audio. Extract mono 16 kHz PCM audio with FFmpeg.
4. Run Qwen3-ASR in English plus forced alignment. Preserve the recognized text and its
   aligned timestamps in the saved transcript and subtitle cues.
5. Check visual frames with ShieldGemma 2 and the **complete recognized
   transcript** with Qwen3Guard. Both checks must pass.
6. Use Qwen3-VL to select among accepted videos from actual, timestamped image frames.
7. Reuse the selected transcript; write ASS subtitles and burn them into the
   selected video with FFmpeg, preserving its audio.

ShieldGemma 2 is an **image** safety classifier, applied throughout the video at
2 frames/second, including the final nominal frame. It checks the model's
`dangerous`, `sexual`, and `violence` policies; any violation probability at or
above 0.5 rejects the candidate. Sampling cannot guarantee detection of brief
events between samples. Clips exceeding the sampling budget are rejected with an
error rather than silently shortening coverage. VL video judging samples up to
16 frames per candidate. The local 16K-context configuration allows two video
candidate slots (32 frames total), validated before any generation. Video judging
evaluates visible evidence; it does not listen to audio. ASR provides recognized
speech for transcript safety and subtitles.

Each candidate slot gets at most four generation/safety attempts. Unsafe text
supplies feedback to the next attempt. Rejected video attempts and their audio
and transcript files are deleted and regenerated without feedback. Candidates
passing safety are retained; an exhausted slot is ignored. A separate quality judge
runs when multiple candidates survive; a sole survivor is used directly without
a selection call. If none survive, the workflow
raises `HumanInterventionRequired`.

Provider errors, missing speech, incomplete alignment, malformed moderation,
truncated text, invalid selections, GPU failures, and media errors stop the run;
they are never converted into fabricated safe results. Automatic resume is not
implemented; start another run in a new output directory after resolving a failure.

## Configuration

All model paths are local directories. `VOCAGENT_DEPLOY_ROOT` defaults to
`/data/group6/vocagent`. The deployment container exposes only GPUs 4–7.

| Variable | Default |
| --- | --- |
| `QWEN_VL_MODEL` | `$VOCAGENT_DEPLOY_ROOT/models/Qwen3-VL-32B-Instruct` |
| `QWEN_GUARD_MODEL_PATH` | `$VOCAGENT_DEPLOY_ROOT/models/Qwen3Guard-Gen-8B` |
| `QWEN_ASR_MODEL` | `$VOCAGENT_DEPLOY_ROOT/models/Qwen3-ASR-1.7B` |
| `QWEN_ALIGNER_MODEL` | `$VOCAGENT_DEPLOY_ROOT/models/Qwen3-ForcedAligner-0.6B` |
| `SHIELDGEMMA_MODEL` | `$VOCAGENT_DEPLOY_ROOT/models/shieldgemma-2-4b-it` |
| `MINIMAX_H3_MODEL` | `/data/group6/minimax-h3/models/MiniMax-H3` |
| `MINIMAX_H3_PYTHON` | `$VOCAGENT_DEPLOY_ROOT/run-a406/h3-venv/bin/python` |
| `MINIMAX_H3_STEPS` | `50` |
| `MINIMAX_H3_TIMEOUT_SECONDS` | `14400` per video candidate |

H3 output is fixed in landscape at a width of 1344, a height of 768, and a requested duration
of 15 seconds. Duration is rounded to its native video-frame grid within the
15-second limit (345 frames at 24 fps, or 14.375 seconds). Outline and storyboard
prompts use the same 15-second budget. These settings are hardcoded.
The earlier 1344×768, 10-second shape passed a full GPU capacity check. Native-resolution
generation is slow on S4000: allow roughly 2.5 hours for a 10-second, 50-point
render based on the measured single iteration; the default worker limit is four
hours and can be overridden. Each clip teaches the single target word through
one clear meaning and a compact scene.

Provider factories must return a complete `Providers` bundle for all workflow
stages. Local runs use checkpoint directories and do not need API endpoints,
keys, or SSH tunnels.

## Outputs and tests

Outputs include `vocabulary.json`, the selected `outline.txt` and `storyboard.txt`,
all accepted text candidates in `outline_candidates/` and `storyboard_candidates/`,
safe video candidates and sidecars in `video_candidates/`, the selected
`audio.wav` and `transcript.json`, `subtitles.ass`, `final.mp4`, and `benchmark.json`.
Text candidates are UTF-8 files named `candidate-N-attempt-M.txt`, with one-based
candidate slot and attempt numbers. They are saved after passing safety and before
quality selection, including accepted alternatives that were not selected.
`--stop-after storyboard` also retains both text candidate directories. Earlier
artifacts, including accepted text candidates, remain available after a provider
or quality-selection failure.

Every started workflow writes `benchmark.json`, including storyboard-only runs
and failed runs. `result.benchmarks` points to the report. It records each
generation, safety check, selection, audio extraction, and
subtitle operation, with candidate slot and attempt numbers for retries. The
report contains per-call measurements, per-step aggregates, and a total for the
workflow, including status and any failure. Input and output preflight failures
that occur before the workflow starts do not create a report.

Wall time is split into model loading, model running, and other work. Model
loading measures checkpoint initialization and device placement; model running
measures actual inference, including GPU synchronization. Other work includes
preprocessing, media I/O, unloading, queueing for the shared GPU lock, and worker
startup. Models already resident in memory need no new loading time. Each step's
wall time sums its calls, so overlapping candidate threads can make the sum of
step times larger than the workflow's actual elapsed time. The total reports
both values. Workflow elapsed time starts after input/output validation and
provider creation, and ends before the caller closes providers. Per-call CPU
time measures the executing thread; total process CPU time covers all workflow
threads, with isolated model worker CPU time reported separately. Token/frame
counts, processed audio/video duration, and throughput are included when
available; model throughput uses model running time.
GPU memory reports peak allocated and reserved bytes by device, using the maximum
observed peak across calls rather than adding memory usage. Custom providers
still receive workflow wall and CPU timing; model-specific metrics require
provider instrumentation.
H3 worker measurements are imported into its generation step. If a hard timeout
kills the worker before it saves a report, that call records
`worker_benchmark_status: unavailable` rather than treating its missing model
measurements as measured zero time.

```sh
python3 -m unittest discover -s tests -v
```

Core workflow tests need only Python 3.10+. Real media tests require FFmpeg and
FFprobe with libass, H.264, and AAC. Numerical H3 equivalence tests require
PyTorch; real model smoke tests require the `hs` environments and checkpoints.
Fake-provider tests alone do not establish model/runtime compatibility.

Official references: [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct),
[Qwen3Guard](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B),
[Qwen3-ASR and aligner](https://github.com/QwenLM/Qwen3-ASR),
[ShieldGemma 2](https://huggingface.co/google/shieldgemma-2-4b-it),
[H3 MUSA recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-MUSA.md).
