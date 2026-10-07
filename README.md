# vocagent

A vocabulary-to-video agent with separate generation, safety, and selection calls.
The default provider factory is `vocagent.local:create_providers`, intended to run
entirely on `hs`, using physical GPUs **4–7** (four MTT S4000, 48 GiB each).

Model prompts live in [`vocagent/prompts/`](vocagent/prompts/), grouped into
generation, safety, quality selection, and diagnostics modules. Edit the prompt
text and duration-specific builders there.

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
uses an earlier prompt with H3's native dialogue format. See [deployment status](deploy/README.md)
for the exact completed checks and remaining work.

## Run on hs

After deploying and setting up the environment as described in
[deploy/README.md](deploy/README.md):

```sh
cd /data/group6/vocagent/run-a406/app
bash deploy/run_hs.sh sunrise curious journey \
  --output-dir /data/group6/vocagent/outputs/example
```

The CLI defaults to a complete subtitled video. `--stop-after storyboard`
stops after the selected text artifacts. The output directory must be new or
empty. Candidate counts default to two per stage:

```sh
bash deploy/run_hs.sh sunrise curious journey \
  --outline-candidate-count 2 --storyboard-candidate-count 2 \
  --video-candidate-count 2 --output-dir /data/group6/vocagent/outputs/example-2
```

The Python API:

```python
from vocagent.local import create_providers
from vocagent.pipeline import Workflow

providers = create_providers()
try:
    result = Workflow(providers).run(["sunrise", "curious"], "output")
    print(result.final_video)
finally:
    providers.close()
```

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
4. Run Qwen3-ASR plus forced alignment. Preserve the recognized text and its
   aligned timestamps in the saved transcript and subtitle cues.
5. Check visual frames with ShieldGemma 2 and the **complete recognized
   transcript** with Qwen3Guard. Both checks must pass.
6. Require every requested word or phrase in the actual ASR transcript. Missing
   vocabulary rejects the clip and uses the same bounded retry budget.
7. Use Qwen3-VL to select among accepted videos from actual, timestamped image frames.
8. Reuse the selected transcript; write ASS subtitles and burn them into the
   selected video with FFmpeg, preserving its audio.

ShieldGemma 2 is an **image** safety classifier, applied throughout the video at
2 frames/second, including the final nominal frame. It checks the model's
`dangerous`, `sexual`, and `violence` policies; any violation probability at or
above 0.5 rejects the candidate. Sampling cannot guarantee detection of brief
events between samples. Clips exceeding the sampling budget are rejected with an
error rather than silently shortening coverage. VL video judging samples up to
16 frames per candidate. The local 16K-context configuration allows two video
candidate slots (32 frames total), validated before any generation. Video judging
evaluates visible evidence; it does not listen to audio. Spoken vocabulary matching uses normalized complete
words or phrases, allowing case and punctuation differences. It does not infer
synonyms, homophones, translations, or inflected forms.

Each candidate slot gets at most four generation/safety/content attempts. Unsafe text
supplies feedback to the next attempt. Rejected video attempts and their audio
and transcript files are deleted and regenerated without feedback. Candidates
passing safety and spoken-vocabulary coverage are retained; an exhausted slot is ignored. A separate quality judge
still runs when only one candidate survives. If none survive, the workflow
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
| `MINIMAX_H3_WIDTH`, `MINIMAX_H3_HEIGHT` | `1344`, `768` |
| `MINIMAX_H3_SECONDS`, `MINIMAX_H3_STEPS` | `10`, `50` |
| `MINIMAX_H3_TIMEOUT_SECONDS` | `14400` per video candidate |

H3 duration is 5–15 seconds, rounded to its native video-frame grid. The default
1344×768, 10-second shape passed a full GPU capacity check. Native-resolution
generation is slow on S4000: allow roughly 2.5 hours for a 10-second, 50-point
render based on the measured single iteration; the default worker limit is four
hours and can be overridden. Many vocabulary
items may not fit naturally in a single clip; choose a small vocabulary set.

Provider factories must return a complete `Providers` bundle for all workflow
stages. Local runs use checkpoint directories and do not need API endpoints,
keys, or SSH tunnels.

## Outputs and tests

Outputs include `vocabulary.json`, `outline.txt`, `storyboard.txt`, safe video
candidates and sidecars in `video_candidates/`, the selected `audio.wav` and
`transcript.json`, `subtitles.ass`, and `final.mp4`. Earlier artifacts remain
available after a provider failure.

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
