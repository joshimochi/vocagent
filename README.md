# vocagent

A small Python implementation of [workflow.md](workflow.md), using only the
standard library. Python 3.10+ is required. FFmpeg must be on `PATH`, with the
`ass` filter (libass), `libx264`, and AAC support.

LLM generation, video generation, and speech-to-text are deliberately
unimplemented. Text safety/quality agents, visual safety/quality models, and a
separate transcript-based audio safety model are also explicit provider stubs:
there are no fake safety decisions or model API calls in the application.

## Workflow

1. Accept vocabulary words or phrases.
2. Generate and select a story outline.
3. Generate and select a storyboard from the selected outline.
4. For each video attempt, generate a video from the selected storyboard, extract
   mono, 16 kHz PCM audio with FFmpeg, and transcribe it into validated timestamped
   segments. Save the audio and transcript alongside that candidate.
5. Check the video's visuals and its audio transcript with separate safety models.
   Select the best video from the candidates that pass both checks.
6. Reuse the selected candidate's audio and transcript, and write ASS subtitles
   using the original segment timestamps.
7. Burn the subtitles into the selected video with FFmpeg, preserving its audio.

Each of the three generation stages has an independently configurable candidate
count, defaulting to two. A stage fills its candidate slots sequentially by
default. Enable concurrent candidate generation to use up to its configured count
of worker threads. Each slot gets at most four sequential generation/safety
attempts. A safe candidate ends that slot's retries; an
exhausted slot is ignored. Once all slots finish, the quality judge picks from
the safe candidates in slot order, even if only one survived. If none survived,
the workflow raises `HumanInterventionRequired` and stops before the next stage.

For text, the preceding safety failure reason is passed to the next attempt in
the same slot. Each successfully transcribed video attempt runs both visual and
audio safety checks. Failure of either check deletes that attempt's video, audio,
and transcript, and retries the slot without feedback. Generation, safety, and
judging are separate dependencies. Supply separate agents for text generation,
safety, and quality, and specialized models for visual safety, audio safety, and
video quality.

Provider, transcript validation, and FFmpeg errors stop the run; they are not
silently classified as unsafe candidates or retried. An error stops new attempts;
already-started attempts finish before the error is raised. Earlier artifacts
remain available for inspection after a failure. After human
intervention, start a new run in a new or empty directory; automatic resume is
not implemented.

## Connect providers

Implement the methods in
[`vocagent/providers.py`](vocagent/providers.py).
You can subclass the stubs or supply objects with matching methods. A video
provider must write a nonempty video with an audio track to the requested path.
When concurrent candidate generation is enabled, generation, safety,
speech-to-text, and media implementations must be thread-safe for up to the
configured candidate count of concurrent calls on the same instance for each
stage. Each video attempt receives distinct video, audio, and transcript paths.
Each stage's quality judge runs once after all slots finish; workflow stages
remain sequential.
Safety providers return `SafetyResult(safe=True)` or
`SafetyResult(safe=False, reason="...")`. Quality providers return a zero-based
index into the safe candidates passed to them. Speech-to-text returns
`TranscriptSegment(start=0.0, end=1.5, text="...")` objects with times in seconds.
`VideoSafetyModel.check(video_path)` assesses the visuals;
`AudioSafetyModel.check(transcript)` assesses spoken content from a string formed
by joining the segment texts with newlines. It receives neither raw audio nor
timestamped objects. An audio safety implementation is required: the default
stub raises an unimplemented error instead of passing the candidate.

Create a module such as `my_providers.py` with a factory that returns a
`Providers` instance containing your implementations:

```python
from vocagent.providers import Providers


def create_providers():
    return Providers(
        llm=MyLLMGenerator(),
        text_safety=MyTextSafetyAgent(),
        text_quality=MyTextQualityAgent(),
        video=MyVideoGenerator(),
        video_safety=MyVideoSafetyModel(),
        video_quality=MyVideoQualityModel(),
        speech_to_text=MySpeechToText(),
        audio_safety=MyAudioSafetyModel(),
    )
```

The `My...` classes above stand for your implementations; they are intentionally
not included. Then run from the repository root:

```sh
python3 -m vocagent sunrise curious journey \
  --providers my_providers:create_providers --output-dir output \
  --outline-candidate-count 6 --storyboard-candidate-count 3 --video-candidate-count 2
```

The three candidate count flags accept positive integers and each defaults to
`2` when omitted. They change the number of candidate slots, while each slot
still gets at most four generation/safety attempts.

Add `--concurrent-candidates` to enable concurrent candidate generation.

Without `--providers`, the CLI uses the stubs and stops at the first LLM call
with a clear unimplemented message. Exit codes are `0` for success, `1` for
provider/configuration/media errors, and `2` when all candidates fail safety and
human intervention is required or CLI arguments are invalid. The output directory
must be new or empty.

The same pipeline can be called from Python:

```python
from vocagent.pipeline import Workflow
from my_providers import create_providers

workflow = Workflow(
    create_providers(),
    outline_candidate_count=6,
    storyboard_candidate_count=3,
    video_candidate_count=2,
)
result = workflow.run(["sunrise", "curious"], "output")
print(result.final_video)
```

The three Python keyword arguments also accept positive integers and each
defaults to `2` when omitted. Add `concurrent_candidates=True` to the `Workflow`
constructor to enable concurrent candidate generation.

## macOS and Linux setup

Use Python 3.10+ and an FFmpeg build with the `ass` filter (libass), `libx264`,
and AAC support. Install them through your system's package manager if needed,
along with a font suitable for your subtitle language. FFmpeg and pip are not
bundled with this repository. Use `python3` consistently,
since a `python` command is not present on every system. Check the prerequisites:

```sh
python3 --version
ffmpeg -version
ffmpeg -hide_banner -h filter=ass
ffmpeg -hide_banner -h encoder=libx264
ffmpeg -hide_banner -h encoder=aac
```

Running `python3 -m vocagent` from the repository root requires no installation
and no Python runtime dependencies. To optionally install the `vocagent` command,
create a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

If creating the environment fails because `venv` or `ensurepip` is unavailable,
install your system's Python virtual-environment support first (the
`python3-venv` package on Debian/Ubuntu).

## Outputs

The output directory contains `vocabulary.json`, `outline.txt`, `storyboard.txt`,
safe candidates under `video_candidates/`, `audio.wav`, `transcript.json`,
`subtitles.ass`, and `final.mp4`. Each retained candidate has sibling
`candidate-N-attempt-N.mp4`, `.wav`, and `.json` files. The winning candidate's
WAV and JSON are copied to the top-level `audio.wav` and `transcript.json`;
selection does not run speech-to-text again. Other safe candidates and their
sidecars remain available for inspection. `WorkflowResult` includes the selected
video's path as well as the paths of the other main artifacts. Subtitle text and
timing come from the same validated segments that passed audio safety.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

Tests use deterministic fake providers to exercise the workflow without model
services. Media and subtitle tests cover FFmpeg commands, failures, and subtitle
formatting. Real audio extraction runs when FFmpeg is available; the real
subtitle-burning test also requires the `ass` filter and is skipped if it is
missing. Check support with `ffmpeg -hide_banner -h filter=ass`.
