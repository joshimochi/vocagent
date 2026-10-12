# Generation and selection prompts

Prompts teach children one English word through a short cartoon story. The source
files contain the exact instructions; this page explains how they are used.

| Task | Prompt source | Result |
| --- | --- | --- |
| Generate an outline | [`generation.py`](vocagent/prompts/generation.py), `outline_prompt()` | A story outline under 100 words |
| Generate a video prompt | [`generation.py`](vocagent/prompts/generation.py), `video_prompt()` and the [H3 guide](vocagent/prompts/h3_prompt_guide.md) | The three H3 fields below |
| Choose an outline or video prompt | [`quality.py`](vocagent/prompts/quality.py), `QUALITY_PROMPT` | `{"index": <zero-based integer>}` |
| Choose a video | [`quality.py`](vocagent/prompts/quality.py), `VIDEO_QUALITY_PROMPT` | `{"index": <zero-based integer>}` |
| Check text or speech safety | [`safety.py`](vocagent/prompts/safety.py) and [`qwen.py`](vocagent/qwen.py) | Qwen3Guard's native safety verdict |

## Generation

Each request has a system prompt and a JSON user message. Input fields are data,
not instructions. For example, an outline request supplies:

```json
{"target_word": "share"}
```

A video-prompt request also supplies the selected outline:

```json
{"target_word": "share", "outline": "<selected outline>"}
```

Both stages use simple English, one clear meaning of the unchanged target word,
concrete actions, natural repetition, and a warm tone. Outlines use one simple
event, one speaking character, and 2–3 brief plot beats with a clear ending.
Video prompts preserve that story and add timing, visuals, speech, and audio cues.

The video-prompt builder loads the full bundled H3 guide with `importlib.resources`,
then appends the vocabulary task, configured duration, and speech budget:
`floor(seconds × 2)` words, or at most 30 words for a 15-second video.
Speech must leave time for pauses and visible action.

Video prompts return these fields in order, without Markdown fences or commentary:

```text
integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
```

The selected prompt goes to MiniMax H3 unchanged. See the [H3 guide](vocagent/prompts/h3_prompt_guide.md)
for shot timing, speaker IDs, dialogue markup, and sound formatting.

## Selection and safety

Text selection receives `stage` (`outline` or `video_prompt`), `target_word`, and
`candidates`, each with an `index` and `text`. It favors clear meaning, natural
word use, coherent events, consistent characters, and age-appropriate content.
Video prompts are also judged on visible actions and time for speech and pauses.

Video selection receives the selected `video_prompt` and timestamped frames for
each candidate. It judges visible meaning, continuity, prompt fidelity, and
rendering defects. Sampled frames cannot establish pronunciation, unseen events,
or motion smoothness.

Qwen3Guard checks generated text and transcribed speech with its native response
template. Only a valid `Safe` verdict passes. Generation retries can include
`safety_feedback`. Transcription supports audio safety and subtitles; the pipeline
does not enforce exact spoken target-word coverage.
