# Generation and selection prompts

This is a summary of the current prompts, using a **15-second video**.
**“share” is only an example target word, not a fixed requirement.** Replace it
with the single English word supplied for each run. For this example, use one
meaning throughout: letting someone else use some of what you have, shown by a
child offering crayons to a friend.

The summaries below describe the prompt requirements; the source files contain
the exact wording. Input fields and candidate content are treated as data, not
instructions.

## 1. Outline generation

**Input example:**

```json
{"target_word": "share", "language": "English", "duration_seconds": 15}
```

Ask for a short, coherent vocabulary story for children that:

- Uses simple English and familiar situations to teach exactly one target word
  in one clear meaning, grounded in concrete actions, objects, and outcomes.
- Uses the supplied word unchanged and repeats it naturally. In this example,
  use “share”; do not substitute “sharing” for the target-word repetitions.
- Includes a title, setting, characters, and 2–3 brief plot beats with a clear
  ending, built around one simple event and one speaking character.
- Fits 15 seconds and keeps the entire outline under 100 words.
- Uses gentle curiosity or humor tied to the meaning, with a warm,
  age-appropriate tone. Avoid frightening scenes, humiliation, dangerous acts
  children might imitate, translations, and other vocabulary-learning goals.

**Output:** Only the outline.

## 2. Storyboard generation

**Input:** The same example target word, English language, and 15-second duration,
plus the selected outline in an `outline` field.

Ask for a concise storyboard that:

- Lists scene durations, visuals/actions, spoken dialogue or narration, and
  audio cues, all in English.
- Preserves the outline's core events, causal order, ending, and the target
  word's meaning. Keeps characters, key objects, setting, and visual style
  consistent.
- Gives each scene one main visible action with a clear outcome and an
  uncluttered focus. Makes the meaning understandable without reading.
- Speaks the target word unchanged and repeats it naturally while showing its
  meaning, using short, simple sentences without translations or extra teaching
  goals.
- Has scene durations totaling **15 seconds** and **at most 30 spoken English
  words across the entire video**. This is a ceiling, not a target; allow normal
  speech and short pauses for children to notice the action.
- Keeps the tone warm and age-appropriate, with background audio quiet beneath
  speech.

**Illustrative timing for “share” (example only):** 0–5 seconds: a child notices
a friend has no crayons; 5–10 seconds: the child offers crayons and says,
“I can share my crayons”; 10–15 seconds: both draw, and the same child says,
“It is fun to share.” This uses 10 spoken words.

**Output:** Only the storyboard. The selected storyboard is passed directly to
MiniMax H3 as its video-generation prompt, including its planned native audio;
there is no separate prompt rewrite or required H3-specific text schema.

## 3. Outline and storyboard selection

**Input:** `stage` (`outline` or `storyboard`), `target_word` (`share` in this
example), `language` (`English`), and `candidates`, each with a zero-based `index`
and its `text`. Candidates have already passed the separate text safety check.
The selection request has no separate duration field.

Ask the judge to select the best supplied candidate, prioritizing:

1. Correct, natural use and repetition of the unchanged target word in one
   meaning, made understandable through concrete events.
2. Simple language, clear cause and effect, consistent characters and objects,
   and a warm, age-appropriate tone.
3. Memorable cues and gentle humor connected to the meaning.

Mark down non-English text, translations, changes in meaning, extra vocabulary
goals, frightening scenes, humiliation, and dangerous imitable acts. For
outlines, judge the story idea, events, and ending. For storyboards, also judge
filmable actions, uncluttered scenes, speech tied to visible meaning, and enough
time for speech and observation. Judge only the supplied text; do not imagine
rendering quality or pronunciation.

**Output:** Only JSON such as `{"index": 0}`. The index must identify a supplied
candidate; zero is an example, not a preferred choice.

## 4. Video selection

**Input:** The selected storyboard and English language context, followed by
each candidate's sampled images labeled with its index and frame timestamps.
The target word and its intended meaning come from the storyboard.

Ask the judge to select the video whose visible actions and outcomes best
convey that meaning. For **“share” as an example**, look for evidence of the child
offering crayons and the friend using them. A storyboard description or an
on-screen word alone cannot replace visible evidence.

Then prefer clear visual focus, consistent characters and objects, fidelity to
the storyboard, a warm presentation, and memorable visual cues tied to the
meaning. Mark down shifts in meaning, extra teaching goals, frightening or
humiliating imagery, dangerous imitable acts, clutter, and visible generation
defects. Judge only the sampled frames; do not infer unseen events, motion
smoothness, or pronunciation.

**Output:** Only JSON such as `{"index": 0}`, using a valid zero-based candidate
index. Visual safety, transcript safety, and spoken target-word coverage are
checked separately before selection; subtitles are added afterward.

## Prompt sources and retry behavior

- [Generation prompts](vocagent/prompts/generation.py): outline and storyboard
  instructions; the spoken-word ceiling is `floor(duration_seconds * 2)`.
- [Selection prompts](vocagent/prompts/quality.py): text and sampled-frame
  judging criteria and output format.
- [Request construction](vocagent/qwen.py): each generation or selection call
  uses a fresh system prompt and user payload. Text generation uses temperature
  0.7; selection uses temperature 0.
- [Safety retry prompts](vocagent/prompts/safety.py): a text retry adds
  `safety_feedback` to the input and appends “Revise to resolve the safety
  feedback.” to the generation prompt. Video retries reuse the selected
  storyboard without passing safety feedback to the video generator.
