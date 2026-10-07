1. Accept exactly one English target word, rejecting multiple words, phrases, and malformed
   input. Allow ASCII letters with internal apostrophes or hyphens, and treat the
   word as English without a dictionary lookup.
2. Generate, safety-check, and select an English story outline focused on one
   clear meaning of the target word.
3. Generate, safety-check, and select an English storyboard that keeps that
   meaning and uses English dialogue containing the target word.
4. For each video candidate attempt:
   - Generate a video from the selected storyboard.
   - Extract audio with FFmpeg and transcribe it in English into timestamped segments.
   - Validate the segments and save the audio and transcript beside the video.
   - Check visuals with `VideoSafetyModel` and spoken content with a separate
     `AudioSafetyModel`, passing the segment texts joined with newlines.
   - After both safety models pass, require the target word in the actual
     recognized speech, using normalized complete-word matching.
   - Keep the candidate only when both safety and spoken-vocabulary checks pass.
5. Choose the best safe video with `VideoQualityModel`.
6. Copy the selected candidate's existing audio and transcript to `audio.wav`
   and `transcript.json`, and create ASS subtitles from its original segments.
7. Burn the subtitles into the selected video with FFmpeg.

The default local implementation uses Qwen3-VL-32B for text generation, text
selection, and sampled-frame video selection; Qwen3Guard-Gen-8B for text and
complete recognized-transcript safety; MiniMax H3 for native audio/video;
Qwen3-ASR-1.7B plus ForcedAligner-0.6B for timestamped speech; and ShieldGemma2
4B for sampled-frame visual safety. The CLI and Python API default to the full
video workflow. `--stop-after storyboard` or `stop_after="storyboard"` stops after
selected text artifacts. Provider factories supply every workflow stage.
The CLI takes one positional word, for example `curious`. The Python API accepts
`Workflow(providers).run("curious", "output")` or the singleton sequence
`["curious"]` as its input.

Local storyboards use a plain storyboard prompt describing scene durations,
visuals, dialogue and audio cues, with a speaking budget based on clip duration.
The prompts require cartoon animation and an explicit cartoon style in the
storyboard. Language and clip duration are specified in system prompts.
Each generation makes one Qwen request; its response goes directly to the
separate Guard check and selection before video generation.

The local runtime serializes GPU inference and evicts the previous model family
before loading its replacement. It rejects CPU/disk offloading. H3 runs in a
separate dependency environment after the parent releases its model weights.
ShieldGemma is an image classifier applied at 2fps, including the clip tail;
sampled-frame checks do not guarantee observation of brief events between frames.
Model/runtime checks passed on hs; final spoken-vocabulary validation is in
progress, and the single-word changes still need hs validation. See
deploy/README.md for the deployment status.

For the outline, storyboard, and video stages, fill the configured number of
candidate slots with generation/safety/content attempts, then let a separate quality
judge choose from the safe candidates in slot order when multiple candidates
survive. If only one candidate survives, use it directly without a selection call.

Configure the stages independently with `outline_candidate_count`,
`storyboard_candidate_count`, and `video_candidate_count`. Each must be a positive
integer and defaults to 2. The local video judge permits at most two slots
(16 frames each, 32 total) within its verified context budget; this is checked
before generation. Candidate slots run sequentially by default. Optional
concurrent generation uses up to the configured count of concurrent candidate
attempts for that stage; stages remain sequential. Generation, safety, media,
and speech-to-text implementations must support concurrent calls on the same
instance when concurrency is enabled. In Python,
pass these counts as keyword arguments to `Workflow` and set
`concurrent_candidates=True` to enable concurrency.
The CLI exposes them as `--outline-candidate-count`,
`--storyboard-candidate-count`, and `--video-candidate-count`, plus
`--concurrent-candidates` to enable concurrency.

For LLM-generated texts, a separate safety agent checks the result. An unsafe
result's failure reason is passed to the next generation attempt in the same
slot. A separate quality agent chooses the best safe result.

Save each accepted outline and storyboard as UTF-8 text before quality selection
in `outline_candidates/` and `storyboard_candidates/`, respectively. Files are named
`candidate-N-attempt-M.txt`, using one-based slot and attempt numbers. Retain all
accepted alternatives, and write the selected texts to `outline.txt` and
`storyboard.txt`. These candidate files are also kept when stopping after the
storyboard or when a later provider or quality-selection call fails.

Each video attempt writes sibling `candidate-N-attempt-N.mp4`, `.wav`, and `.json`
files under `video_candidates/`. Both safety checks run after successful
transcription and validation; the audio safety model receives transcript text,
not raw audio. If either check fails, delete all three files and retry without
feedback. A safe clip missing the target word is also deleted and retried
within the same four-attempt budget. Matching allows case and punctuation
differences, not synonyms, homophones, translations, or inflected forms.
Retain every accepted candidate and its sidecars. Reuse the winner's
transcript for subtitles without another speech-to-text call.

For each candidate, workflow attempts remain capped at 4 sequential attempts,
regardless of the stage's candidate count. If the result fails safety or content checks after 4
attempts, simply ignore the candidate. If no candidate succeeds, the system will
stop to allow human intervention.

Provider, transcript validation, and FFmpeg errors stop the run instead of being
treated as unsafe results or retried. New attempts stop while already-started
attempts finish. Provider classes define extension interfaces; factories must
supply every stage, a video candidate-count validator, and a `close()` resource
hook. The default local factory supplies concrete implementations for every stage.
