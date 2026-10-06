1. Accept vocabulary words or phrases.
2. Generate, safety-check, and select a story outline.
3. Generate, safety-check, and select a storyboard.
4. For each video candidate attempt:
   - Generate a video from the selected storyboard.
   - Extract audio with FFmpeg and transcribe it into timestamped segments.
   - Validate the segments and save the audio and transcript beside the video.
   - Check visuals with `VideoSafetyModel` and spoken content with a separate
     `AudioSafetyModel`, passing the segment texts joined with newlines.
   - Keep the candidate only when both safety models pass.
5. Choose the best safe video with `VideoQualityModel`.
6. Copy the selected candidate's existing audio and transcript to `audio.wav`
   and `transcript.json`, and create ASS subtitles from its original segments.
7. Burn the subtitles into the selected video with FFmpeg.

The built-in text implementation uses Qwen3 for generation and quality judging,
and Qwen3Guard for response safety. The CLI defaults to finishing after step 3
with these built-in providers. Custom provider factories default to the full
video workflow; `--stop-after` overrides either CLI default. Python callers use
`Workflow.run(..., stop_after="storyboard")` to finish after step 3 with the
selected outline and storyboard. The Python API still defaults to the full
video workflow. Video and speech providers remain unimplemented.

For the outline, storyboard, and video stages, fill the configured number of
candidate slots with generation/safety attempts, then let a separate quality
judge choose from the safe candidates in slot order, even if only one survived.

Configure the stages independently with `outline_candidate_count`,
`storyboard_candidate_count`, and `video_candidate_count`. Each must be a positive
integer and defaults to 2. Candidate slots run sequentially by default. Optional
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

Each video attempt writes sibling `candidate-N-attempt-N.mp4`, `.wav`, and `.json`
files under `video_candidates/`. Both safety checks run after successful
transcription and validation; the audio safety model receives transcript text,
not raw audio. If either check fails, delete all three files and retry without
feedback. Retain every safe candidate and its sidecars. Reuse the winner's
transcript for subtitles without another speech-to-text call.

For each candidate, generation attempts remain capped at 4 sequential attempts,
regardless of the stage's candidate count. If the result is unsafe after 4
attempts, simply ignore the candidate. If no candidate succeeds, the system will
stop to allow human intervention.

Provider, transcript validation, and FFmpeg errors stop the run instead of being
treated as unsafe results or retried. New attempts stop while already-started
attempts finish. Video generation, visual and audio safety, video quality, and
speech-to-text integrations remain explicit unimplemented stubs until supplied
by the caller.
