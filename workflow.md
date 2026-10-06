1. given a vocabulary
2. llm generates story outline
3. llm generates storyboard
4. video model generates video
5. ffmpeg extracts audio
6. speech to text with timestamp
7. create ASS subtitle
8. ffmpeg combines to final video

For generating story outline, storyboard and video, use a safety + quality filter:
1. Generate the configured number of candidates for the stage.
2. For each candidate, check safety until pass.
3. Choose the best of the candidates that pass safety.

Configure the stages independently with `outline_candidate_count`,
`storyboard_candidate_count`, and `video_candidate_count`. Each must be a positive
integer and defaults to 2. Candidate slots run sequentially by default. Optional
concurrent generation uses up to the configured count of concurrent
generation/safety calls for that stage; stages remain sequential. In Python,
pass these counts as keyword arguments to `Workflow` and set
`concurrent_candidates=True` to enable concurrency.
The CLI exposes them as `--outline-candidate-count`,
`--storyboard-candidate-count`, and `--video-candidate-count`, plus
`--concurrent-candidates` to enable concurrency.

For LLM generated texts, use a separate agent to determine whether the outcome is safe. If it is unsafe, the reason of failure is passed to the generator to regenerate. Then use a separate agent to determine the best outcome.

For generated videos, use specialized models to check safety and choose winner. Unsafe candidate is simply discarded and regenerated.

For each candidate, generation attempts remain capped at 4 sequential attempts,
regardless of the stage's candidate count. If the result is unsafe after 4
attempts, simply ignore the candidate. If no candidate succeeds, the system will
stop to allow human intervention.
