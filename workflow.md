1. given a vocabulary
2. llm generates story outline
3. llm generates storyboard
4. video model generates video
5. ffmpeg extracts audio
6. speech to text with timestamp
7. create ASS subtitle
8. ffmpeg combines to final video

For generating story outline, storyboard and video, use a safety + quality filter:
1. Generate 4 candidates sequentially by default, with optional concurrent generation using up to 4 worker threads.
2. For each candidate, check safety until pass.
3. Choose the best of the 4 candidates.

For LLM generated texts, use a separate agent to determine whether the outcome is safe. If it is unsafe, the reason of failure is passed to the generator to regenerate. Then use a separate agent to determine the best outcome.

For generated videos, use specialized models to check safety and choose winner. Unsafe candidate is simply discarded and regenerated.

For each candidate, generation attempt is capped at 4. If the result is unsafe after 4 attempts, simply ignore the candidate. If no candidate succeed, the system will stop to allow human intervention.
