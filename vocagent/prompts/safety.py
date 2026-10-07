"""Native Guard request context and generation retry instructions."""

TEXT_SAFETY_PROMPT = "Write a vocabulary-learning story {stage}."
AUDIO_SAFETY_PROMPT = (
    "Write the spoken dialogue and narration for a vocabulary-learning video."
)
SAFETY_FEEDBACK_PROMPT = " Revise to resolve the safety feedback."
SAFETY_RETRY_FEEDBACK = (
    "The previous candidate failed the safety check. "
    "Generate a safer alternative."
)
