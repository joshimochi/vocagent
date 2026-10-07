"""Single English target validation."""

from collections.abc import Sequence
import re


def normalize_target_word(vocabulary: str | Sequence[str]) -> str:
    """Trim one English word, accepting legacy singleton vocabulary sequences.

    Validate word shape rather than consulting a dictionary. Internal
    apostrophes and hyphens permit contractions and compound words.
    """
    message = (
        "Provide exactly one English word, using letters with optional "
        "internal apostrophes or hyphens."
    )
    if isinstance(vocabulary, str):
        word = vocabulary
    elif isinstance(vocabulary, Sequence) and len(vocabulary) == 1:
        word = vocabulary[0]
    else:
        raise ValueError(message)
    if not isinstance(word, str):
        raise ValueError(message)
    word = word.strip()
    if not re.fullmatch(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*", word):
        raise ValueError(message)
    return word
