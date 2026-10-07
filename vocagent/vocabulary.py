"""Single English target validation and coverage in recognized speech."""

from collections.abc import Sequence
import re
import unicodedata


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


def _tokens(text: str) -> tuple[str, ...]:
    """Normalize case/punctuation while preserving complete alphabetic words.

    CJK scripts need no spaces between words. Treat their characters as tokens
    so a requested phrase can occur within a longer sentence. Other scripts
    retain word boundaries: 'pear' must not match 'appear' or 'pearl'.
    """
    tokens = []
    word = []

    def flush():
        if word:
            tokens.append("".join(word))
            word.clear()

    for character in unicodedata.normalize("NFKC", text).casefold():
        if not (character.isalnum() or unicodedata.category(character).startswith("M")):
            flush()
        elif unicodedata.name(character, "").startswith((
            "CJK ", "HIRAGANA ", "KATAKANA ", "KATAKANA-HIRAGANA ", "HANGUL ",
        )):
            flush()
            tokens.append(character)
        else:
            word.append(character)
    flush()
    return tuple(tokens)


def missing_vocabulary(vocabulary: Sequence[str], transcript: str) -> tuple[str, ...]:
    """Return items whose complete normalized token sequence is absent.

    Matching allows case, Unicode width, and punctuation differences, but does
    not infer synonyms, homophones, translations, or inflected forms. Items
    without lexical characters cannot establish spoken coverage and are missing.
    """
    spoken = _tokens(transcript)
    missing = []
    for item in vocabulary:
        expected = _tokens(item)
        if not expected or not any(
            spoken[start:start + len(expected)] == expected
            for start in range(len(spoken) - len(expected) + 1)
        ):
            missing.append(item)
    return tuple(missing)
