"""Bounded safety retries followed by an independent quality decision."""

from collections.abc import Callable, Sequence
from typing import TypeVar

from .models import SafetyResult


T = TypeVar("T")


class HumanInterventionRequired(RuntimeError):
    """All candidate slots failed safety checks and the workflow must stop."""


def select_candidate(
    *,
    stage: str,
    generate: Callable[[int, int, str | None], T],
    check_safety: Callable[[T], SafetyResult],
    choose_best: Callable[[Sequence[T]], int],
    pass_feedback: bool,
) -> T:
    """Choose among four slots, each allowed at most four safety attempts.

    The generator receives a one-based slot and attempt, plus the previous
    unsafe reason when feedback is enabled. The quality judge receives only
    the safe survivors and returns a zero-based index into that sequence.
    Provider exceptions propagate without being treated as safety failures.
    """
    accepted: list[T] = []
    for slot in range(1, 5):
        feedback = None
        for attempt in range(1, 5):
            candidate = generate(slot, attempt, feedback)
            safety = check_safety(candidate)
            if (
                not isinstance(safety, SafetyResult)
                or type(safety.safe) is not bool
                or not isinstance(safety.reason, str)
            ):
                raise ValueError(
                    f"{stage}: the safety checker must return a SafetyResult "
                    "with a boolean safe value and a string reason."
                )
            if safety.safe:
                accepted.append(candidate)
                break
            if pass_feedback:
                feedback = safety.reason if safety.reason.strip() else (
                    "The previous candidate failed the safety check. "
                    "Generate a safer alternative."
                )

    if not accepted:
        raise HumanInterventionRequired(
            f"{stage}: no safe candidate after four slots with four attempts "
            "each; human intervention is required."
        )

    index = choose_best(tuple(accepted))
    if type(index) is not int or not 0 <= index < len(accepted):
        raise ValueError(
            f"{stage}: the quality judge must return an integer index from "
            f"0 to {len(accepted) - 1}; got {index!r}."
        )
    return accepted[index]
