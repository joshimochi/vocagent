"""Bounded safety retries followed by an independent quality decision."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import TypeVar

from .models import SafetyResult


T = TypeVar("T")


class HumanInterventionRequired(RuntimeError):
    """All candidate slots failed safety checks and the workflow must stop."""


def validate_safety_result(safety: SafetyResult, stage: str) -> SafetyResult:
    """Reject malformed provider decisions before accepting or discarding work."""
    if (
        not isinstance(safety, SafetyResult)
        or type(safety.safe) is not bool
        or not isinstance(safety.reason, str)
    ):
        raise ValueError(
            f"{stage}: the safety checker must return a SafetyResult "
            "with a boolean safe value and a string reason."
        )
    return safety


def select_candidate(
    *,
    stage: str,
    generate: Callable[[int, int, str | None], T],
    check_safety: Callable[[T], SafetyResult],
    choose_best: Callable[[Sequence[T]], int],
    pass_feedback: bool,
    candidate_count: int = 2,
    concurrent_candidates: bool = False,
) -> T:
    """Choose among configurable slots with up to four safety attempts each.

    The generator receives a one-based slot and attempt, plus the previous
    unsafe reason when feedback is enabled. The quality judge receives only
    the safe survivors in slot order and returns a zero-based index into that
    sequence. Slots run sequentially by default. With concurrent_candidates=True,
    generation and safety callbacks must support up to candidate_count concurrent calls.
    Each slot's retries remain sequential. Provider exceptions stop new attempts
    and propagate after in-flight work finishes, without becoming safety failures.
    """
    if type(candidate_count) is not int or candidate_count < 1:
        raise ValueError(f"{stage}: candidate_count must be a positive integer.")
    stopped = Event()

    def generate_slot(slot: int) -> list[T]:
        feedback = None
        try:
            for attempt in range(1, 5):
                if stopped.is_set():
                    return []
                candidate = generate(slot, attempt, feedback)
                safety = validate_safety_result(check_safety(candidate), stage)
                if safety.safe:
                    return [candidate]
                if pass_feedback:
                    feedback = safety.reason if safety.reason.strip() else (
                        "The previous candidate failed the safety check. "
                        "Generate a safer alternative."
                    )
        except Exception:
            stopped.set()
            raise
        return []

    if concurrent_candidates:
        with ThreadPoolExecutor(max_workers=candidate_count) as executor:
            slot_results = list(executor.map(generate_slot, range(1, candidate_count + 1)))
    else:
        slot_results = [generate_slot(slot) for slot in range(1, candidate_count + 1)]
    accepted = [candidate for survivors in slot_results for candidate in survivors]

    if not accepted:
        raise HumanInterventionRequired(
            f"{stage}: no safe candidate after {candidate_count} slots with four attempts "
            "each; human intervention is required."
        )

    index = choose_best(tuple(accepted))
    if type(index) is not int or not 0 <= index < len(accepted):
        raise ValueError(
            f"{stage}: the quality judge must return an integer index from "
            f"0 to {len(accepted) - 1}; got {index!r}."
        )
    return accepted[index]
