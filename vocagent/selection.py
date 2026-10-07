"""Bounded safety/content retries followed by an independent quality decision."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import logging
from threading import Event
from typing import TypeVar

from .models import SafetyResult
from .prompts.safety import SAFETY_RETRY_FEEDBACK


T = TypeVar("T")
log = logging.getLogger(__name__)


class HumanInterventionRequired(RuntimeError):
    """All candidate slots failed acceptance checks and the workflow must stop."""


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


def validate_selection_index(index: int, candidate_count: int, stage: str) -> int:
    """Validate a quality decision at the provider boundary."""
    if type(index) is not int or not 0 <= index < candidate_count:
        raise ValueError(
            f"{stage}: the quality judge must return an integer index from "
            f"0 to {candidate_count - 1}; got {index!r}."
        )
    return index


def select_candidate(
    *,
    stage: str,
    generate: Callable[[int, int, str | None], T],
    check_safety: Callable[[T], SafetyResult],
    choose_best: Callable[[Sequence[T]], int],
    pass_feedback: bool,
    candidate_count: int = 2,
    concurrent_candidates: bool = False,
    check_content: Callable[[T], str | None] | None = None,
    on_accept: Callable[[int, int, T], None] | None = None,
) -> T:
    """Choose among configurable slots with up to four attempts each.

    The generator receives a one-based slot and attempt, plus the previous
    unsafe reason when feedback is enabled. The quality judge receives only
    the safe survivors in slot order and returns a zero-based index into that
    sequence. A sole survivor is returned without quality judging.
    Slots run sequentially by default. With concurrent_candidates=True,
    generation and safety callbacks must support up to candidate_count concurrent calls.
    Optional check_content runs only after safety passes and returns None to
    accept or a nonempty rejection reason. Content and safety rejections share
    the same four-attempt budget; provider errors still stop the workflow.
    Optional on_accept receives the one-based slot, attempt, and accepted
    candidate before quality judging, and may run concurrently across slots.
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
                    log.info("%s slot %d attempt %d passed safety", stage, slot, attempt)
                    reason = check_content(candidate) if check_content is not None else None
                    if reason is None:
                        if on_accept is not None:
                            on_accept(slot, attempt, candidate)
                        return [candidate]
                    if not isinstance(reason, str) or not reason.strip():
                        raise ValueError(
                            f"{stage}: the content checker must return None or a nonempty reason."
                        )
                    log.warning(
                        "%s slot %d attempt %d rejected for content: %r",
                        stage, slot, attempt, reason,
                    )
                else:
                    log.warning(
                        "%s slot %d attempt %d rejected: %r", stage, slot, attempt,
                        safety.reason or "Safety checker supplied no reason.",
                    )
                    reason = safety.reason if safety.reason.strip() else SAFETY_RETRY_FEEDBACK
                if pass_feedback:
                    feedback = reason
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
            f"{stage}: no acceptable candidate after {candidate_count} slots with four attempts "
            "each; human intervention is required."
        )

    if len(accepted) == 1:
        return accepted[0]

    index = validate_selection_index(choose_best(tuple(accepted)), len(accepted), stage)
    return accepted[index]
