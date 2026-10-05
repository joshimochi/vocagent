"""Tests for the workflow's bounded safety and quality decisions."""

import unittest
from threading import Barrier, Event, Lock
from unittest.mock import Mock

from vocagent.models import SafetyResult
from vocagent.selection import (
    HumanInterventionRequired,
    select_candidate,
)


class SelectionTests(unittest.TestCase):
    def test_all_four_safe_slots_reach_independent_quality_judge(self):
        generate = Mock(side_effect=lambda slot, attempt, feedback: f"slot {slot}")
        check = Mock(return_value=SafetyResult(True))
        judge = Mock(return_value=2)

        result = select_candidate(
            stage="outline", generate=generate, check_safety=check,
            choose_best=judge, pass_feedback=True,
        )

        self.assertEqual(result, "slot 3")
        self.assertCountEqual(
            generate.call_args_list,
            [((slot, 1, None),) for slot in range(1, 5)],
        )
        self.assertEqual(check.call_count, 4)
        judge.assert_called_once_with(("slot 1", "slot 2", "slot 3", "slot 4"))

    def test_text_retries_receive_latest_reason_and_reset_for_next_slot(self):
        calls = []

        def generate(slot, attempt, feedback):
            calls.append((slot, attempt, feedback))
            return slot, attempt

        def check(candidate):
            slot, attempt = candidate
            if slot == 1 and attempt < 3:
                return SafetyResult(False, f"unsafe attempt {attempt}")
            return SafetyResult(True)

        judge = Mock(return_value=0)
        result = select_candidate(
            stage="storyboard", generate=generate, check_safety=check,
            choose_best=judge, pass_feedback=True,
        )

        self.assertEqual(result, (1, 3))
        self.assertEqual(sorted(calls), [
            (1, 1, None), (1, 2, "unsafe attempt 1"),
            (1, 3, "unsafe attempt 2"), (2, 1, None),
            (3, 1, None), (4, 1, None),
        ])
        judge.assert_called_once_with(((1, 3), (2, 1), (3, 1), (4, 1)))

    def test_slots_generate_and_check_concurrently_with_ordered_survivors(self):
        generation_barrier = Barrier(4, timeout=5)
        safety_barrier = Barrier(4, timeout=5)
        finished = {slot: Event() for slot in range(1, 5)}
        calls = []
        completion_order = []
        lock = Lock()

        def generate(slot, attempt, feedback):
            with lock:
                calls.append((slot, attempt, feedback))
            if attempt == 1:
                generation_barrier.wait()
            return slot, attempt

        def check(candidate):
            slot, attempt = candidate
            if attempt == 1:
                safety_barrier.wait()
            if attempt < slot:
                return SafetyResult(False, f"slot {slot} attempt {attempt}")
            if slot < 4:
                self.assertTrue(finished[slot + 1].wait(timeout=5))
            with lock:
                completion_order.append(slot)
            finished[slot].set()
            return SafetyResult(True)

        judge = Mock(return_value=1)
        result = select_candidate(
            stage="outline", generate=generate, check_safety=check,
            choose_best=judge, pass_feedback=True,
        )

        self.assertEqual(result, (2, 2))
        self.assertEqual(completion_order, [4, 3, 2, 1])
        judge.assert_called_once_with(((1, 1), (2, 2), (3, 3), (4, 4)))
        for slot in range(1, 5):
            self.assertEqual(
                [call for call in calls if call[0] == slot],
                [(slot, attempt, None if attempt == 1 else f"slot {slot} attempt {attempt - 1}")
                 for attempt in range(1, slot + 1)],
            )

    def test_video_retries_never_receive_feedback(self):
        generate = Mock(side_effect=lambda slot, attempt, feedback: (slot, attempt))

        result = select_candidate(
            stage="video", generate=generate,
            check_safety=lambda candidate: SafetyResult(
                candidate[1] == 2, "unsuitable content",
            ),
            choose_best=lambda candidates: 3, pass_feedback=False,
        )

        self.assertEqual(result, (4, 2))
        self.assertEqual(generate.call_count, 8)
        self.assertTrue(all(call.args[2] is None for call in generate.call_args_list))

    def test_exhausted_slots_are_ignored_and_single_survivor_is_judged(self):
        generate = Mock(side_effect=lambda slot, attempt, feedback: (slot, attempt))
        judge = Mock(return_value=0)

        result = select_candidate(
            stage="outline", generate=generate,
            check_safety=lambda candidate: SafetyResult(candidate == (3, 4), "unsafe"),
            choose_best=judge, pass_feedback=True,
        )

        self.assertEqual(result, (3, 4))
        self.assertEqual(generate.call_count, 16)
        judge.assert_called_once_with(((3, 4),))
        for slot in range(1, 5):
            slot_calls = [call.args for call in generate.call_args_list if call.args[0] == slot]
            self.assertEqual([call[1] for call in slot_calls], [1, 2, 3, 4])
            self.assertIsNone(slot_calls[0][2])

    def test_no_survivors_stop_for_human_intervention_with_exact_budget(self):
        generate = Mock(return_value="unsafe")
        check = Mock(return_value=SafetyResult(False, "unsafe"))
        judge = Mock()

        with self.assertRaisesRegex(HumanInterventionRequired, "video.*human intervention"):
            select_candidate(
                stage="video", generate=generate, check_safety=check,
                choose_best=judge, pass_feedback=False,
            )

        self.assertEqual(generate.call_count, 16)
        self.assertEqual(check.call_count, 16)
        self.assertCountEqual(
            [call.args[:2] for call in generate.call_args_list],
            [(slot, attempt) for slot in range(1, 5) for attempt in range(1, 5)],
        )
        judge.assert_not_called()

    def test_missing_unsafe_reason_gets_useful_feedback(self):
        for reason in ("", "  \n "):
            with self.subTest(reason=reason):
                generate = Mock(side_effect=lambda slot, attempt, feedback: attempt)
                select_candidate(
                    stage="outline", generate=generate,
                    check_safety=lambda attempt: SafetyResult(attempt == 2, reason),
                    choose_best=lambda candidates: 0, pass_feedback=True,
                )
                retries = [call.args for call in generate.call_args_list if call.args[1] == 2]
                self.assertEqual(len(retries), 4)
                for slot, attempt, feedback in retries:
                    self.assertIsInstance(feedback, str)
                    self.assertIn("safety", feedback)
                    self.assertIn("safer", feedback)

    def test_quality_judge_index_must_be_a_plain_integer_in_range(self):
        class IntegerSubclass(int):
            pass

        for index in (-1, 4, True, False, 1.0, "1", None, IntegerSubclass(1)):
            with self.subTest(index=index, type=type(index)):
                with self.assertRaisesRegex(ValueError, "outline.*integer index"):
                    select_candidate(
                        stage="outline", generate=lambda slot, attempt, feedback: slot,
                        check_safety=lambda candidate: SafetyResult(True),
                        choose_best=lambda candidates: index, pass_feedback=True,
                    )

    def test_quality_index_is_checked_against_survivors_not_original_slots(self):
        with self.assertRaisesRegex(ValueError, "0 to 0"):
            select_candidate(
                stage="video", generate=lambda slot, attempt, feedback: slot,
                check_safety=lambda candidate: SafetyResult(candidate == 3),
                choose_best=lambda candidates: 2, pass_feedback=False,
            )

    def test_generation_errors_propagate_without_safety_retry(self):
        error = RuntimeError("provider unavailable")
        generate = Mock(side_effect=error)
        check = Mock()
        judge = Mock()

        with self.assertRaises(RuntimeError) as caught:
            select_candidate(
                stage="outline", generate=generate, check_safety=check,
                choose_best=judge, pass_feedback=True,
            )

        self.assertIs(caught.exception, error)
        self.assert_first_attempts_only(generate)
        check.assert_not_called()
        judge.assert_not_called()

    def test_safety_errors_propagate_without_regeneration(self):
        error = RuntimeError("safety checker unavailable")
        generate = Mock(return_value="candidate")
        check = Mock(side_effect=error)
        judge = Mock()

        with self.assertRaises(RuntimeError) as caught:
            select_candidate(
                stage="outline", generate=generate, check_safety=check,
                choose_best=judge, pass_feedback=True,
            )

        self.assertIs(caught.exception, error)
        self.assert_first_attempts_only(generate)
        self.assertEqual(check.call_args_list, [(("candidate",),)] * generate.call_count)
        judge.assert_not_called()

    def test_malformed_safety_results_stop_without_retry_or_quality_judging(self):
        malformed_results = (
            None, True, "safe", {"safe": True, "reason": ""},
            SafetyResult("false"), SafetyResult(0), SafetyResult(1),
            SafetyResult(True, None), SafetyResult(False, []),
        )
        for safety in malformed_results:
            with self.subTest(safety=safety):
                generate = Mock(return_value="candidate")
                check = Mock(return_value=safety)
                judge = Mock()

                with self.assertRaisesRegex(ValueError, "storyboard.*SafetyResult"):
                    select_candidate(
                        stage="storyboard", generate=generate, check_safety=check,
                        choose_best=judge, pass_feedback=True,
                    )

                self.assert_first_attempts_only(generate)
                self.assertEqual(check.call_args_list, [(("candidate",),)] * generate.call_count)
                judge.assert_not_called()

    def assert_first_attempts_only(self, generate):
        calls = [call.args for call in generate.call_args_list]
        self.assertGreaterEqual(len(calls), 1)
        self.assertLessEqual(len(calls), 4)
        self.assertEqual(len({slot for slot, attempt, feedback in calls}), len(calls))
        self.assertTrue(all(attempt == 1 and feedback is None
                            for slot, attempt, feedback in calls))

    def test_quality_errors_propagate_without_regeneration(self):
        error = RuntimeError("quality judge unavailable")
        generate = Mock(return_value="candidate")
        judge = Mock(side_effect=error)

        with self.assertRaises(RuntimeError) as caught:
            select_candidate(
                stage="outline", generate=generate,
                check_safety=lambda candidate: SafetyResult(True),
                choose_best=judge, pass_feedback=True,
            )

        self.assertIs(caught.exception, error)
        self.assertEqual(generate.call_count, 4)
        judge.assert_called_once_with(("candidate",) * 4)


if __name__ == "__main__":
    unittest.main()
