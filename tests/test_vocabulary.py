"""Validate single English target inputs."""

import unittest

from vocagent.vocabulary import normalize_target_word


class TargetWordTests(unittest.TestCase):
    def test_single_word_and_legacy_singleton_sequence_preserve_spelling(self):
        for vocabulary, expected in (
            (" curious ", "curious"), ([" Apple "], "Apple"),
            (("sunrise",), "sunrise"), ("can't", "can't"),
            ("can’t", "can’t"), ("well-known", "well-known"),
        ):
            with self.subTest(vocabulary=vocabulary):
                self.assertEqual(normalize_target_word(vocabulary), expected)

    def test_rejects_multiple_targets_phrases_and_non_english_word_shapes(self):
        for vocabulary in (
            None, [], ["apple", "pear"], ["apple", "apple"], [None], [1],
            "", "  ", "green pear", ["green pear"], "apple\npear", "你好",
            "123", "apple2", "apple!", "-apple", "apple-", "well--known",
            "'apple'", b"a", {"apple": "pear"},
        ):
            with self.subTest(vocabulary=vocabulary), \
                    self.assertRaisesRegex(ValueError, "exactly one English word"):
                normalize_target_word(vocabulary)


if __name__ == "__main__":
    unittest.main()
