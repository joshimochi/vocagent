"""Speech coverage must preserve word boundaries and support ordinary ASR text."""

import unittest

from vocagent.vocabulary import missing_vocabulary


class VocabularyTests(unittest.TestCase):
    def test_case_width_punctuation_and_phrase_boundaries(self):
        self.assertEqual(missing_vocabulary(
            ["curious", "DISCOVER", "ice cream", "Café"],
            "ＣＵＲＩＯＵＳ? Discover ice-cream at the cafe\u0301!",
        ), ())
        self.assertEqual(missing_vocabulary(
            ["pear", "discover", "ice cream"], "A pearl appears. Discovery: cream ice.",
        ), ("pear", "discover", "ice cream"))

    def test_cjk_phrases_are_found_inside_unspaced_sentences(self):
        self.assertEqual(missing_vocabulary(
            ["好奇", "发现", "リンゴ", "사과"],
            "我很好奇并发现了秘密。赤いリンゴです。사과를먹어요.",
        ), ())
        self.assertEqual(missing_vocabulary(["发现"], "现在发明了新工具。"), ("发现",))

    def test_phrase_must_be_contiguous_and_have_spoken_characters(self):
        self.assertEqual(missing_vocabulary(
            ["green pear", "green apple", "!!!"], "A GREEN,\npear and a green small apple.",
        ), ("green apple", "!!!"))

    def test_known_unrelated_h3_transcript_fails(self):
        self.assertEqual(missing_vocabulary(
            ["curious", "discover"], "History of some writings.",
        ), ("curious", "discover"))


if __name__ == "__main__":
    unittest.main()
