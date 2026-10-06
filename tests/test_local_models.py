"""Model API contracts and fail-closed speech/visual moderation behavior."""

from contextlib import contextmanager, nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from vocagent.local_models import (
    ALIGNER_MODEL, ASR_MODEL, SHIELD_MODEL, SHIELD_POLICIES,
    LocalModelError, Qwen3SpeechToText, ShieldGemma2VideoSafety,
    aligned_segments, shield_verdict,
)
from vocagent.models import SafetyResult, TranscriptSegment
from vocagent.video_frames import VideoFrame


def word(text, start, end):
    return SimpleNamespace(text=text, start_time=start, end_time=end)


class Runtime:
    device = "musa:0"
    dtype = "bfloat16"

    def __init__(self, resource=None):
        self.resource = resource
        self.keys = []
        self.asserted = []
        self.active = False

    def assert_on_accelerator(self, model):
        self.asserted.append(model)

    @contextmanager
    def use(self, key, loader):
        self.keys.append(key)
        self.active = True
        try:
            yield self.resource if self.resource is not None else loader()
        finally:
            self.active = False


class SpeechTests(unittest.TestCase):
    def test_alignment_preserves_original_punctuation_and_mixed_language(self):
        result = aligned_segments(
            "你好，café! Friend's apple.",
            [word("你", 0, .5), word("好", .5, 1), word("café", 1, 2),
             word("Friend's", 3, 4), word("apple", 4, 5)], duration=5,
        )
        self.assertEqual(result, [
            TranscriptSegment(0, 2, "你好，café!"),
            TranscriptSegment(3, 5, "Friend's apple."),
        ])

    def test_alignment_preserves_text_of_zero_duration_tail_tokens(self):
        result = aligned_segments(
            "One. Two!", [word("One", 0, 1), word("Two", 1, 1)],
            duration=2, max_cue_characters=2,
        )
        self.assertEqual(result, [TranscriptSegment(0, 1, "One. Two!")])

    def test_bad_or_partial_alignment_never_loses_unreviewed_speech(self):
        cases = [
            ("hello world", [word("hello", 0, 1)]),
            ("hello world", [word("world", 0, 1), word("hello", 1, 2)]),
            ("hello", []),
            ("hello", [word("hello", float("nan"), 1)]),
            ("hello", [word("hello", -1, 1)]),
            ("hello", [word("hello", True, 1)]),
            ("hello", [word("hello", 1, 0)]),
            ("hello", [word("hello", 0, 8)]),
            ("hello", [word("hello", 0, 0)]),
            ("hello world", [word("hello", 1, 2), word("world", 0, 1)]),
        ]
        for text, items in cases:
            with self.subTest(text=text, items=items), self.assertRaises(LocalModelError):
                aligned_segments(text, items, duration=3)

    def test_official_asr_and_aligner_both_load_explicitly_on_musa(self):
        runtime = Runtime()
        speech = SimpleNamespace(model=MagicMock(), forced_aligner=SimpleNamespace(model=MagicMock()))
        factory = MagicMock(return_value=speech)
        fake_module = SimpleNamespace(Qwen3ASRModel=SimpleNamespace(from_pretrained=factory))
        with patch.dict("sys.modules", {"qwen_asr": fake_module}), patch(
            "vocagent.gpu_ops.patch_musa_convolutions",
        ) as convolution_patch:
            result = Qwen3SpeechToText(runtime)._load()
        self.assertIs(result, speech)
        self.assertEqual(factory.call_args.args, (ASR_MODEL,))
        options = factory.call_args.kwargs
        self.assertIsNone(options["device_map"])
        self.assertEqual(options["attn_implementation"], "eager")
        self.assertEqual(options["forced_aligner"], ALIGNER_MODEL)
        self.assertEqual(options["forced_aligner_kwargs"], {
            "dtype": "bfloat16", "device_map": None, "attn_implementation": "eager",
        })
        speech.model.to.assert_called_once_with("musa:0")
        speech.forced_aligner.model.to.assert_called_once_with("musa:0")
        self.assertEqual(speech.device, speech.model.device)
        self.assertEqual(speech.forced_aligner.device, speech.forced_aligner.model.device)
        self.assertEqual(options["max_inference_batch_size"], 1)
        self.assertEqual(runtime.asserted, [speech.model, speech.forced_aligner.model])
        self.assertEqual(convolution_patch.call_count, 2)

    def test_transcription_uses_official_timestamps_and_original_time_base(self):
        speech = MagicMock()
        speech.transcribe.return_value = [SimpleNamespace(
            text="Hello, world!", time_stamps=SimpleNamespace(items=[
                word("Hello", 180.1, 180.5), word("world", 180.5, 181),
            ]),
        )]
        runtime = Runtime(speech)
        info = MagicMock(return_value=SimpleNamespace(duration=182))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.write_bytes(b"audio")
            with patch.dict("sys.modules", {"soundfile": SimpleNamespace(info=info)}):
                result = Qwen3SpeechToText(runtime).transcribe(path)
        self.assertEqual(result, [TranscriptSegment(180.1, 181, "Hello, world!")])
        speech.transcribe.assert_called_once_with(
            audio=str(path), language=None, return_time_stamps=True,
        )
        self.assertEqual(runtime.keys, [("qwen3-speech", ASR_MODEL, ALIGNER_MODEL)])
        self.assertFalse(runtime.active)

    def test_missing_alignment_stops_transcription_and_releases_runtime(self):
        speech = MagicMock()
        speech.transcribe.return_value = [SimpleNamespace(text="unreviewed", time_stamps=None)]
        runtime = Runtime(speech)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.write_bytes(b"audio")
            with patch.dict("sys.modules", {"soundfile": SimpleNamespace(
                info=lambda _: SimpleNamespace(duration=2),
            )}), self.assertRaises(LocalModelError):
                Qwen3SpeechToText(runtime).transcribe(path)
        self.assertFalse(runtime.active)


class ShieldTests(unittest.TestCase):
    def test_yes_probability_is_the_violation_column(self):
        self.assertEqual(shield_verdict([[.1, .9]] * 3, [0], threshold=.5), SafetyResult(True))
        result = shield_verdict(
            [[.1, .9], [.8, .2], [.1, .9], [.6, .4], [.1, .9], [.1, .9]],
            [0, 3.5], threshold=.5,
        )
        self.assertFalse(result.safe)
        self.assertIn("sexual at 0.00s", result.reason)
        self.assertIn("dangerous at 3.50s", result.reason)

    def test_missing_nan_or_malformed_scores_never_pass(self):
        for rows in (
            [], [[.1, .9]], [[float("nan"), .9]] * 3, [[.5, .9]] * 3,
            [[-1, 2]] * 3, [[.1]] * 3, [[True, False]] * 3,
        ):
            with self.subTest(rows=rows), self.assertRaises(LocalModelError):
                shield_verdict(rows, [0], threshold=.5)

    def test_shield_loads_bfloat16_on_musa_without_auto_device_map(self):
        runtime = Runtime()
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        model.config.text_config.tie_word_embeddings = True
        embedding = SimpleNamespace(weight=object())
        model.model.get_output_embeddings.return_value = embedding
        model.model.get_input_embeddings.return_value = embedding
        factory = MagicMock(return_value=(model, {"missing_keys": ["model.lm_head.weight"]}))
        processor_factory = MagicMock()
        fake_module = SimpleNamespace(
            ShieldGemma2ForImageClassification=SimpleNamespace(from_pretrained=factory),
            AutoProcessor=SimpleNamespace(from_pretrained=processor_factory),
        )
        with patch.dict("sys.modules", {"transformers": fake_module}), patch(
            "vocagent.gpu_ops.patch_musa_convolutions",
        ):
            ShieldGemma2VideoSafety(runtime)._load()
        factory.assert_called_once_with(
            SHIELD_MODEL, dtype="bfloat16", device_map=None, attn_implementation="eager",
            output_loading_info=True,
        )
        model.model.tie_weights.assert_called_once()
        model.to.assert_called_once_with("musa:0")
        self.assertEqual(runtime.asserted, [model])

    def test_shield_rejects_random_untied_output_head(self):
        model = MagicMock()
        model.config.text_config.tie_word_embeddings = True
        factory = MagicMock(return_value=(model, {"missing_keys": ["model.lm_head.weight"]}))
        fake_module = SimpleNamespace(
            ShieldGemma2ForImageClassification=SimpleNamespace(from_pretrained=factory),
            AutoProcessor=MagicMock(),
        )
        with patch.dict("sys.modules", {"transformers": fake_module}), self.assertRaisesRegex(
            LocalModelError, "not tied",
        ):
            ShieldGemma2VideoSafety(Runtime())._load()
        model.to.assert_not_called()

    def test_bad_configuration_is_rejected_before_loading(self):
        for options in (
            {"sample_fps": 0}, {"sample_fps": float("nan")}, {"sample_fps": True},
            {"max_frames": 0}, {"max_frames": True}, {"threshold": 0}, {"threshold": 1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ShieldGemma2VideoSafety(Runtime(), **options)

    def test_frame_review_covers_later_unsafe_frame_and_passes_all_policies(self):
        model = MagicMock()
        probability_outputs = []
        for rows in ([[.1, .9]] * 3, [[.1, .9], [.1, .9], [.9, .1]]):
            tensor = MagicMock()
            tensor.detach.return_value.float.return_value.cpu.return_value.tolist.return_value = rows
            probability_outputs.append(SimpleNamespace(probabilities=tensor))
        model.side_effect = probability_outputs
        processor = MagicMock()
        processor.return_value.to.return_value = {"input_ids": "on_musa"}
        runtime = Runtime((model, processor))
        image = MagicMock()
        fake_pil = SimpleNamespace(Image=SimpleNamespace(open=MagicMock(return_value=image)))
        fake_torch = SimpleNamespace(inference_mode=nullcontext)
        frames = [VideoFrame(0, b"first"), VideoFrame(4, b"last")]
        with patch("vocagent.video_frames.sample_video_frames", return_value=frames) as sampler, patch.dict(
            "sys.modules", {"PIL": fake_pil, "torch": fake_torch},
        ):
            result = ShieldGemma2VideoSafety(runtime).check(Path("video.mp4"))
        self.assertFalse(result.safe)
        self.assertIn("violence at 4.00s", result.reason)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(processor.call_args.kwargs["policies"], list(SHIELD_POLICIES))
        processor.return_value.to.assert_called_with(device="musa:0", dtype="bfloat16")
        self.assertEqual(sampler.call_args.kwargs["sample_fps"], 2)
        self.assertFalse(runtime.active)


if __name__ == "__main__":
    unittest.main()
