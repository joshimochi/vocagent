"""The default deployment connects every stage to an actual local adapter."""

from pathlib import Path
import tempfile
import unittest

from vocagent.local import create_providers
from vocagent.local_chat import LocalChatClient
from vocagent.local_models import Qwen3SpeechToText, ShieldGemma2VideoSafety
from vocagent.minimax import MiniMaxH3VideoGenerator
from vocagent.qwen import QwenAudioSafety, QwenVideoQuality


class LocalFactoryTests(unittest.TestCase):
    def test_complete_factory_has_shared_residency_and_independent_agents(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('Qwen3-VL-32B-Instruct', 'Qwen3Guard-Gen-8B',
                         'Qwen3-ASR-1.7B', 'Qwen3-ForcedAligner-0.6B', 'shieldgemma-2-4b-it'):
                path = Path(directory) / 'models' / name
                path.mkdir(parents=True)
                (path / 'config.json').write_text('{}')
            providers = create_providers({
                'VOCAGENT_DEPLOY_ROOT': directory,
                'MINIMAX_H3_TIMEOUT_SECONDS': '14400',
                'MINIMAX_H3_WIDTH': 'ignored',
                'MINIMAX_H3_HEIGHT': 'ignored',
                'MINIMAX_H3_SECONDS': 'ignored',
            })
            self.assertIsInstance(providers.llm.client, LocalChatClient)
            self.assertIsInstance(providers.video, MiniMaxH3VideoGenerator)
            self.assertIsInstance(providers.video_quality, QwenVideoQuality)
            self.assertIsInstance(providers.audio_safety, QwenAudioSafety)
            self.assertIsInstance(providers.speech_to_text, Qwen3SpeechToText)
            self.assertIsInstance(providers.video_safety, ShieldGemma2VideoSafety)
            self.assertIsNot(providers.llm.client, providers.text_quality.client)
            for provider in (providers.llm, providers.text_safety, providers.text_quality,
                             providers.audio_safety, providers.video_quality):
                self.assertIs(provider.client.runtime, providers.runtime)
            self.assertIs(providers.video.runtime, providers.runtime)
            self.assertEqual(providers.video.timeout, 14400)
            self.assertEqual(providers.video.settings.width, 768)
            self.assertEqual(providers.video.settings.height, 1344)
            self.assertEqual(providers.video.settings.seconds, 15)
            self.assertEqual(providers.llm.storyboard_seconds, providers.video.settings.seconds)
            providers.video_quality.validate_candidate_count(2)
            with self.assertRaisesRegex(ValueError, 'at most 2 candidate slots'):
                providers.video_quality.validate_candidate_count(3)
            self.assertIsNone(providers.runtime._torch)  # import/configuration never loads weights

    def test_missing_weights_fail_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Missing local checkpoint'):
                create_providers({'VOCAGENT_DEPLOY_ROOT': directory})
