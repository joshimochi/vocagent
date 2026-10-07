"""Complete self-hosted provider factory for the hs GPU allocation."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping

from .gpu_runtime import GPURuntime
from .local_chat import LocalChatClient
from .local_models import Qwen3SpeechToText, ShieldGemma2VideoSafety
from .providers import Providers
from .qwen import QwenAudioSafety, QwenGenerator, QwenGuard, QwenQuality, QwenVideoQuality


@dataclass
class LocalProviders(Providers):
    runtime: GPURuntime = field(repr=False)

    def close(self):
        self.runtime.close()


def create_providers(environ: Mapping[str, str] | None = None) -> Providers:
    env = os.environ if environ is None else environ
    root = Path(env.get("VOCAGENT_DEPLOY_ROOT", "/data/group6/vocagent"))
    runtime = GPURuntime(env.get("VOCAGENT_BACKEND", "musa"))

    def checkpoint(variable, name):
        path = Path(env.get(variable, str(root / "models" / name)))
        if not (path / "config.json").is_file():
            raise ValueError(f"Missing local checkpoint {path}; download it before running the agent.")
        return str(path)

    vl = checkpoint("QWEN_VL_MODEL", "Qwen3-VL-32B-Instruct")
    guard = checkpoint("QWEN_GUARD_MODEL_PATH", "Qwen3Guard-Gen-8B")
    asr = checkpoint("QWEN_ASR_MODEL", "Qwen3-ASR-1.7B")
    aligner = checkpoint("QWEN_ALIGNER_MODEL", "Qwen3-ForcedAligner-0.6B")
    shield = checkpoint("SHIELDGEMMA_MODEL", "shieldgemma-2-4b-it")
    # Delayed import keeps model SDKs optional for configuration and tests.
    from .minimax import H3Settings
    from .isolated_h3 import IsolatedMiniMaxH3VideoGenerator

    settings = H3Settings(
        steps=int(env.get("MINIMAX_H3_STEPS", "50")),
    )
    return LocalProviders(
        runtime=runtime,
        llm=QwenGenerator(LocalChatClient(runtime, vl, vision=True), storyboard_seconds=settings.seconds),
        text_safety=QwenGuard(LocalChatClient(runtime, guard)),
        text_quality=QwenQuality(LocalChatClient(runtime, vl, vision=True)),
        video=IsolatedMiniMaxH3VideoGenerator(
            runtime, env.get("MINIMAX_H3_MODEL", "/data/group6/minimax-h3/models/MiniMax-H3"),
            python_executable=env.get("MINIMAX_H3_PYTHON", str(root / "run-a406/h3-venv/bin/python")),
            settings=settings,
            timeout=float(env.get("MINIMAX_H3_TIMEOUT_SECONDS", "14400")),
        ),
        video_safety=ShieldGemma2VideoSafety(runtime, shield),
        # 32 full-resolution frames passed the deployed 16K-context GPU probe.
        # Validate this budget before generation so large slot counts fail early.
        video_quality=QwenVideoQuality(
            LocalChatClient(runtime, vl, vision=True), max_total_frames=32,
        ),
        speech_to_text=Qwen3SpeechToText(runtime, asr, aligner),
        audio_safety=QwenAudioSafety(LocalChatClient(runtime, guard)),
    )
