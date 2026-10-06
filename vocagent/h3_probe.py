"""Verify the real H3 video/audio decoders on one GPU, without CPU inference."""

import argparse
import gc
import json
from pathlib import Path
import time

from .gpu_runtime import GPURuntime
from .minimax import _patch_audio_convolutions, _patch_pointwise_conv3d


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("musa", "cuda"), default="musa")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--component", choices=("video", "audio", "both"), default="both")
    args = parser.parse_args(argv)
    runtime = GPURuntime(args.backend)
    torch = runtime.torch
    torch.set_num_threads(8)
    accelerator = getattr(torch, args.backend)
    accelerator.set_device(args.gpu)
    device = f"{args.backend}:{args.gpu}"
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames

    results = []
    for name in ("video", "audio"):
        if args.component not in (name, "both"):
            continue
        started = time.monotonic()
        model_class = AutoencoderKLMiniMaxH3 if name == "video" else AutoencoderKLMiniMaxH3Audio
        folder = "vae" if name == "video" else "audio_vae"
        model = model_class.from_pretrained(
            str(args.model / folder), torch_dtype=torch.float32, local_files_only=True,
        ).eval().requires_grad_(False).to(device)
        runtime.assert_on_accelerator(model)
        if args.backend == "musa":
            (_patch_pointwise_conv3d if name == "video" else _patch_audio_convolutions)(model)
        shape = (1, 24, 37, 4, 4) if name == "video" else (
            2, len(model.config.latents_mean), audio_latent_num_frames(124),
        )
        with torch.inference_mode(), torch.autocast(
            device_type=args.backend, dtype=torch.float16, enabled=name == "video",
        ):
            decoded = model.decode(torch.zeros(shape, device=device), return_dict=False)[0]
        if decoded.device.type != args.backend or not torch.isfinite(decoded).all().item():
            raise RuntimeError(f"{name} decoder did not produce finite GPU output.")
        result = {
            "component": name, "device": str(decoded.device), "output_shape": list(decoded.shape),
            "weight_gib": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3,
            "seconds": round(time.monotonic() - started, 3), "finite": True,
        }
        print(json.dumps(result), flush=True)
        results.append(result)
        del decoded, model
        gc.collect()
        accelerator.empty_cache()
    return results


if __name__ == "__main__":
    main()
