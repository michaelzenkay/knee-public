"""Load classifier assets and predict twelve study-level findings."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .model import Config, NativeKnee


def load_model(checkpoint, dino_config, device="cpu", chunk=1):
    """Load a trained classifier and its DINO configuration from local assets."""
    from transformers import Dinov2Config, Dinov2Model

    config = Dinov2Config.from_json_file(str(Path(dino_config)))
    config._attn_implementation = "sdpa"
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = replace(Config(**blob["identity"]["architecture"]), chunk=chunk)
    model = NativeKnee(Dinov2Model(config), architecture)
    model.load_state_dict(blob["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def predict(model, images, device="cpu", precision="fp32"):
    if precision not in ("fp32", "bf16") or (
        precision == "bf16" and torch.device(device).type != "cuda"
    ):
        raise ValueError("Choose fp32, or bf16 on CUDA")
    model = model.to(device).eval()
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )
    with context:
        logits = model(images)
    probabilities = logits.float().sigmoid().cpu().numpy()
    if probabilities.shape != (12,) or not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Nonfinite or malformed prediction; no fallback probabilities"
        )
    return probabilities
