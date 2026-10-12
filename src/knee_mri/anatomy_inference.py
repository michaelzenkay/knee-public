from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .anatomy import (
    FusedKneeAnatomyResNet34Unet2p5D,
    attachment_slice_scores,
    normalize_volume,
    slice_triplet,
)


def load_anatomy(checkpoint, device="cpu"):
    """Load the segmentation/ACL/MCL checkpoint strictly; no unused laterality head."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = FusedKneeAnatomyResNet34Unet2p5D()
    model.load_state_dict(blob["model"], strict=True)
    return model.to(device).eval()


def spatial_centroid(logits: torch.Tensor) -> torch.Tensor:
    """Activation-weighted x/y coordinates for [N,C,H,W] logits."""
    height, width = logits.shape[-2:]
    weights = torch.softmax(logits.flatten(2), dim=2)
    x_axis = torch.linspace(0, 1, width, device=logits.device)
    y_axis = torch.linspace(0, 1, height, device=logits.device)
    yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
    x = (weights * xx.flatten()).sum(2)
    y = (weights * yy.flatten()).sum(2)
    return torch.stack([x, y], dim=2)


@torch.inference_mode()
def predict_anatomy(
    volume,
    model,
    plane,
    *,
    device="cpu",
    laterality=None,
    orientation=None,
    spatial_evidence_weight=0.5,
):
    """Companion model masks and normalized endpoint coordinates; no rendering.

    Mirrors the portfolio inference policy: 256px sagittal segmentation,
    192px quantized endpoint inputs, full-series selection. MCL requires explicit
    laterality and usable orientation. Input is a physically ordered (Z,Y,X) volume.
    """
    if volume.ndim != 3 or not len(volume):
        raise ValueError("Expected a nonempty 3D volume")
    model = model.to(device).eval()
    norm = normalize_volume(volume)
    masks = None
    n = len(volume)
    if plane.lower() == "sagittal":
        results = []
        for start in range(0, n, 4):
            x = torch.from_numpy(
                np.stack(
                    [
                        normalize_volume(slice_triplet(volume, z))
                        for z in range(start, min(n, start + 4))
                    ]
                )
            )
            x = (
                F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
                .float()
                .to(device)
            )
            results.append(model(x)["segmentation_probabilities"].cpu().numpy() > 0.5)
        masks = np.concatenate(results)
    task = (
        "acl"
        if plane.lower() == "sagittal"
        else "mcl"
        if plane.lower() == "coronal"
        else None
    )
    flip = False
    if task == "mcl":
        if (
            laterality not in ("L", "R")
            or orientation is None
            or abs(orientation[0]) < 0.5
            or np.argmax(np.abs(orientation[:3])) != 0
        ):
            task = None
        else:
            flip = ("L" if orientation[0] >= 0 else "R") != (
                "R" if laterality == "L" else "L"
            )
    endpoints = []
    if task:
        small = F.interpolate(
            torch.from_numpy(norm[:, None]),
            size=(192, 192),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        small = (small.clamp(0, 1) * 255).to(torch.uint8).numpy()
        logits, presence = [], []
        for start in range(0, n, 4):
            x = torch.from_numpy(
                np.stack(
                    [slice_triplet(small, z) for z in range(start, min(n, start + 4))]
                )
                / 255
            )
            if flip:
                x = x.flip(-1)
            result = model(x.float().to(device))
            logits.append(result[task + "_attachment_logits"].cpu())
            presence.append(result[task + "_attachment_presence_logits"].cpu())
        logits, presence = torch.cat(logits), torch.cat(presence)
        scores = attachment_slice_scores(
            presence, logits, spatial_evidence_weight=spatial_evidence_weight
        )
        xy = spatial_centroid(logits).numpy()
        for k, name in enumerate(("femoral", "tibial")):
            z = int(scores[:, k].argmax())
            x, y = map(float, xy[z, k])
            x = 1 - x if flip else x
            endpoints.append(
                dict(structure=task.upper(), attachment=name, slice=z, x=x, y=y)
            )
    return dict(masks=masks, endpoints=endpoints, mcl_flip=flip)
