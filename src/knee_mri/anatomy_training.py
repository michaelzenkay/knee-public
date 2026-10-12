"""Balanced segmentation and ACL/MCL training from reviewed masks and points."""

import numpy as np
import torch
from torch.nn import functional as F

from .anatomy import attachment_slice_scores, normalize_volume, slice_triplet
from .losses import (
    attachment_localization_loss,
    attachment_slice_classification_loss,
    attachment_slice_geometry_loss,
    partial_segmentation_loss,
)


def attachment_batch(
    volume, endpoints, *, size=192, sigma_xy=0.025, sigma_z=1.0, flip_x=False
):
    """One ordered case window; endpoints are [femoral/tibial, (x, y, slice)].

    x/y are normalized to [0, 1]; slice is the index within this volume.
    Crop the annotated training window first and adjust slice indices to it.
    """
    points = np.asarray(endpoints, dtype=np.float32).copy()
    if points.shape != (2, 3) or not np.isfinite(points).all():
        raise ValueError("Expected two finite x/y/slice attachment points")
    if np.any((points[:, :2] < 0) | (points[:, :2] > 1)) or np.any(
        (points[:, 2] < 0) | (points[:, 2] > len(volume) - 1)
    ):
        raise ValueError("Attachments must be inside the supplied case window")
    normalized = normalize_volume(volume)
    small = F.interpolate(
        torch.from_numpy(normalized[:, None]),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    small = (small.clamp(0, 1) * 255).to(torch.uint8).numpy()
    images = torch.from_numpy(
        np.stack([slice_triplet(small, z) for z in range(len(small))]) / 255
    )
    if flip_x:
        images = images.flip(-1)
        points[:, 0] = 1 - points[:, 0]
    axis = np.linspace(0, 1, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    maps = np.stack(
        [
            np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma_xy**2))
            for x, y, _ in points
        ]
    )
    presence = np.exp(
        -0.5 * ((np.arange(len(small))[:, None] - points[:, 2]) / sigma_z) ** 2
    ).astype(np.float32)
    return dict(
        image=images.float(),
        target_heatmaps=torch.from_numpy(
            np.broadcast_to(maps, (len(small), *maps.shape)).copy()
        ),
        target_presence=torch.from_numpy(presence),
    )


def anatomy_loss(model, batch, task, device="cpu", class_weights=None):
    output = model(batch["image"].to(device))
    if task == "segmentation":
        return partial_segmentation_loss(
            output["segmentation_logits"],
            batch["target"].to(device),
            batch["supervised_classes"].to(device),
            class_weights=class_weights,
        )[0]
    if task not in ("acl", "mcl"):
        raise ValueError("Expected segmentation, acl or mcl supervision")
    maps = output[task + "_attachment_logits"]
    presence = output[task + "_attachment_presence_logits"]
    target = batch["target_presence"].to(device)
    loss, _ = attachment_localization_loss(
        maps, presence, batch["target_heatmaps"].to(device), target
    )
    scores = attachment_slice_scores(presence, maps, spatial_evidence_weight=0.5)
    slice_loss = attachment_slice_classification_loss(
        scores, target, label_smoothing=0.05
    )
    geometry_loss, _ = attachment_slice_geometry_loss(scores, target)
    return loss + slice_loss + 0.5 * geometry_loss


def train_anatomy_epoch(
    model,
    loaders,
    optimizer,
    *,
    device="cpu",
    steps_per_task=32,
    class_weights=None,
    scaler=None,
):
    """Equal optimizer steps for segmentation, ACL and MCL; no task mixing within a batch."""
    tasks = ("segmentation", "acl", "mcl")
    if steps_per_task < 1 or any(
        task not in loaders or len(loaders[task]) == 0 for task in tasks
    ):
        raise ValueError(
            "Each task needs a nonempty re-iterable loader and positive steps"
        )
    device = torch.device(device)
    model.to(device).train()
    iterators = {task: iter(loaders[task]) for task in tasks}
    order = [task for task in tasks for _ in range(steps_per_task)]
    np.random.shuffle(order)
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    totals = dict.fromkeys(tasks, 0.0)
    for task in order:
        try:
            batch = next(iterators[task])
        except StopIteration:
            iterators[task] = iter(loaders[task])
            batch = next(iterators[task])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = anatomy_loss(model, batch, task, device, class_weights)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite anatomy loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        totals[task] += float(loss.detach())
    return {task: total / steps_per_task for task, total in totals.items()}
