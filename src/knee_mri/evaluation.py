"""Finding metrics and physical attachment error."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .anatomy import KNEE_STRUCTURE_CLASS_NAMES
from .data import FINDINGS, load_study
from .losses import partial_segmentation_loss


def classification_metrics(targets, probabilities):
    y, p = np.asarray(targets), np.asarray(probabilities)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != len(FINDINGS) or not len(y):
        raise ValueError("Expected nonempty [studies, 12] targets and probabilities")
    if not np.isfinite([y, p]).all() or np.any((y < 0) | (y > 1) | (p < 0) | (p > 1)):
        raise ValueError("Targets and probabilities must be finite and in [0, 1]")
    hard = y >= 0.5
    auc = {
        name: float(roc_auc_score(hard[:, i], p[:, i]))
        if len(np.unique(hard[:, i])) == 2
        else None
        for i, name in enumerate(FINDINGS)
    }
    if any(value is None for value in auc.values()):
        raise ValueError(
            "Development cohort needs both classes for all twelve findings"
        )
    safe = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(
        macro_auc=sum(auc.values()) / 12,
        per_endpoint_auc=auc,
        soft_bce=float(-(y * np.log(safe) + (1 - y) * np.log1p(-safe)).mean()),
    )


@torch.inference_mode()
def evaluate_classifier(model, records, cache, device="cpu"):
    model = model.to(device).eval()
    device = torch.device(device)
    targets, probabilities = [], []
    for record in records:
        images, y = load_study(cache, record)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(images)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite prediction")
        probabilities.append(logits.float().sigmoid().cpu().numpy())
        targets.append(y.numpy())
    return classification_metrics(targets, probabilities), np.asarray(probabilities)


def endpoint_error_mm(predicted_xyz_mm, reference_xyz_mm):
    """Per-endpoint Euclidean error after mapping both points to the same frame."""
    predicted, reference = np.asarray(predicted_xyz_mm), np.asarray(reference_xyz_mm)
    if predicted.shape != reference.shape or predicted.shape[-1:] != (3,):
        raise ValueError("Expected matching physical XYZ coordinates")
    if not np.isfinite([predicted, reference]).all():
        raise ValueError("Endpoint coordinates must be finite")
    return np.linalg.norm(predicted - reference, axis=-1)


@torch.no_grad()
def evaluate_segmentation(model, loader, device="cpu", class_weights=None):
    model.eval()
    loss_sum = weight_sum = 0.0
    intersections = torch.zeros(len(KNEE_STRUCTURE_CLASS_NAMES), device=device)
    denominators = torch.zeros_like(intersections)
    counts = torch.zeros_like(intersections)
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        supervised = batch["supervised_classes"].to(device)
        outputs = model(images)
        loss, _ = partial_segmentation_loss(
            outputs["logits"],
            targets,
            supervised,
            class_weights=class_weights,
        )
        weight = float(supervised.sum())
        loss_sum += float(loss) * weight
        weight_sum += weight
        predicted = outputs["probabilities"] >= 0.5
        truth = targets >= 0.5
        intersections += ((predicted & truth).sum(dim=(-1, -2)) * supervised).sum(dim=0)
        denominators += (
            (predicted.sum(dim=(-1, -2)) + truth.sum(dim=(-1, -2))) * supervised
        ).sum(dim=0)
        counts += supervised.sum(dim=0)
    dice = 2 * intersections / denominators.clamp(min=1)
    return {
        "loss": loss_sum / max(weight_sum, 1.0),
        "dice": {
            name: float(dice[index].cpu())
            for index, name in enumerate(KNEE_STRUCTURE_CLASS_NAMES)
            if counts[index] > 0
        },
        "supervised_class_slices": {
            name: int(counts[index].cpu())
            for index, name in enumerate(KNEE_STRUCTURE_CLASS_NAMES)
        },
    }
