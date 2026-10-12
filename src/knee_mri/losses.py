"""Partial segmentation and point-supervised attachment objectives."""

import torch
from torch.nn import functional as F


def partial_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    supervised_classes: torch.Tensor,
    *,
    dice_weight: float = 1.0,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute BCE+Dice only for explicitly annotated sample/class pairs."""
    if logits.shape != targets.shape:
        raise ValueError("Segmentation logits and targets must have equal shapes")
    expected = logits.shape[:2]
    if tuple(supervised_classes.shape) != expected:
        raise ValueError(
            f"Expected supervised_classes shape {expected}, got "
            f"{tuple(supervised_classes.shape)}"
        )
    weights = supervised_classes.to(logits.dtype)
    if class_weights is not None:
        if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
            raise ValueError("class_weights must contain one value per channel")
        weights = (
            weights
            * class_weights.to(
                device=logits.device,
                dtype=logits.dtype,
            )[None, :]
        )
    denominator = weights.sum().clamp(min=1.0)
    per_pixel_bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ).mean(dim=(-1, -2))
    bce = (per_pixel_bce * weights).sum() / denominator
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(-1, -2))
    total = probabilities.sum(dim=(-1, -2)) + targets.sum(dim=(-1, -2))
    per_class_dice_loss = 1.0 - (2.0 * intersection + 1.0) / (total + 1.0)
    dice = (per_class_dice_loss * weights).sum() / denominator
    loss = bce + dice_weight * dice
    return loss, {"bce": bce.detach(), "dice_loss": dice.detach()}


def attachment_localization_loss(
    attachment_logits: torch.Tensor,
    presence_logits: torch.Tensor,
    target_heatmaps: torch.Tensor,
    target_presence: torch.Tensor,
    *,
    spatial_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Train a 3-D endpoint field as slice presence plus in-slice location.

    ``target_presence`` is a soft Gaussian along the through-plane axis. The
    target heatmap supplies x/y location and is normalized before KL loss, so
    the large background area cannot dominate the sparse endpoint signal.
    Spatial loss is weighted by through-plane presence and is therefore absent
    on slices too far from an annotated endpoint.
    """
    if attachment_logits.shape != target_heatmaps.shape:
        raise ValueError("Attachment logits and target heatmaps must match")
    if presence_logits.shape != target_presence.shape:
        raise ValueError("Attachment presence logits and targets must match")
    if attachment_logits.shape[:2] != presence_logits.shape:
        raise ValueError("Attachment channel dimensions do not match")
    presence = F.binary_cross_entropy_with_logits(presence_logits, target_presence)
    flat_target = target_heatmaps.flatten(2)
    target_mass = flat_target.sum(dim=2, keepdim=True)
    distributions = flat_target / target_mass.clamp(min=1e-8)
    log_probabilities = F.log_softmax(attachment_logits.flatten(2), dim=2)
    spatial_per_endpoint = F.kl_div(
        log_probabilities, distributions, reduction="none"
    ).sum(dim=2)
    spatial_mask = target_presence
    spatial = (spatial_per_endpoint * spatial_mask).sum() / spatial_mask.sum().clamp(
        min=1.0
    )
    total = presence + spatial_weight * spatial
    return total, {
        "presence_bce": presence.detach(),
        "spatial_kl": spatial.detach(),
    }


def attachment_slice_classification_loss(
    presence_logits: torch.Tensor,
    target_presence: torch.Tensor,
    *,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Classify each endpoint's annotated slice within one case window.

    The leading dimension must contain every contiguous slice from one knee.
    Unlike independent slice BCE, this objective directly compares candidate
    slices and prevents both endpoint heads from satisfying the loss with the
    same broad through-plane response.
    """
    if presence_logits.ndim != 2 or presence_logits.shape != target_presence.shape:
        raise ValueError("Expected matching [slices, attachments] tensors")
    target_slice = target_presence.argmax(dim=0)
    return F.cross_entropy(
        presence_logits.transpose(0, 1),
        target_slice,
        label_smoothing=label_smoothing,
    )


def attachment_slice_geometry_loss(
    slice_scores: torch.Tensor,
    target_presence: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Regress endpoint slice locations and their signed relative geometry.

    Cross entropy rewards the correct discrete slice, but with few studies two
    heads can still learn nearly identical broad distributions. Their softmax
    expectations provide an ordinal coordinate. The second term explicitly
    teaches whether the femoral endpoint is medial/lateral to the tibial one in
    this consistently ordered case window, including a zero target when both
    clicks belong on the same slice.
    """
    if slice_scores.ndim != 2 or slice_scores.shape != target_presence.shape:
        raise ValueError("Expected matching [slices, attachments] tensors")
    positions = torch.arange(
        slice_scores.shape[0],
        device=slice_scores.device,
        dtype=slice_scores.dtype,
    )
    probabilities = torch.softmax(slice_scores, dim=0)
    predicted = (probabilities * positions[:, None]).sum(dim=0)
    target = target_presence.argmax(dim=0).to(slice_scores.dtype)
    endpoint = F.smooth_l1_loss(predicted, target)
    separation = F.smooth_l1_loss(predicted[0] - predicted[1], target[0] - target[1])
    return endpoint + separation, {
        "slice_ordinal": endpoint.detach(),
        "slice_separation": separation.detach(),
    }
