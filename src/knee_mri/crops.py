from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F


@dataclass(frozen=True)
class CropPolicy:
    # LPS millimeters: +/- left-right, anterior/posterior, inferior/superior.
    # These are minimum retained extents, never a hard maximum crop.
    left_right_mm: float = 80
    anterior_mm: float = 80
    posterior_mm: float = 80
    inferior_mm: float = 70
    superior_mm: float = 110
    tissue_margin_mm: float = 12
    sampling_policy: str = "acquisition_limited_common_spacing_v1"
    max_center_disagreement_mm: float = 20


def foreground_bounds(pixels, spacing):
    """Union over EVERY slice; protects tissue but is not a segmentation label.

    Low threshold, retaining all sufficiently large connected components. Noise,
    coil signal, or ambiguous foreground may preserve the full field intentionally.
    """
    d, h, w = pixels.shape
    # The analysis thumbnail only determines a conservative outer tissue bound;
    # classifier pixels are not generated from it.
    step = max(1, int(np.ceil(max(h, w) / 256)))
    small = pixels[:, ::step, ::step].astype(np.float32)
    lo, hi = np.percentile(small, [1, 99])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return None
    union = np.zeros(small.shape[1:], bool)
    for plane in small:
        mask = ndimage.gaussian_filter(plane, sigma=0.7) > lo + 0.035 * (hi - lo)
        labels, n = ndimage.label(mask)
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        good = counts >= max(9, int(0.001 * mask.size))
        union |= good[labels]
    yy, xx = np.nonzero(union)
    if len(yy) < 0.01 * union.size:
        return None
    return [
        int(yy.min() * step),
        min(h, int((yy.max() + 1) * step)),
        int(xx.min() * step),
        min(w, int((xx.max() + 1) * step)),
    ]


def project_box(volume, center, policy):
    """Conservative enclosing rectangle in each source image, including obliquity."""
    center = np.asarray(center, float)
    low = center + [-policy.left_right_mm, -policy.anterior_mm, -policy.inferior_mm]
    high = center + [policy.left_right_mm, policy.posterior_mm, policy.superior_mm]
    corners = np.asarray(list(product(*zip(low, high))))
    orientation = np.asarray(volume["orientation"], float)
    origins = np.asarray(volume["origins_mm"], float)
    spacing = np.asarray(volume["spacing_mm"], float)
    # IOP first triplet follows increasing COLUMN; second follows increasing ROW.
    x = (corners @ orientation[:3])[None, :] - (origins @ orientation[:3])[:, None]
    y = (corners @ orientation[3:])[None, :] - (origins @ orientation[3:])[:, None]
    return [
        int(np.floor(y.min() / spacing[0])),
        int(np.ceil(y.max() / spacing[0])) + 1,
        int(np.floor(x.min() / spacing[1])),
        int(np.ceil(x.max() / spacing[1])) + 1,
    ]


def sampling_spacing(volume, policy):
    """Only reduce demonstrably expanded reconstruction grids, conservatively.

    AcquisitionMatrix is a heuristic bound, not a recovered k-space resolution.
    Missing/ambiguous metadata retains original spacing. Never upsample.
    """
    spacing = np.asarray(volume["spacing_mm"], float)
    matrix = np.asarray(volume.get("acquisition_matrix", []), int)
    if matrix.shape != (4,) or (matrix < 0).any():
        return spacing, "missing_or_invalid_acquisition_matrix"
    rows = matrix[[0, 2]]
    cols = matrix[[1, 3]]
    if any(np.count_nonzero(a) != 1 for a in (rows, cols, matrix[:2], matrix[2:])):
        return spacing, "ambiguous_acquisition_matrix"
    counts = np.asarray([rows.sum(), cols.sum()], float)
    ratio = np.asarray(volume["pixels"].shape[1:]) / counts
    if (ratio < 1).any():
        return spacing, "inconsistent_acquisition_matrix"
    # Per-axis cap prevents discarding an axis whose acquired sampling is finer.
    cap = spacing * ratio
    # Do not independently coarsen axes: that stretches anatomy in pixel space.
    # The finer acquired axis limits BOTH. Already-anisotropic native pixels may
    # remain anisotropic when equalizing would require upsampling.
    target = np.maximum(spacing, float(cap.min()))
    return target, "acquisition_bounded"


def plan_crop(volume, anchor=None, policy=CropPolicy()):
    pixels = volume["pixels"]
    d, h, w = pixels.shape
    spacing = np.asarray(volume["spacing_mm"], float)
    full = [0, h, 0, w]
    bounds = full.copy()
    reasons = []
    minimum = None
    if anchor is None or not anchor.get("accepted", False):
        reasons.append("no_accepted_acl_anchor")
    elif (
        np.asarray(anchor.get("center_lps_mm", [])).shape != (3,)
        or not np.isfinite(np.asarray(anchor["center_lps_mm"], float)).all()
    ):
        reasons.append("invalid_anchor")
    else:
        tissue = foreground_bounds(pixels, spacing)
        if tissue is None:
            reasons.append("ambiguous_tissue_envelope")
        else:
            minimum = project_box(volume, anchor["center_lps_mm"], policy)
            my, mx = np.ceil(policy.tissue_margin_mm / spacing).astype(int)
            protected = [tissue[0] - my, tissue[1] + my, tissue[2] - mx, tissue[3] + mx]
            bounds = [
                max(0, min(minimum[0], protected[0])),
                min(h, max(minimum[1], protected[1])),
                max(0, min(minimum[2], protected[2])),
                min(w, max(minimum[3], protected[3])),
            ]
            if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
                bounds = full.copy()
                reasons.append("invalid_crop_fallback")
    desired, spacing_reason = sampling_spacing(volume, policy)
    size = np.asarray([bounds[1] - bounds[0], bounds[3] - bounds[2]])
    output = np.minimum(
        size, np.maximum(1, np.ceil(size * spacing / desired).astype(int))
    )
    actual = spacing * size / output
    return {
        "policy": asdict(policy),
        "bounds_yxyx": bounds,
        "input_shape": [d, h, w],
        "output_shape": [d, *map(int, output)],
        "output_spacing_mm": actual.tolist(),
        "all_slices_retained": True,
        "fallback_reasons": reasons,
        "spacing_reason": spacing_reason,
        "minimum_box_outside_source": bool(
            minimum is not None
            and (minimum[0] < 0 or minimum[1] > h or minimum[2] < 0 or minimum[3] > w)
        ),
        "cropped_area_fraction": 1 - float(np.prod(size) / (h * w)),
        "patch_fraction": float(
            np.prod(np.ceil(output / 14)) / np.prod(np.ceil(np.array([h, w]) / 14))
        ),
    }


def apply_crop(volume, plan):
    """One float anti-aliased resampling; preserve physical pixel-center geometry."""
    y0, y1, x0, x1 = plan["bounds_yxyx"]
    pixels = volume["pixels"][:, y0:y1, x0:x1]
    if list(volume["pixels"].shape) != plan["input_shape"]:
        raise ValueError("Plan/source shape mismatch")
    size = plan["output_shape"][1:]
    out = dict(volume)
    if list(pixels.shape[1:]) != size:
        pixels = F.interpolate(
            torch.from_numpy(pixels.copy())[:, None],
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[:, 0].numpy()
    out["pixels"] = np.ascontiguousarray(pixels, dtype=np.float32)
    old = np.asarray(volume["spacing_mm"], float)
    new = np.asarray(plan["output_spacing_mm"], float)
    iop = np.asarray(volume["orientation"], float)
    offset = iop[:3] * (x0 * old[1] + (new[1] - old[1]) / 2) + iop[3:] * (
        y0 * old[0] + (new[0] - old[0]) / 2
    )
    out["origins_mm"] = (np.asarray(volume["origins_mm"]) + offset).astype(np.float32)
    out["spacing_mm"] = new.astype(np.float32)
    return out
