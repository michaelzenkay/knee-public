import hashlib
import io
from pathlib import Path

import numpy as np
import pydicom
import torch

FINDINGS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)


POLICY = "acl_anchored_acquisition_float32_all_listed_series_v1"


def safe_id(uid):
    if (
        not isinstance(uid, str)
        or not uid
        or uid in (".", "..")
        or any(c in uid for c in "/\\:")
    ):
        raise ValueError("Unsafe study/series identifier")
    return uid


def decode_native(directory):
    records = []
    source_hash = hashlib.sha256()
    paths = sorted(
        p
        for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in ("", ".dcm", ".dicom")
    )
    if not paths:
        raise ValueError("Empty native acquisition")
    for path in paths:
        raw = path.read_bytes()
        source_hash.update(hashlib.sha256(raw).digest())
        ds = pydicom.dcmread(io.BytesIO(raw))
        array = ds.pixel_array.astype(np.float32)
        if array.ndim != 2:
            raise ValueError(
                "Multi-frame input needs explicit support, not silent slice exclusion"
            )
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        array = array * slope + intercept
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            array = -array
        spacing = np.asarray(ds.PixelSpacing, dtype=np.float32)
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float32)
        origin = np.asarray(ds.ImagePositionPatient, dtype=np.float32)
        if spacing.shape != (2,) or orientation.shape != (6,) or origin.shape != (3,):
            raise ValueError("Invalid DICOM geometry")
        if (
            not np.isfinite(array).all()
            or not np.isfinite(np.r_[spacing, orientation, origin]).all()
            or (spacing <= 0).any()
        ):
            raise ValueError("Invalid native pixels or geometry")
        if (
            not np.allclose(
                [np.linalg.norm(orientation[:3]), np.linalg.norm(orientation[3:])],
                [1, 1],
                atol=0.01,
            )
            or abs(orientation[:3] @ orientation[3:]) > 0.01
        ):
            raise ValueError("Non-orthonormal DICOM orientation")
        if records and (
            array.shape != records[0][1].shape
            or not np.allclose(spacing, records[0][2], atol=1e-5)
            or not np.allclose(orientation, records[0][3], atol=1e-4)
        ):
            raise ValueError(
                "Mixed acquisition grid/orientation requires explicit handling"
            )
        z = float(origin @ np.cross(orientation[:3], orientation[3:]))
        records.append(
            (
                z,
                array,
                spacing,
                orientation,
                origin,
                str(getattr(ds, "FrameOfReferenceUID", "")),
                tuple(int(v) for v in getattr(ds, "AcquisitionMatrix", [])),
            )
        )
    records.sort(key=lambda r: r[0])
    positions = np.asarray([r[0] for r in records], np.float32)
    if len(positions) > 1 and (np.diff(positions) <= 1e-4).any():
        raise ValueError("Repeated slice positions; do not mix echoes/volumes")
    return dict(
        pixels=np.stack([r[1] for r in records]),
        positions_mm=positions,
        spacing_mm=records[0][2],
        orientation=records[0][3],
        origins_mm=np.stack([r[4] for r in records]),
        source_sha256=np.asarray(source_hash.hexdigest()),
        frame_of_reference_uid=np.asarray(
            records[0][5] if len({r[5] for r in records}) == 1 else ""
        ),
        acquisition_matrix=np.asarray(
            records[0][6] if len({r[6] for r in records}) == 1 else [], dtype=np.int32
        ),
    )


def normalize_float(volume):
    """Smooth bounded floating-point intensity transform, no percentile clipping.

    p1/p99 determine center/scale ONLY. Arctan still compresses contrast; it is not
    a claim of unchanged intensity. Cached float32 pixels remain untransformed.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("Nonfinite native volume")
    lo, hi = np.percentile(volume, [1, 99])
    scale = max((hi - lo) / 2, 1e-6)
    return (0.5 + np.arctan((volume - (hi + lo) / 2) / scale) / np.pi).astype(
        np.float32
    )


def model_input(arrays, protocol):
    entry = {
        key: torch.from_numpy(np.asarray(arrays[key]).copy())
        for key in ("positions_mm", "spacing_mm", "orientation")
    }
    entry["pixels"] = torch.from_numpy(normalize_float(arrays["pixels"]))
    entry["protocol"] = protocol
    return entry


def load_study(root, record):
    series = []
    for item in record["series"]:
        path = (Path(root) / item["path"]).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("Unsafe cache path")
        with np.load(path, allow_pickle=False) as arrays:
            if arrays["pixels"].dtype != np.float32:
                raise ValueError("Not a native float32 cache")
            entry = model_input(arrays, item["protocol"])
        series.append(entry)
    return series, torch.tensor(record["targets"], dtype=torch.float32)
