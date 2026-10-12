"""DICOM series to physically ordered, cropped classifier inputs."""

from pathlib import Path

from .crops import CropPolicy, apply_crop, plan_crop
from .data import decode_native, model_input, safe_id
from .localizer import ACLAnchor, anchor_for_series


class ExamPreprocessor:
    def __init__(self, checkpoint=None, split=None, device="cpu"):
        if (checkpoint is None) != (split is None):
            raise ValueError("Supply both localizer checkpoint and its split audit CSV")
        self.localizer = ACLAnchor(checkpoint, split, device) if checkpoint else None
        self.policy = CropPolicy()

    def process(self, raw_root, uid, rows):
        volumes = []
        proposals = []
        uid = safe_id(uid)
        seen = set()
        planes = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
        for row in rows:
            sid = safe_id(row["SeriesInstanceUID"])
            if sid in seen:
                raise ValueError("Duplicate series in exam")
            seen.add(sid)
            if row.get("StudyInstanceUID") != uid:
                raise ValueError("Series belongs to another study")
            if row["Anatomical_Plane"] not in planes:
                raise ValueError("Unsupported plane")
            if str(row["Fluid_Sensitive"]) not in ("0", "1") or str(
                row["Fat_Suppression"]
            ) not in ("0", "1"):
                raise ValueError("Invalid sequence flags")
            folder = Path(raw_root) / uid / sid
            volume = decode_native(folder)
            if row["Anatomical_Plane"] == "Sagittal" and self.localizer is not None:
                proposals.append(self.localizer.predict(volume, sid))
            protocol = (
                planes[row["Anatomical_Plane"]] * 4
                + int(row["Fluid_Sensitive"]) * 2
                + int(row["Fat_Suppression"])
            )
            volumes.append((sid, protocol, volume))
        if not volumes:
            raise ValueError("Empty exam")
        processed = []
        for sid, protocol, volume in volumes:
            anchor = anchor_for_series(
                proposals,
                sid,
                str(volume["frame_of_reference_uid"]),
                self.policy.max_center_disagreement_mm,
            )
            plan = plan_crop(volume, anchor, self.policy)
            out = apply_crop(volume, plan)
            processed.append(
                (
                    out,
                    {
                        "series_uid": sid,
                        "protocol": protocol,
                        "crop": plan,
                        "anchor": anchor,
                        "source_sha256": str(volume["source_sha256"]),
                    },
                )
            )
        return processed, proposals

    def model_inputs(self, raw_root, uid, rows):
        """Inference uses exactly the same transform and normalization as caching."""
        processed, proposals = self.process(raw_root, uid, rows)
        return [model_input(v, m["protocol"]) for v, m in processed], {
            "series": [m for v, m in processed],
            "proposals": proposals,
        }
