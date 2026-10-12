"""Frozen ACL localizer used only to anchor classifier crops."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

from .model import sha

CLASS_NAMES = ("acl", "pcl")
ATTACHMENT_NAMES = ("femoral_attachment", "tibial_attachment")
ARCHITECTURE_VERSION = "cruciate_segmenter_2p5d_v2"


class ACLAnchor:
    def __init__(self, checkpoint, split, device="cpu"):

        self.device = torch.device(device)
        checkpoint = Path(checkpoint)
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = blob["config"]
        if self.config["architecture_version"] != ARCHITECTURE_VERSION:
            raise ValueError("Unsupported ACL localizer")
        if self.config.get("acquisition_plane") != "sagittal" or self.config.get(
            "canonical_medial_side_alignment", False
        ):
            raise ValueError("Unsupported localizer orientation contract")
        previous = pd.read_csv(
            checkpoint.parent / "attachment_predictions.csv", dtype=str
        )
        previous = previous[["StudyInstanceUID", "data_split"]].drop_duplicates()
        current = pd.read_csv(split, dtype=str)
        if (
            previous.StudyInstanceUID.duplicated().any()
            or current.StudyInstanceUID.duplicated().any()
        ):
            raise ValueError("Ambiguous localizer split membership")
        matched = previous.merge(
            current, on="StudyInstanceUID", how="left", validate="one_to_one"
        )
        if matched.data_split.ne(matched.split).any():
            raise ValueError("Localizer/current split mismatch")
        if int(previous.data_split.eq("weak_train").sum()) != int(
            self.config["train_studies"]
        ):
            raise ValueError("Localizer cohort record incomplete")
        self.model = (
            CruciateSegmenter2p5D(base_channels=int(self.config["base_channels"]))
            .to(self.device)
            .eval()
        )
        self.model.load_state_dict(blob["model"], strict=True)
        self.model.requires_grad_(False)
        self.provenance = {
            "checkpoint_sha256": sha(checkpoint),
            "split_sha256": sha(Path(split)),
            "membership_audit_studies": len(matched),
            "split_mismatches": 0,
            "trained_on_competition_data": True,
            "selected_on_weak_dev": True,
            "search": "all_native_slices_no_human_prompt",
            "architecture_source_sha256": sha(Path(__file__)),
        }

    @torch.inference_mode()
    def predict(self, volume, series_uid):
        from .anatomy import attachment_slice_scores, normalize_volume

        raw = normalize_volume(volume["pixels"])
        small = F.interpolate(
            torch.from_numpy(raw)[:, None],
            size=(int(self.config["image_size"]),) * 2,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        # Preserve LOCALIZER training preprocessing only; not classifier pixels.
        small = (small.clamp(0, 1) * 255).to(torch.uint8).float() / 255
        heatmaps = []
        presence = []
        for start in range(0, len(small), 8):
            ids = torch.arange(start, min(start + 8, len(small)))
            triplets = small[
                (ids[:, None] + torch.tensor([-1, 0, 1])[None]).clamp(0, len(small) - 1)
            ]
            out = self.model(triplets.to(self.device))
            heatmaps.append(out["attachment_logits"].cpu())
            presence.append(out["attachment_presence_logits"].cpu())
        maps = torch.cat(heatmaps)
        scores = attachment_slice_scores(
            torch.cat(presence),
            maps,
            spatial_evidence_weight=float(
                self.config.get("spatial_evidence_weight", 0)
            ),
        )
        endpoints = []
        confidences = []
        reasons = []
        indices = []
        for k in range(2):
            z = int(scores[:, k].argmax())
            indices.append(z)
            weights = maps[z, k].flatten().softmax(0).reshape(maps.shape[-2:])
            x = float((weights.sum(0) * torch.linspace(0, 1, weights.shape[1])).sum())
            y = float((weights.sum(1) * torch.linspace(0, 1, weights.shape[0])).sum())
            confidence = float(scores[z, k].sigmoid())
            confidences.append(confidence)
            if confidence < 0.5:
                reasons.append("low_endpoint_score")
            if not 0.1 < x < 0.9 or not 0.1 < y < 0.9:
                reasons.append("endpoint_near_inplane_edge")
            if z in (0, len(small) - 1):
                reasons.append("endpoint_at_stack_edge")
            sp = volume["spacing_mm"]
            iop = volume["orientation"]
            h, w = volume["pixels"].shape[1:]
            endpoints.append(
                volume["origins_mm"][z]
                + iop[:3] * (x * (w - 1) * sp[1])
                + iop[3:] * (y * (h - 1) * sp[0])
            )
        length = float(np.linalg.norm(endpoints[0] - endpoints[1]))
        if not 10 <= length <= 60:
            reasons.append("implausible_endpoint_separation")
        if (
            abs(
                float(
                    volume["positions_mm"][indices[0]]
                    - volume["positions_mm"][indices[1]]
                )
            )
            > 20
        ):
            reasons.append("endpoints_far_apart_through_plane")
        return {
            "source_series_uid": str(series_uid),
            "frame_of_reference_uid": str(volume.get("frame_of_reference_uid", "")),
            "center_lps_mm": np.mean(endpoints, axis=0).tolist(),
            "endpoints_lps_mm": np.asarray(endpoints).tolist(),
            "endpoint_scores": confidences,
            "endpoint_separation_mm": length,
            "accepted": not reasons,
            "reasons": sorted(set(reasons)),
            "score_is_calibrated_probability": False,
        }


def anchor_for_series(proposals, series_uid, frame_uid, max_disagreement_mm=20):
    compatible = [
        p
        for p in proposals
        if p["accepted"]
        and (
            p["source_series_uid"] == str(series_uid)
            or bool(frame_uid)
            and p["frame_of_reference_uid"] == str(frame_uid)
        )
    ]
    if not compatible:
        return None
    centers = np.asarray([p["center_lps_mm"] for p in compatible])
    if (
        np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max()
        > max_disagreement_mm
    ):
        return None
    # Choose one actual prediction, not averaged landmarks from different series.
    return max(compatible, key=lambda p: min(p["endpoint_scores"]))


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        groups = min(8, output_channels)
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, inputs):
        return self.block(inputs)


class CruciateSegmenter2p5D(nn.Module):
    """Small U-Net architecture used by the frozen ACL crop-localizer checkpoint."""

    def __init__(self, base_channels: int = 24):
        super().__init__()
        self.encoder1 = ConvBlock(3, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.decoder2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.decoder1 = ConvBlock(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, len(CLASS_NAMES), 1)
        self.attachment_output = nn.Conv2d(base_channels, len(ATTACHMENT_NAMES), 1)
        self.attachment_presence = nn.Linear(base_channels * 4, len(ATTACHMENT_NAMES))

    def forward(self, slice_triplets):
        if slice_triplets.ndim != 4 or slice_triplets.shape[1] != 3:
            raise ValueError("Expected 2.5-D input shaped [batch,3,height,width]")
        level1 = self.encoder1(slice_triplets)
        level2 = self.encoder2(F.max_pool2d(level1, 2))
        encoded = self.bottleneck(F.max_pool2d(level2, 2))
        up2 = F.interpolate(
            encoded, size=level2.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded2 = self.decoder2(torch.cat([up2, level2], dim=1))
        up1 = F.interpolate(
            decoded2, size=level1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded1 = self.decoder1(torch.cat([up1, level1], dim=1))
        logits = self.output(decoded1)
        attachment_logits = self.attachment_output(decoded1)
        attachment_presence_logits = self.attachment_presence(
            F.adaptive_avg_pool2d(encoded, 1).flatten(1)
        )
        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "attachment_logits": attachment_logits,
            "attachment_presence_logits": attachment_presence_logits,
            "attachment_presence": torch.sigmoid(attachment_presence_logits),
        }
