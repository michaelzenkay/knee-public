"""KneeXNet-compatible ResNet-34 U-Net with project-trained ACL/MCL heads."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ATTACHMENT_NAMES = ("femoral_attachment", "tibial_attachment")
KNEE_STRUCTURE_CLASS_NAMES = (
    "femoral_cartilage",
    "tibial_cartilage",
    "patellar_cartilage",
    "meniscus",
)


class ResNet34BasicBlock(nn.Module):
    """Torch-only ResNet-34 block with KneeXNet-compatible parameter names."""

    expansion = 1

    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        identity = inputs
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(output + identity)


class KneeXNetCompatibleResNet34Encoder(nn.Module):
    """Five-scale ResNet-34 matching the released KneeXNet state dictionary."""

    out_channels = (3, 64, 64, 128, 256, 512)

    def __init__(self):
        super().__init__()
        self._current_channels = 64
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=3)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)
        self._reset_parameters()

    def _make_layer(self, output_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self._current_channels != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self._current_channels,
                    output_channels,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        layers = [
            ResNet34BasicBlock(
                self._current_channels,
                output_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self._current_channels = output_channels
        layers.extend(
            ResNet34BasicBlock(output_channels, output_channels)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        features = [inputs]
        output = self.relu(self.bn1(self.conv1(inputs)))
        features.append(output)
        output = self.layer1(self.maxpool(output))
        features.append(output)
        output = self.layer2(output)
        features.append(output)
        output = self.layer3(output)
        features.append(output)
        output = self.layer4(output)
        features.append(output)
        return features


class KneeXNetDecoderConv(nn.Sequential):
    """Conv-BN-ReLU named like SMP decoder Conv2dReLU."""

    def __init__(self, input_channels, output_channels):
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class KneeXNetDecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.conv1 = KneeXNetDecoderConv(
            input_channels + skip_channels,
            output_channels,
        )
        self.attention1 = nn.Identity()
        self.conv2 = KneeXNetDecoderConv(output_channels, output_channels)
        self.attention2 = nn.Identity()

    def forward(self, inputs, output_size, skip=None):
        output = F.interpolate(inputs, size=output_size, mode="nearest")
        if skip is not None:
            output = self.attention1(torch.cat([output, skip], dim=1))
        output = self.conv1(output)
        return self.attention2(self.conv2(output))


class KneeXNetCompatibleUnetDecoder(nn.Module):
    """SMP U-Net decoder layout used by the released KneeXNet checkpoints."""

    def __init__(self):
        super().__init__()
        self.center = nn.Identity()
        self.blocks = nn.ModuleList(
            [
                KneeXNetDecoderBlock(512, 256, 256),
                KneeXNetDecoderBlock(256, 128, 128),
                KneeXNetDecoderBlock(128, 64, 64),
                KneeXNetDecoderBlock(64, 64, 32),
                KneeXNetDecoderBlock(32, 0, 16),
            ]
        )

    def forward(self, features):
        spatial_shapes = [feature.shape[-2:] for feature in features][::-1]
        reversed_features = features[1:][::-1]
        output = self.center(reversed_features[0])
        skips = reversed_features[1:]
        for index, block in enumerate(self.blocks):
            skip = skips[index] if index < len(skips) else None
            output = block(output, spatial_shapes[index + 1], skip)
        return output


class FusedKneeAnatomyResNet34Unet2p5D(nn.Module):
    """Deep shared anatomy model compatible with a KneeXNet ResNet-34 U-Net."""

    ATTACHMENT_TASKS = ("acl", "mcl")
    TEACHER_FOREGROUND_CHANNELS = (1, 2, 3, 4)

    def __init__(self, *, freeze_batch_norm_statistics=False):
        super().__init__()
        self.encoder = KneeXNetCompatibleResNet34Encoder()
        self.decoder = KneeXNetCompatibleUnetDecoder()
        # Keep the released five-channel head intact. Our four independent
        # sigmoid tasks consume foreground channels 1:5; channel 0 is the
        # teacher background logit and intentionally receives no partial-mask
        # loss. Preserving this shape makes checkpoint transfer exact.
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(16, 5, kernel_size=3, padding=1),
        )
        self.attachment_outputs = nn.ModuleDict(
            {
                task: nn.Conv2d(16, len(ATTACHMENT_NAMES), 1)
                for task in self.ATTACHMENT_TASKS
            }
        )
        self.attachment_presence = nn.ModuleDict(
            {
                task: nn.Linear(512, len(ATTACHMENT_NAMES))
                for task in self.ATTACHMENT_TASKS
            }
        )
        self.freeze_batch_norm_statistics = bool(freeze_batch_norm_statistics)
        self.encoder_training_stage = "full"

    def encode(self, slice_triplets):
        if slice_triplets.ndim != 4 or slice_triplets.shape[1] != 3:
            raise ValueError("Expected 2.5-D input shaped [batch,3,height,width]")
        return self.encoder(slice_triplets)

    def decode(self, features):
        return self.decoder(features)

    def forward(self, slice_triplets):
        features = self.encode(slice_triplets)
        decoded = self.decode(features)
        teacher_logits = self.segmentation_head(decoded)
        segmentation_logits = teacher_logits[:, self.TEACHER_FOREGROUND_CHANNELS]
        encoded = features[-1]
        pooled = F.adaptive_avg_pool2d(encoded, 1).flatten(1)
        result = {
            "logits": segmentation_logits,
            "probabilities": torch.sigmoid(segmentation_logits),
            "segmentation_logits": segmentation_logits,
            "segmentation_probabilities": torch.sigmoid(segmentation_logits),
            "teacher_background_logit": teacher_logits[:, :1],
            "encoded": encoded,
        }
        for task in self.ATTACHMENT_TASKS:
            attachment_logits = self.attachment_outputs[task](decoded)
            presence_logits = self.attachment_presence[task](pooled)
            result[task + "_attachment_logits"] = attachment_logits
            result[task + "_attachment_presence_logits"] = presence_logits
            result[task + "_attachment_presence"] = torch.sigmoid(presence_logits)
        return result

    def initialize_from_kneexnet(self, checkpoint_path):
        """Load the released segmentation trunk and report exact provenance."""
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state, dict):
            raise TypeError("KneeXNet checkpoint must contain a state dictionary")
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        incompatible = self.load_state_dict(state, strict=False)
        allowed_missing_prefixes = (
            "attachment_outputs.",
            "attachment_presence.",
        )
        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if incompatible.unexpected_keys or disallowed_missing:
            raise RuntimeError(
                "KneeXNet checkpoint is not exactly trunk-compatible: "
                f"unexpected={incompatible.unexpected_keys}, "
                f"missing={disallowed_missing}"
            )
        return {
            "source": "KneeXNet-2.5D released segmentation checkpoint",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": digest.hexdigest(),
            "loaded_parameter_tensors": len(state),
            "new_random_head_tensors": list(incompatible.missing_keys),
            "foreground_channel_mapping": {
                name: channel
                for name, channel in zip(
                    KNEE_STRUCTURE_CLASS_NAMES,
                    self.TEACHER_FOREGROUND_CHANNELS,
                )
            },
        }

    def optimizer_parameter_groups(
        self,
        *,
        encoder_lr,
        decoder_lr,
        head_lr,
        weight_decay,
    ):
        return [
            {
                "name": "encoder",
                "params": list(self.encoder.parameters()),
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "name": "decoder",
                "params": list(self.decoder.parameters()),
                "lr": decoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "name": "heads",
                "params": [
                    *self.segmentation_head.parameters(),
                    *self.attachment_outputs.parameters(),
                    *self.attachment_presence.parameters(),
                ],
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
        ]

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_batch_norm_statistics:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def set_encoder_training_stage(self, stage):
        if stage not in {"frozen", "deep", "full"}:
            raise ValueError("encoder stage must be frozen, deep, or full")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = stage == "full"
        if stage == "deep":
            for parameter in self.encoder.layer4.parameters():
                parameter.requires_grad = True
        self.encoder_training_stage = stage


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    if not finite.size:
        return np.zeros_like(volume, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        high = low + 1.0
    return np.clip((volume.astype(np.float32) - low) / (high - low), 0, 1)


def slice_triplet(volume: np.ndarray, index: int) -> np.ndarray:
    indices = np.clip([index - 1, index, index + 1], 0, len(volume) - 1)
    return volume[indices].astype(np.float32, copy=False)


def attachment_slice_scores(
    presence_logits: torch.Tensor,
    attachment_logits: torch.Tensor,
    *,
    spatial_evidence_weight: float = 0.0,
) -> torch.Tensor:
    """Combine global slice evidence with endpoint-map confidence.

    The presence logits see a globally pooled encoded slice. The spatial
    endpoint maps retain more local evidence but have an arbitrary logit scale.
    We therefore standardize their log-mean-exp evidence across the contiguous
    case window before adding it. A zero weight uses presence logits alone.
    """
    if presence_logits.ndim != 2:
        raise ValueError("Expected presence logits shaped [slices, attachments]")
    if attachment_logits.ndim != 4:
        raise ValueError(
            "Expected attachment logits shaped [slices, attachments, height, width]"
        )
    if attachment_logits.shape[:2] != presence_logits.shape:
        raise ValueError("Presence and spatial attachment channels must match")
    if spatial_evidence_weight == 0:
        return presence_logits
    flattened = attachment_logits.flatten(2)
    spatial_evidence = torch.logsumexp(flattened, dim=2) - np.log(flattened.shape[2])
    spatial_evidence = spatial_evidence - spatial_evidence.mean(dim=0, keepdim=True)
    spatial_scale = spatial_evidence.std(dim=0, keepdim=True, unbiased=False).clamp(
        min=1e-6
    )
    return presence_logits + spatial_evidence_weight * (
        spatial_evidence / spatial_scale
    )
