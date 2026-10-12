import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

GENERIC_WEIGHTS_SHA256 = (
    "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
)


GENERIC_CONFIG_SHA256 = (
    "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_generic_dino(root):
    from transformers import AutoModel

    root = Path(root)
    for name, expected in [
        ("model.safetensors", GENERIC_WEIGHTS_SHA256),
        ("config.json", GENERIC_CONFIG_SHA256),
    ]:
        if sha(root / name) != expected:
            raise ValueError(
                f"Generic-only initialization rejected: {name} checksum differs"
            )
    model = AutoModel.from_pretrained(
        str(root), local_files_only=True, attn_implementation="sdpa"
    )
    if model.config.hidden_size != 384 or model.config.patch_size != 14:
        raise ValueError("Expected generic DINOv2-small/14")
    return model


@dataclass(frozen=True)
class Config:
    width: int = 192
    heads: int = 6
    findings: int = 12
    spatial_queries: int = 4
    depth_layers: int = 2
    sequence_layers: int = 1
    chunk: int = 1
    checkpoint_spatial: bool = True
    checkpoint_depth: bool = True
    max_patches_per_slice: int = 4096  # explicit rejection, never silent resizing
    dropout: float = 0.1


class Readout(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.qnorm = nn.LayerNorm(width)
        self.knorm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, queries, tokens):
        keys = self.knorm(tokens)
        output = (
            queries
            + self.attention(self.qnorm(queries), keys, keys, need_weights=False)[0]
        )
        return output + self.ff(output)


class NativeKnee(nn.Module):
    def __init__(self, dino, config=Config()):
        super().__init__()
        if (
            config.chunk < 1
            or config.spatial_queries < 1
            or config.width % config.heads
        ):
            raise ValueError("Invalid chunk/query/attention dimensions")
        self.config = config
        self.dino = dino
        w, f, k = config.width, config.findings, config.spatial_queries
        self.patch_projection = nn.Linear(384, w)
        self.xy = nn.Sequential(nn.Linear(2, w), nn.GELU(), nn.Linear(w, w))
        self.depth_position = nn.Sequential(nn.Linear(3, w), nn.GELU(), nn.Linear(w, w))
        self.orientation = nn.Linear(8, w)  # 6 direction cosines + 2 pixel spacings
        self.protocol = nn.Embedding(
            12, w
        )  # plane x binary fluid/fat flags; no site/UID input
        self.spatial_queries = nn.Parameter(torch.randn(f * k, w) * 0.02)
        self.depth_queries = nn.Parameter(torch.randn(f, k, w) * 0.02)
        self.study_queries = nn.Parameter(torch.randn(f, 1, w) * 0.02)
        self.spatial_read = Readout(w, config.heads)
        self.depth_read = Readout(w, config.heads)
        self.study_read = Readout(w, config.heads)

        def encoder(layers):
            block = nn.TransformerEncoderLayer(
                w,
                config.heads,
                w * 4,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(block, layers, enable_nested_tensor=False)

        self.depth_encoder = encoder(config.depth_layers)
        self.sequence_encoder = encoder(config.sequence_layers)
        self.norm = nn.LayerNorm(w)
        self.classifier = nn.Parameter(torch.randn(f, w) * 0.02)
        self.bias = nn.Parameter(torch.zeros(f))
        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def set_phase(self, joint):
        self.dino.requires_grad_(joint)
        self.dino.train(self.training and joint)

    def _spatial_chunk(self, images_cpu, spacing_cpu):
        device = self.classifier.device
        images = images_cpu.to(device=device, dtype=torch.float32)
        n, h, w = images.shape
        pad_h, pad_w = (-h) % 14, (-w) % 14
        # Edge replication pads at most 13 px; no whole-patch padding or crop.
        pixels = F.pad(images[:, None], (0, pad_w, 0, pad_h), mode="replicate")
        gh, gw = pixels.shape[-2] // 14, pixels.shape[-1] // 14
        if gh * gw > self.config.max_patches_per_slice:
            raise ValueError(
                f"Native grid {gh}x{gw} exceeds declared memory guard; no automatic downsampling"
            )
        pixels = (pixels.expand(-1, 3, -1, -1) - self.rgb_mean) / self.rgb_std
        patches = self.dino(pixel_values=pixels).last_hidden_state[:, 1:]
        if patches.shape[1:] != (gh * gw, 384):
            raise ValueError("Unexpected native DINO patch grid")
        yy, xx = torch.meshgrid(
            torch.arange(gh, device=device),
            torch.arange(gw, device=device),
            indexing="ij",
        )
        spacing = spacing_cpu.to(device)
        coords = (
            torch.stack(
                (
                    (xx.flatten() * 14 + 6.5 - (w - 1) / 2) * spacing[1],
                    (yy.flatten() * 14 + 6.5 - (h - 1) / 2) * spacing[0],
                ),
                -1,
            ).float()
            / 100
        )
        tokens = self.patch_projection(patches) + self.xy(coords)[None]
        # Spatial reasoning has occurred inside DINO; queries retain several
        # distinct evidence vectors per finding instead of a global mean.
        queries = self.spatial_queries[None].expand(n, -1, -1)
        return self.spatial_read(queries, tokens).reshape(
            n, self.config.findings, self.config.spatial_queries, -1
        )

    def forward(self, series):
        """One variable-size exam: list of CPU native stacks and physical metadata.

        No missing-series token, slot padding, random slices, or rigid shape batch.
        Slice position travels with each feature; array reordering alone is not
        meaningful if coordinates travel too. Content-to-position pairing is.
        """
        if not series:
            raise ValueError("No imaging: v4 refuses to train on prior-only examples")
        c = self.config
        outputs = []
        for item in series:
            pixels, z = item["pixels"], item["positions_mm"]
            if pixels.ndim != 3 or len(pixels) != len(z) or len(z) == 0:
                raise ValueError("Slice/geometry alignment error")
            if not torch.isfinite(pixels).all() or not torch.isfinite(z).all():
                raise ValueError("Nonfinite imaging or geometry")
            chunks = []
            for part in pixels.split(c.chunk):
                # CPU inputs move to GPU INSIDE recomputation: previous native
                # image chunks need not stay on GPU for the full exam backward.
                chunks.append(
                    checkpoint(
                        self._spatial_chunk,
                        part,
                        item["spacing_mm"],
                        use_reentrant=False,
                    )
                    if c.checkpoint_spatial
                    and self.training
                    and torch.is_grad_enabled()
                    else self._spatial_chunk(part, item["spacing_mm"])
                )
            spatial = torch.cat(chunks, 0)
            device = spatial.device
            z = z.to(device)
            centered = z - (z[0] + z[-1]) / 2
            geom = torch.stack(
                (centered / 100, torch.sin(centered / 10), torch.cos(centered / 10)), -1
            )
            tokens = spatial + self.depth_position(geom)[:, None, None, :]
            tokens = tokens.permute(1, 0, 2, 3).reshape(c.findings, -1, c.width)
            contextual = (
                checkpoint(self.depth_encoder, tokens, use_reentrant=False)
                if c.checkpoint_depth and self.training and torch.is_grad_enabled()
                else self.depth_encoder(tokens)
            )
            summary = self.depth_read(self.depth_queries, contextual)
            meta = torch.cat((item["orientation"], item["spacing_mm"] / 10)).to(device)
            protocol = torch.tensor(item["protocol"], device=device, dtype=torch.long)
            summary = summary + self.orientation(meta) + self.protocol(protocol)
            outputs.append(summary)
        sequences = torch.cat(outputs, 1)  # F x (series*K) x width, never mean(S)
        contextual = self.sequence_encoder(sequences)
        study = self.norm(self.study_read(self.study_queries, contextual)[:, 0])
        return (study * self.classifier).sum(-1) + self.bias

    def architecture(self):
        return asdict(self.config)
