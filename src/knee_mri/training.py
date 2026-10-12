"""Classifier training from prepared studies, without distributed/run management."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import FINDINGS, load_study
from .evaluation import evaluate_classifier
from .model import Config, NativeKnee, load_generic_dino, sha


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 24
    warmup_epochs: int = 2
    patience: int = 5
    accumulation: int = 8
    encoder_lr: float = 2e-6
    head_lr: float = 1e-4
    seed: int = 20260913


def validate_records(records):
    """The input manifest contains only study-disjoint training/development data."""
    ids = [r["uid"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate study ID")
    if {r["split"] for r in records} != {"weak_train", "weak_dev"}:
        raise ValueError("Expected nonempty weak_train and weak_dev partitions")
    patients = {}
    for record in records:
        if record.get("patient_id"):
            patient = record["patient_id"]
            if patient in patients and patients[patient] != record["split"]:
                raise ValueError("Patient occurs in both training and development")
            patients[patient] = record["split"]
        target = np.asarray(record["targets"])
        if (
            target.shape != (12,)
            or not np.isfinite(target).all()
            or np.any((target < 0) | (target > 1))
        ):
            raise ValueError("Each study needs twelve finite targets in [0, 1]")
        if not record["series"]:
            raise ValueError("Study has no imaging")


def train_classifier(
    cache, generic_dino, output, *, config=TrainingConfig(), device="cuda"
):
    """Start a new classifier run from generic DINOv2 and a prepared-study manifest."""
    if (
        min(config.epochs, config.patience, config.accumulation) < 1
        or not 0 <= config.warmup_epochs < config.epochs
    ):
        raise ValueError("Invalid epoch, warmup or accumulation settings")
    cache, output = Path(cache).resolve(), Path(output)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest["findings"] != list(FINDINGS):
        raise ValueError("Finding order differs from model outputs")
    records = manifest["studies"]
    validate_records(records)
    train = [r for r in records if r["split"] == "weak_train"]
    dev = [r for r in records if r["split"] == "weak_dev"]
    for record in records:
        for item in record["series"]:
            path = (cache / item["path"]).resolve()
            if not path.is_relative_to(cache) or sha(path) != item["sha256"]:
                raise ValueError("Prepared acquisition hash/path mismatch")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("Training requires BF16 CUDA support or CPU")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = NativeKnee(load_generic_dino(generic_dino), Config(chunk=16)).to(device)
    heads = [p for name, p in model.named_parameters() if not name.startswith("dino.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.dino.parameters(), "lr": config.encoder_lr},
            {"params": heads, "lr": config.head_lr},
        ],
        weight_decay=0.01,
    )
    output.mkdir(parents=True, exist_ok=False)
    model.dino.config.to_json_file(output / "dino_config.json")
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    history, best, best_loss, stale, stale_loss = [], -float("inf"), float("inf"), 0, 0
    for epoch in range(1, config.epochs + 1):
        joint = epoch > config.warmup_epochs
        model.train()
        model.set_phase(joint)
        optimizer.param_groups[0]["lr"] = config.encoder_lr * min(
            1.0, max(0.0, (epoch - config.warmup_epochs) / 2)
        )
        order = list(range(len(train)))
        random.shuffle(order)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, index in enumerate(order):
            images, y = load_study(cache, train[index])
            group_start = (step // config.accumulation) * config.accumulation
            group_size = min(config.accumulation, len(order) - group_start)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                loss = F.binary_cross_entropy_with_logits(
                    model(images).float(), y.to(device)
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            (loss / group_size).backward()
            losses.append(float(loss.detach()))
            if (step + 1) % config.accumulation == 0 or step + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        report, _ = evaluate_classifier(model, dev, cache, device)
        improved = report["macro_auc"] > best + 1e-5
        if improved:
            best, stale = report["macro_auc"], 0
        elif joint:
            stale += 1
        if report["soft_bce"] < best_loss - 1e-5:
            best_loss, stale_loss = report["soft_bce"], 0
        elif joint:
            stale_loss += 1
        result = dict(epoch=epoch, train_bce=float(np.mean(losses)), **report)
        history.append(result)
        (output / "metrics.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        checkpoint = dict(
            identity={"architecture": model.architecture()},
            model=model.state_dict(),
            epoch=epoch,
        )
        torch.save(checkpoint, output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        print(result, flush=True)
        if joint and max(stale, stale_loss) >= config.patience:
            break
    return history
