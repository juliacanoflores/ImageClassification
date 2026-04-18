"""Single-run training function called by the W&B sweep agent.

Each run does two phases:
  1. Warmup  – backbone frozen, only the head is trained.
  2. Fine-tune – last `unfreeze_layers` backbone blocks + head, lower LR.

Run this file directly to smoke-test a single run with default config.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torchvision
import wandb
from sklearn.metrics import confusion_matrix, f1_score

from cnn import CNN, load_data

SEED = 42
TRAIN_DIR = "./dataset/training"
VALID_DIR = "./dataset/validation"
IMG_SIZE = 224
BATCH_SIZE = 32
ENTITY = "javi_paula_julia"
PROJECT = "image-classification"

MODEL_REGISTRY = {
    "EfficientNetV2-S": lambda: torchvision.models.efficientnet_v2_s(weights="DEFAULT"),
    "ConvNeXt-Small": lambda: torchvision.models.convnext_small(weights="DEFAULT"),
    "Swin-T": lambda: torchvision.models.swin_t(weights="DEFAULT"),
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _build_optimizer(param_groups, name: str, weight_decay: float):
    if name == "AdamW":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    if name == "SGD":
        return torch.optim.SGD(param_groups, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), correct / len(loader.dataset), f1, all_preds, all_labels


def train_run():
    """Entry point for W&B sweep agent."""
    run = wandb.init(entity=ENTITY, project=PROJECT)
    cfg = wandb.config
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, num_classes = load_data(
        TRAIN_DIR, VALID_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    class_names = train_loader.dataset.classes

    base = MODEL_REGISTRY[cfg.model]()
    model = CNN(base, num_classes, unfreezed_layers=0, device=device)
    criterion = nn.CrossEntropyLoss()

    run.name = (
        f"{cfg.model}_lr{cfg.lr_head:.0e}_wd{cfg.weight_decay}"
        f"_unfreeze{cfg.unfreeze_layers}_{cfg.optimizer}"
    )

    best_val_acc = 0.0
    best_state = None

    # ── Phase 1: warmup (head only) ────────────────────────────────────
    for p in model.feature_extractor.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = _build_optimizer(
        [{"params": model.classifier.parameters(), "lr": cfg.lr_head}],
        cfg.optimizer, cfg.weight_decay,
    )

    for epoch in range(cfg.warmup_epochs):
        t0 = time.time()
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, _, _ = _eval_epoch(model, valid_loader, criterion, device)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wandb.log({
            "epoch": epoch + 1, "phase": 1,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": va_loss, "val/acc": va_acc, "val/f1_macro": va_f1,
            "epoch_time_s": time.time() - t0,
        })

    # ── Phase 2: fine-tune (unfreeze last N blocks) ────────────────────
    unfreeze_n = cfg.unfreeze_layers
    for p in model.feature_extractor.parameters():
        p.requires_grad = False
    if unfreeze_n > 0:
        children = list(model.feature_extractor.children())
        for layer in children[-unfreeze_n:]:
            for p in layer.parameters():
                p.requires_grad = True

    backbone_params = [p for p in model.feature_extractor.parameters() if p.requires_grad]
    param_groups = [{"params": model.classifier.parameters(), "lr": cfg.lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.lr_backbone})

    optimizer = _build_optimizer(param_groups, cfg.optimizer, cfg.weight_decay)

    for epoch in range(cfg.finetune_epochs):
        t0 = time.time()
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, _, _ = _eval_epoch(model, valid_loader, criterion, device)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wandb.log({
            "epoch": cfg.warmup_epochs + epoch + 1, "phase": 2,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": va_loss, "val/acc": va_acc, "val/f1_macro": va_f1,
            "epoch_time_s": time.time() - t0,
        })

    # ── Final eval on best checkpoint ──────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    _, final_acc, final_f1, all_preds, all_labels = _eval_epoch(
        model, valid_loader, criterion, device
    )

    wandb.log({
        "best_val_acc": final_acc,
        "best_val_f1_macro": final_f1,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=all_labels, preds=all_preds, class_names=class_names
        ),
    })

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{cfg.model.replace(' ', '_')}_{run.id}.pt"
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact(f"best-{cfg.model.replace(' ', '_')}", type="model")
    artifact.add_file(model_path)
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    # Smoke-test with a single run using default config
    default_cfg = {
        "model": "Swin-T",
        "lr_head": 1e-3,
        "lr_backbone": 1e-5,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "unfreeze_layers": 2,
        "warmup_epochs": 3,
        "finetune_epochs": 5,
    }
    wandb.init(entity=ENTITY, project=PROJECT, config=default_cfg, name="smoke-test")
    train_run()
