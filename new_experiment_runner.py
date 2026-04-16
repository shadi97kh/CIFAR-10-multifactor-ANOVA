#!/usr/bin/env python3
"""
CIFAR-10 Multifactor ANOVA Experiment — Project Update 1

Design: 3 × 3 × 2 full factorial = 18 treatment combinations × 3 reps = 54 rows
        27 trained models (9 configs × 3 seeds), each evaluated on 2 test datasets

Factor A – Augmentation:    None, Basic (flip+rotate), Advanced (+colorjitter+cutout)
Factor C – Optimizer:       SGD, Adam, AdamW
Factor C – Test Dataset:    Clean, Adversarial (FGSM ε=8/255)
Block    – GPU/Computer:    recorded via --computer-id flag

Response variable: classification accuracy (%)

Usage:
    python new_experiment_runner.py --computer-id computer1
    python new_experiment_runner.py --computer-id computer2 --resume
    python new_experiment_runner.py --dry-run

Estimated total time (27 training runs):
    GPU (T4/RTX):  ~20-30 minutes
    CPU:           ~1.5-2 hours
"""

import argparse
import csv
import itertools
import os
import random
import socket
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ── Config ────────────────────────────────────────────────────────────────────

AUGMENTATIONS = ["none", "basic", "advanced"]
OPTIMIZERS    = ["sgd", "adam", "adamw"]
SEEDS         = [42, 123]
NUM_EPOCHS    = 10
BATCH_SIZE    = 128
LR            = 0.001
DATA_DIR      = "./data"
RESULTS_CSV   = "main_results.csv"
EPSILON       = 8.0 / 255.0   # FGSM perturbation strength for adversarial evaluation

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ── Cutout ────────────────────────────────────────────────────────────────────

class Cutout:
    """Randomly masks a square patch of the image (used in Advanced augmentation)."""
    def __init__(self, length=8):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones_like(img)
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        mask[:, y1:y2, x1:x2] = 0.0
        return img * mask


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transform(augmentation):
    """Training transforms for 32×32 CIFAR-10 images."""
    ops = []
    if augmentation in ("basic", "advanced"):
        ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.RandomRotation(10))
    if augmentation == "advanced":
        ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    if augmentation == "advanced":
        ops.append(Cutout(length=4))   # fixed 32×32, so length=max(4, 32//8)=4
    return transforms.Compose(ops)


def get_test_transform():
    """Test transforms — normalization only, no augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ── Simple CNN (~90K parameters) ─────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(name, params):
    if name == "sgd":
        return optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
    elif name == "adam":
        return optim.Adam(params, lr=LR, weight_decay=1e-4)
    elif name == "adamw":
        return optim.AdamW(params, lr=LR, weight_decay=1e-4)


# ── Adversarial attack (evaluation only) ─────────────────────────────────────

def fgsm_attack(model, images, labels, criterion, epsilon):
    """Single-step FGSM perturbation for evaluation."""
    images_adv = images.clone().detach().requires_grad_(True)
    outputs = model(images_adv)
    loss = criterion(outputs, labels)
    loss.backward()
    perturbation = epsilon * images_adv.grad.sign()
    return (images + perturbation).detach()


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_clean(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def evaluate_adversarial(model, loader, criterion, device, epsilon):
    """Evaluate model accuracy under FGSM attack."""
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs_adv = fgsm_attack(model, imgs, labels, criterion, epsilon)
        with torch.no_grad():
            preds = model(imgs_adv).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(augmentation, optimizer_name, seed, computer_id, device):
    """
    Train one model and evaluate on both clean and adversarial test sets.
    Returns a list of 2 row dicts (one per test_dataset_type).
    """
    set_seed(seed)

    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                  transform=get_train_transform(augmentation))
    test_set  = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                  transform=get_test_transform())

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=True)

    model     = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters())

    # Training
    epoch_times = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        t0 = time.perf_counter()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        epoch_times.append(time.perf_counter() - t0)

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate_clean(model, test_loader, device)
            print(f"    Epoch {epoch:>2d}/{NUM_EPOCHS} | Clean acc: {acc:.2f}% "
                  f"| Time: {epoch_times[-1]:.1f}s")

    # Evaluation on both datasets
    clean_acc = evaluate_clean(model, test_loader, device)
    adv_acc   = evaluate_adversarial(model, test_loader, criterion, device, EPSILON)

    avg_time   = round(np.mean(epoch_times), 2)
    total_time = round(sum(epoch_times), 2)
    ts         = datetime.now().isoformat()
    base_id    = f"{augmentation}_{optimizer_name}_s{seed}"

    return [
        {
            "run_id":            f"{base_id}_clean",
            "augmentation":      augmentation,
            "optimizer":         optimizer_name,
            "seed":              seed,
            "test_dataset_type": "clean",
            "accuracy":          round(clean_acc, 2),
            "computer_id":       computer_id,
            "avg_epoch_time":    avg_time,
            "total_time":        total_time,
            "timestamp":         ts,
        },
        {
            "run_id":            f"{base_id}_adversarial",
            "augmentation":      augmentation,
            "optimizer":         optimizer_name,
            "seed":              seed,
            "test_dataset_type": "adversarial",
            "accuracy":          round(adv_acc, 2),
            "computer_id":       computer_id,
            "avg_epoch_time":    avg_time,
            "total_time":        total_time,
            "timestamp":         ts,
        },
    ]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 3×3×2 ANOVA experiment runner")
    parser.add_argument("--computer-id", default=socket.gethostname(),
                        help="Identifier for this machine (blocked factor). "
                             "Default: hostname.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip training configurations already in the CSV.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the randomized run schedule and exit.")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    # 27 unique training configurations (aug × opt × seed)
    training_combos = list(itertools.product(AUGMENTATIONS, OPTIMIZERS, SEEDS))
    random.seed(0)   # fixed meta-seed for reproducible run order
    random.shuffle(training_combos)

    print(f"Device:      {device}")
    print(f"Computer ID: {args.computer_id}")
    print(f"Training runs: {len(training_combos)} × 2 evaluations = "
          f"{len(training_combos) * 2} CSV rows")
    print(f"Epochs: {NUM_EPOCHS} | LR: {LR} | Batch: {BATCH_SIZE} | "
          f"FGSM ε: {EPSILON:.4f} ({EPSILON*255:.0f}/255)")

    if args.dry_run:
        for i, (aug, opt, seed) in enumerate(training_combos, 1):
            print(f"  Run {i:>2d}: aug={aug}, opt={opt}, seed={seed}")
        return

    # Resume: skip already-completed (aug, opt, seed) combos
    done = set()
    if args.resume and os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                done.add((row["augmentation"], row["optimizer"], int(row["seed"])))
        # Each training run produces 2 rows; count unique training combos done
        unique_done = len(done)
        print(f"Resuming — {unique_done} training runs already recorded")

    write_header = not (args.resume and os.path.exists(RESULTS_CSV))
    fout = open(RESULTS_CSV, "a" if args.resume else "w", newline="")
    fieldnames = [
        "run_id", "augmentation", "optimizer", "seed",
        "test_dataset_type", "accuracy", "computer_id",
        "avg_epoch_time", "total_time", "timestamp",
    ]
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for i, (aug, opt, seed) in enumerate(training_combos, 1):
        if (aug, opt, seed) in done:
            print(f"[{i}/{len(training_combos)}] SKIP aug={aug} opt={opt} seed={seed}")
            continue

        print(f"\n[{i}/{len(training_combos)}] aug={aug}  opt={opt}  seed={seed}")
        rows = run_single(aug, opt, seed, args.computer_id, device)

        for row in rows:
            writer.writerow(row)
        fout.flush()

        clean_acc = rows[0]["accuracy"]
        adv_acc   = rows[1]["accuracy"]
        print(f"  -> clean={clean_acc:.2f}%  adversarial={adv_acc:.2f}%  "
              f"time={rows[0]['total_time']}s")

    fout.close()
    print(f"\nDone. Results saved to {RESULTS_CSV}")
    print(f"Total rows: {len(training_combos) * 2} expected "
          f"(27 models × 2 test dataset types)")


if __name__ == "__main__":
    main()
