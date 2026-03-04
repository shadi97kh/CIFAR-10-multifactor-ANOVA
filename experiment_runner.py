#!/usr/bin/env python3
"""
CIFAR-10 Multifactor ANOVA Experiment (Simplified for class project)

Design: 3 × 3 × 3 full factorial = 27 treatments × 3 reps = 81 runs
Model:  Simple 3-block CNN (~90K params) — trains in ~30s/run on GPU

Factor A – Resolution:   32×32 (native), 64×64, 96×96
Factor B – Augmentation: None, Basic (flip+rotate), Advanced (+colorjitter+cutout)
Factor C – Optimizer:    SGD, Adam, AdamW

Estimated total time:
  GPU (T4/RTX):  ~40 minutes
  CPU:           ~3-4 hours
"""

import argparse
import csv
import itertools
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ── Config ───────────────────────────────────────────────────────────────────

RESOLUTIONS = [32, 64, 96]
AUGMENTATIONS = ["none", "basic", "advanced"]
OPTIMIZERS = ["sgd", "adam", "adamw"]
SEEDS = [42, 123, 7]
NUM_EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001
DATA_DIR = "./data"
RESULTS_CSV = "experiment_results.csv"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ── Cutout ───────────────────────────────────────────────────────────────────

class Cutout:
    def __init__(self, length=8):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones_like(img)
        y, x = random.randint(0, h - 1), random.randint(0, w - 1)
        y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
        x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
        mask[:, y1:y2, x1:x2] = 0.0
        return img * mask


# ── Transforms ───────────────────────────────────────────────────────────────

def get_train_transform(resolution, augmentation):
    ops = []
    if resolution != 32:
        ops.append(transforms.Resize((resolution, resolution)))

    if augmentation in ("basic", "advanced"):
        ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.RandomRotation(10))
    if augmentation == "advanced":
        ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))

    if augmentation == "advanced":
        ops.append(Cutout(length=max(4, resolution // 8)))

    return transforms.Compose(ops)


def get_test_transform(resolution):
    ops = []
    if resolution != 32:
        ops.append(transforms.Resize((resolution, resolution)))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return transforms.Compose(ops)


# ── Simple CNN (~90K parameters) ─────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    3-block CNN: each block is Conv → BN → ReLU → MaxPool.
    Adaptive pooling makes it resolution-agnostic.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 64 → 128
            nn.Conv2d(128, 128, 3, padding=1) if False else
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


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── Single run ───────────────────────────────────────────────────────────────

def run_single(resolution, augmentation, optimizer_name, seed, device):
    set_seed(seed)

    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                  transform=get_train_transform(resolution, augmentation))
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                 transform=get_test_transform(resolution))

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters())

    # Train
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

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        "test_accuracy": round(100.0 * correct / total, 2),
        "avg_epoch_time": round(np.mean(epoch_times), 2),
        "total_time": round(sum(epoch_times), 2),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if hasattr(torch.backends, "mps")
                          and torch.backends.mps.is_available() else "cpu")

    # Build randomized run order
    schedule = list(itertools.product(RESOLUTIONS, AUGMENTATIONS, OPTIMIZERS, SEEDS))
    random.seed(0)
    random.shuffle(schedule)

    print(f"Device: {device} | Runs: {len(schedule)} | Epochs: {NUM_EPOCHS}")

    if args.dry_run:
        for i, (r, a, o, s) in enumerate(schedule, 1):
            print(f"  Run {i:>2d}: res={r}, aug={a}, opt={o}, seed={s}")
        return

    # Resume support
    done = set()
    if args.resume and os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                done.add((int(row["resolution"]), row["augmentation"],
                          row["optimizer"], int(row["seed"])))
        print(f"Resuming — {len(done)} runs already done")

    write_header = not (args.resume and os.path.exists(RESULTS_CSV))
    fout = open(RESULTS_CSV, "a" if args.resume else "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=[
        "run_id", "resolution", "augmentation", "optimizer", "seed",
        "test_accuracy", "avg_epoch_time", "total_time", "timestamp",
    ])
    if write_header:
        writer.writeheader()

    for i, (res, aug, opt, seed) in enumerate(schedule, 1):
        if (res, aug, opt, seed) in done:
            continue

        print(f"\n[{i}/{len(schedule)}] res={res} aug={aug} opt={opt} seed={seed}")
        result = run_single(res, aug, opt, seed, device)
        print(f"  → accuracy={result['test_accuracy']}%  time={result['total_time']}s")

        writer.writerow({
            "run_id": f"R{res}_{aug}_{opt}_s{seed}",
            "resolution": res,
            "augmentation": aug,
            "optimizer": opt,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            **result,
        })
        fout.flush()

    fout.close()
    print(f"\nDone. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
