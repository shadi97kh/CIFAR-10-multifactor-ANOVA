#!/usr/bin/env python3
"""
CIFAR-10 Multifactor ANOVA Experiment — Adversarial Training Edition

Design: 3 × 3 × 3 full factorial = 27 treatments × 3 reps = 81 runs
Model:  Simple 3-block CNN (~90K params)

Factor A – Resolution:       32×32 (native), 64×64, 96×96
Factor B – Training Method:  Standard, FGSM Adversarial (ε=8/255), PGD Adversarial (ε=8/255, 7 steps)
Factor C – Optimizer:        SGD, Adam, AdamW

Response variables:
  - Clean test accuracy (%)
  - Adversarial test accuracy (%) under FGSM attack (ε=8/255)
  - Robustness gap (clean − adversarial)

Estimated total time:
  GPU (T4/RTX):  ~1.5-2 hours
  CPU:           ~8-10 hours
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
TRAINING_METHODS = ["standard", "fgsm", "pgd"]
OPTIMIZERS = ["sgd", "adam", "adamw"]
SEEDS = [42, 123, 7]
NUM_EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001
DATA_DIR = "./data"
RESULTS_CSV = "adversarial_experiment_results.csv"

# Adversarial parameters
EPSILON = 8.0 / 255.0
PGD_STEPS = 7
PGD_ALPHA = 2.0 / 255.0
EVAL_EPSILON = 8.0 / 255.0

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ── Transforms ───────────────────────────────────────────────────────────────

def get_train_transform(resolution):
    ops = []
    if resolution != 32:
        ops.append(transforms.Resize((resolution, resolution)))
    ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
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


# ── Adversarial attacks ──────────────────────────────────────────────────────

def fgsm_attack(model, images, labels, criterion, epsilon):
    images_adv = images.clone().detach().requires_grad_(True)
    outputs = model(images_adv)
    loss = criterion(outputs, labels)
    loss.backward()
    perturbation = epsilon * images_adv.grad.sign()
    images_adv = (images + perturbation).detach()
    return images_adv


def pgd_attack(model, images, labels, criterion, epsilon, alpha, steps):
    images_adv = images.clone().detach()
    images_adv = images_adv + torch.empty_like(images_adv).uniform_(-epsilon, epsilon)
    images_adv = images_adv.detach()

    for _ in range(steps):
        images_adv.requires_grad_(True)
        outputs = model(images_adv)
        loss = criterion(outputs, labels)
        loss.backward()
        grad_sign = images_adv.grad.sign()
        images_adv = (images_adv.detach() + alpha * grad_sign).detach()
        delta = torch.clamp(images_adv - images, min=-epsilon, max=epsilon)
        images_adv = (images + delta).detach()

    return images_adv


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


# ── Training functions ───────────────────────────────────────────────────────

def train_standard(model, loader, criterion, optimizer, device):
    model.train()
    t0 = time.perf_counter()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
    return time.perf_counter() - t0


def train_fgsm_adversarial(model, loader, criterion, optimizer, device):
    model.train()
    t0 = time.perf_counter()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        model.eval()
        imgs_adv = fgsm_attack(model, imgs, labels, criterion, EPSILON)
        model.train()
        half = imgs.size(0) // 2
        mixed_imgs = torch.cat([imgs[:half], imgs_adv[half:]], dim=0)
        mixed_labels = torch.cat([labels[:half], labels[half:]], dim=0)
        optimizer.zero_grad()
        loss = criterion(model(mixed_imgs), mixed_labels)
        loss.backward()
        optimizer.step()
    return time.perf_counter() - t0


def train_pgd_adversarial(model, loader, criterion, optimizer, device):
    model.train()
    t0 = time.perf_counter()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        model.eval()
        imgs_adv = pgd_attack(model, imgs, labels, criterion,
                               EPSILON, PGD_ALPHA, PGD_STEPS)
        model.train()
        half = imgs.size(0) // 2
        mixed_imgs = torch.cat([imgs[:half], imgs_adv[half:]], dim=0)
        mixed_labels = torch.cat([labels[:half], labels[half:]], dim=0)
        optimizer.zero_grad()
        loss = criterion(model(mixed_imgs), mixed_labels)
        loss.backward()
        optimizer.step()
    return time.perf_counter() - t0


# ── Evaluation ───────────────────────────────────────────────────────────────

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


# ── Single run ───────────────────────────────────────────────────────────────

def run_single(resolution, training_method, optimizer_name, seed, device):
    set_seed(seed)

    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                  transform=get_train_transform(resolution))
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                 transform=get_test_transform(resolution))

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters())

    train_fn = {
        "standard": train_standard,
        "fgsm": train_fgsm_adversarial,
        "pgd": train_pgd_adversarial,
    }[training_method]

    epoch_times = []
    for epoch in range(1, NUM_EPOCHS + 1):
        t = train_fn(model, train_loader, criterion, optimizer, device)
        epoch_times.append(t)
        if epoch % 5 == 0 or epoch == 1:
            clean_acc = evaluate_clean(model, test_loader, device)
            print(f"    Epoch {epoch:>2d}/{NUM_EPOCHS} | "
                  f"Clean acc: {clean_acc:.2f}% | Time: {t:.1f}s")

    clean_acc = evaluate_clean(model, test_loader, device)
    adv_acc = evaluate_adversarial(model, test_loader, criterion,
                                    device, EVAL_EPSILON)

    return {
        "clean_accuracy": round(clean_acc, 2),
        "adversarial_accuracy": round(adv_acc, 2),
        "robustness_gap": round(clean_acc - adv_acc, 2),
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

    schedule = list(itertools.product(
        RESOLUTIONS, TRAINING_METHODS, OPTIMIZERS, SEEDS))
    random.seed(0)
    random.shuffle(schedule)

    print(f"Device: {device} | Runs: {len(schedule)} | Epochs: {NUM_EPOCHS}")
    print(f"Adversarial epsilon = {EPSILON:.4f} ({EPSILON*255:.0f}/255)")
    print(f"PGD steps = {PGD_STEPS}, alpha = {PGD_ALPHA:.4f}")

    if args.dry_run:
        for i, (r, m, o, s) in enumerate(schedule, 1):
            print(f"  Run {i:>2d}: res={r}, method={m}, opt={o}, seed={s}")
        return

    done = set()
    if args.resume and os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                done.add((int(row["resolution"]), row["training_method"],
                          row["optimizer"], int(row["seed"])))
        print(f"Resuming — {len(done)} runs already done")

    write_header = not (args.resume and os.path.exists(RESULTS_CSV))
    fout = open(RESULTS_CSV, "a" if args.resume else "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=[
        "run_id", "resolution", "training_method", "optimizer", "seed",
        "clean_accuracy", "adversarial_accuracy", "robustness_gap",
        "avg_epoch_time", "total_time", "timestamp",
    ])
    if write_header:
        writer.writeheader()

    for i, (res, method, opt, seed) in enumerate(schedule, 1):
        if (res, method, opt, seed) in done:
            continue

        print(f"\n[{i}/{len(schedule)}] res={res} method={method} opt={opt} seed={seed}")
        result = run_single(res, method, opt, seed, device)
        print(f"  -> clean={result['clean_accuracy']}%  "
              f"adv={result['adversarial_accuracy']}%  "
              f"gap={result['robustness_gap']}%  "
              f"time={result['total_time']}s")

        writer.writerow({
            "run_id": f"R{res}_{method}_{opt}_s{seed}",
            "resolution": res,
            "training_method": method,
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
