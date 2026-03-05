# CIFAR-10 Multifactor ANOVA Experiments

Two multifactor ANOVA experiments on CIFAR-10 examining how CNN training configuration choices affect classification accuracy and adversarial robustness.

---

## Experiment 1: Effects of Resolution, Augmentation, and Optimizer on Accuracy

### Design

3 x 3 x 3 full factorial, 3 replications = 81 runs

| Factor | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| **A: Resolution** | 32x32 | 64x64 | 96x96 |
| **B: Augmentation** | None | Basic (flip+rotate) | Advanced (+colorjitter+cutout) |
| **C: Optimizer** | SGD | Adam | AdamW |

**Response:** Test accuracy (%). **Model:** 3-block CNN (~90K params), 10 epochs.

### Scripts

```bash
python experiment_runner.py            # run all 81 conditions
python experiment_runner.py --resume   # resume after interruption
python analysis.py                     # full ANOVA analysis
```

### Results

Grand mean: **73.38%** | R-squared = 0.9803

| Source | F | p-value | Partial eta-sq | Sig |
|---|---|---|---|---|
| Resolution (A) | 178.59 | 1.57e-24 | 0.8687 | Yes |
| Augmentation (B) | 24.57 | 2.58e-08 | 0.4765 | Yes |
| Optimizer (C) | 1080.94 | 2.79e-44 | 0.9756 | Yes |
| A x B | 2.96 | 2.77e-02 | 0.1799 | Yes |
| A x C | 21.96 | 8.42e-11 | 0.6193 | Yes |
| B x C | 4.40 | 3.77e-03 | 0.2457 | Yes |
| A x B x C | 0.80 | 6.09e-01 | 0.1055 | No |

**Key findings:**

- **Optimizer is the dominant factor** (partial eta-sq = 0.976). Adam (76.76%) and AdamW (76.86%) are statistically indistinguishable (Tukey p = 0.99) but both outperform SGD (66.52%) by ~10 percentage points.
- **Resolution goes the "wrong" direction:** 32x32 (native) performs best (75.91%), accuracy drops with upscaling. The simple CNN gains nothing from interpolated extra pixels.
- **Augmentation is significant in the ANOVA** (p < 0.001) but Tukey finds no significant pairwise differences. With only 10 epochs, augmentation acts as noise rather than regularization.
- **Resolution x Optimizer interaction** (partial eta-sq = 0.62): SGD degrades much more steeply with resolution than Adam/AdamW.
- Best: 32, none, Adam = **78.93%** | Worst: 96, advanced, SGD = **60.72%**

---

## Experiment 2: Effects of Resolution, Adversarial Training, and Optimizer on Robustness

### Design

3 x 3 x 3 full factorial, 3 replications = 81 runs

| Factor | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| **A: Resolution** | 32x32 | 64x64 | 96x96 |
| **B: Training Method** | Standard | FGSM Adversarial | PGD Adversarial |
| **C: Optimizer** | SGD | Adam | AdamW |

**Responses:** Clean accuracy (%), adversarial accuracy (% under FGSM at epsilon=8/255), robustness gap (clean minus adversarial).

**Adversarial parameters:** FGSM epsilon=8/255 single-step; PGD epsilon=8/255, 7 steps, alpha=2/255, random start. Both use 50/50 clean/adversarial batch mixing. Evaluation: FGSM at epsilon=8/255.

### Scripts

```bash
python adversarial_experiment_runner.py            # run all 81 conditions
python adversarial_experiment_runner.py --resume   # resume after interruption
python adversarial_analysis.py                     # full ANOVA on all 3 responses
```

### Results: Clean Accuracy

Grand mean: **70.72%** | R-squared = 0.9819

| Source | F | p-value | Partial eta-sq | Sig |
|---|---|---|---|---|
| Resolution (A) | 244.81 | 8.35e-28 | 0.9007 | Yes |
| Training Method (B) | 207.18 | 4.67e-26 | 0.8847 | Yes |
| Optimizer (C) | 959.30 | 6.44e-43 | 0.9726 | Yes |
| A x B | 3.86 | 7.90e-03 | 0.2223 | Yes |
| A x C | 1.71 | 1.61e-01 | 0.1126 | No |
| B x C | 18.36 | 1.41e-09 | 0.5763 | Yes |
| A x B x C | 0.69 | 6.97e-01 | 0.0930 | No |

**Key findings:**

- **Optimizer still dominates clean accuracy** (partial eta-sq = 0.97). Adam/AdamW at ~74% vs SGD at ~63.6%.
- **Adversarial training costs ~5% clean accuracy:** Standard training gets 74.03% vs FGSM 69.16% and PGD 68.98%.
- **FGSM and PGD are indistinguishable on clean accuracy** (Tukey p = 0.99).
- **Training Method x Optimizer interaction** (partial eta-sq = 0.58): SGD drops much harder under adversarial training (from 69% to ~60%) while Adam/AdamW stay relatively stable (~76% to ~73%).
- Best: 32, standard, Adam = **79.35%** | Worst: 96, PGD, SGD = **57.70%**

### Results: Adversarial Accuracy

Grand mean: **41.61%** | R-squared = 0.9976

| Source | F | p-value | Partial eta-sq | Sig |
|---|---|---|---|---|
| Resolution (A) | 504.38 | 1.15e-35 | 0.9492 | Yes |
| Training Method (B) | 10051.09 | 3.59e-70 | 0.9973 | Yes |
| Optimizer (C) | 367.38 | 3.61e-32 | 0.9315 | Yes |
| A x B | 40.84 | 9.98e-16 | 0.7516 | Yes |
| A x C | 18.82 | 9.71e-10 | 0.5823 | Yes |
| B x C | 116.41 | 7.12e-26 | 0.8961 | Yes |
| A x B x C | 5.53 | 4.05e-05 | 0.4504 | Yes |

**Key findings:**

- **Training Method is overwhelmingly dominant** (partial eta-sq = 0.997, F = 10,051). Standard training gives only ~21% adversarial accuracy. FGSM and PGD both reach ~52%, and they are indistinguishable (Tukey p = 0.96).
- **All seven effects are significant**, including the three-way interaction. This is the only analysis where this occurred.
- **Training Method x Optimizer interaction** (partial eta-sq = 0.90): All optimizers converge under adversarial training, but SGD still lags by ~9 points. Under standard training, all optimizers are equally vulnerable (~20%).
- **Resolution x Training Method interaction** (partial eta-sq = 0.75): Standard-trained models collapse from 28% at 32x32 to 17% at 64x64 under attack. Adversarially trained models are more stable across resolutions.
- Best: 32, FGSM, AdamW = **58.87%** | Worst: 96, standard, Adam = **15.12%**

### Results: Robustness Gap

Grand mean: **29.11%** | R-squared = 0.9964

| Source | F | p-value | Partial eta-sq | Sig |
|---|---|---|---|---|
| Resolution (A) | 23.10 | 5.63e-08 | 0.4611 | Yes |
| Training Method (B) | 7101.19 | 4.13e-66 | 0.9962 | Yes |
| Optimizer (C) | 124.66 | 5.80e-21 | 0.8220 | Yes |
| A x B | 38.24 | 3.69e-15 | 0.7391 | Yes |
| A x C | 15.25 | 2.10e-08 | 0.5304 | Yes |
| B x C | 19.24 | 6.88e-10 | 0.5877 | Yes |
| A x B x C | 2.20 | 4.16e-02 | 0.2459 | Yes |

**Key findings:**

- **Standard training has a 53% robustness gap** (clean minus adversarial). FGSM and PGD compress it to ~17%.
- **All seven effects significant** including the three-way interaction (p = 0.042).
- **Resolution x Training Method interaction** (partial eta-sq = 0.74): The gap for standard training increases with resolution (48% at 32x32 to 56% at 64x64), meaning upscaling makes standard models even more vulnerable.

### Practical Takeaways

1. **Adversarial training is the single most important factor for robustness** — it boosts adversarial accuracy by ~30 percentage points at a cost of only ~5% clean accuracy.
2. **FGSM and PGD training produce nearly identical results** (Tukey p > 0.96 across all three responses) — the cheaper single-step FGSM is sufficient, no need for expensive 7-step PGD.
3. **Adam/AdamW outperform SGD for both clean and adversarial accuracy**, and the gap widens under adversarial training (B x C interaction).
4. **Higher resolution hurts across the board** — the simple CNN does not benefit from upscaling, and it actually increases adversarial vulnerability for standard-trained models.
5. **Interactions matter** — the three-way interaction is significant for adversarial accuracy and robustness gap, meaning the optimal configuration depends on all three factors jointly.

---

## Requirements

```bash
pip install torch torchvision pandas numpy scipy statsmodels matplotlib seaborn
```

## Output Files

```
# Experiment 1 (Augmentation)
experiment_results.csv
results/
plots/

# Experiment 2 (Adversarial)
adversarial_experiment_results.csv
adversarial_results/
adversarial_plots/
```

## Google Colab

```python
!pip install torch torchvision pandas scipy statsmodels matplotlib seaborn

# Experiment 1
!python experiment_runner.py
!python analysis.py

# Experiment 2
!python adversarial_experiment_runner.py
!python adversarial_analysis.py
```
