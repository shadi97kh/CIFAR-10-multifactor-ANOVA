# CIFAR-10 Multifactor ANOVA

ESI 6247 – Industrial Experimentation (Spring 2026)
Varvara Vythoulka, Zahra Khodagholi

A multifactor ANOVA experiment on CIFAR-10 examining how **data augmentation**, **optimizer choice**, and **test dataset type** (clean vs. adversarial) affect CNN classification accuracy.

---

## Design

**3 × 3 × 2 full factorial** with 3 replications = **54 observations** (27 trained models × 2 test datasets)

| Factor | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| **A: Augmentation** | None | Basic (flip + rotate) | Advanced (+ colorjitter + cutout) |
| **B: Optimizer** | SGD | Adam | AdamW |
| **C: Test Dataset** | Clean | Adversarial (FGSM ε = 8/255) | — |

**Block:** GPU / Computer (recorded per run via `--computer-id`)
**Response:** Classification accuracy (%)
**Model:** 3-block CNN (~90K params), fixed 32×32, 10 epochs, lr = 0.001, batch = 128
**Seeds:** 42, 123, 7

---

## Files

| File | Purpose |
|---|---|
| `new_experiment_runner.py` | Trains 27 models, evaluates each on clean + adversarial, writes `main_results.csv` |
| `new_analysis.py` | Descriptive stats, assumption checks, 3-way ANOVA with block, Tukey HSD, plots |

---

## Setup

```bash
git clone https://github.com/shadi97kh/CIFAR-10-multifactor-ANOVA.git
cd CIFAR-10-multifactor-ANOVA

conda create -n anova_exp python=3.11 -y
conda activate anova_exp
pip install torch torchvision pandas numpy scipy statsmodels matplotlib seaborn
```

---

## Running the Experiment

The team splits the 27 training runs by seed. Each member runs the full set of 9 configurations (3 augmentations × 3 optimizers) for their assigned seed(s).

### Step 1 — Edit `SEEDS` in `new_experiment_runner.py` (line 43)

```python
# Zahra's machine (UCF Newton HPC):
SEEDS = [42, 123]

# Varvara's machine:
SEEDS = [7]
```

### Step 2 — Run with your block label

```bash
# Zahra's machine:
python new_experiment_runner.py --computer-id computer1

# Varvara's machine:
python new_experiment_runner.py --computer-id computer2
```

### Step 3 — Resume if interrupted

```bash
python new_experiment_runner.py --computer-id computer1 --resume
```

### Dry run (preview schedule)

```bash
python new_experiment_runner.py --dry-run
```

---

## Merging Results

After both members finish, combine their CSVs:

```bash
# Rename each file
mv main_results.csv results_computer1.csv
# (Varvara uploads hers as results_computer2.csv)

# Concatenate
head -1 results_computer1.csv > combined_results.csv
tail -n +2 results_computer1.csv >> combined_results.csv
tail -n +2 results_computer2.csv >> combined_results.csv
```

---

## Running the Analysis

```bash
python new_analysis.py --input combined_results.csv
```

Outputs:

```
main_results/
├── anova_table.csv           # Type II SS, F, p, η², partial η²
├── descriptive_stats.csv     # cell means, SEM, 95% CI
├── tukey_augmentation.csv
├── tukey_optimizer.csv
├── tukey_test_dataset_type.csv
└── summary.txt               # full text report

main_plots/
├── main_effects.png
├── interaction_AB.png
├── interaction_AC.png
├── interaction_BC.png
├── boxplots.png
├── heatmap.png
├── clean_vs_adversarial.png
├── residual_qq.png
├── residual_histogram.png
└── residual_vs_fitted.png
```

The analysis automatically includes `computer_id` as a fixed block term in the ANOVA model when more than one computer is present.

---

## CSV Schema (`main_results.csv`)

| Column | Description |
|---|---|
| `run_id` | Unique identifier, e.g. `basic_adam_s42_clean` |
| `augmentation` | `none` / `basic` / `advanced` |
| `optimizer` | `sgd` / `adam` / `adamw` |
| `seed` | `42`, `123`, or `7` |
| `test_dataset_type` | `clean` / `adversarial` |
| `accuracy` | Classification accuracy (%) |
| `computer_id` | Block label (machine that ran the training) |
| `avg_epoch_time` | Seconds |
| `total_time` | Seconds |
| `timestamp` | ISO-8601 timestamp |

---

## Google Colab

```python
!pip install torch torchvision pandas scipy statsmodels matplotlib seaborn
!python new_experiment_runner.py --computer-id colab
!python new_analysis.py --input main_results.csv
```

---

## Requirements

```
torch, torchvision, pandas, numpy, scipy, statsmodels, matplotlib, seaborn
```
