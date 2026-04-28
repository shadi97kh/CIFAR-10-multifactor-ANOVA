#!/usr/bin/env python3
"""
Power analysis for the 3×3×2 factorial ANOVA.

Computes the number of replications per cell (= number of seeds) needed
to achieve a target power for each main effect and interaction.

Method
------
For a balanced factorial ANOVA, the noncentrality parameter for the
F-test of a given effect is:

    lambda = N * f^2

where N is the total sample size and f is Cohen's f (effect size).
Cohen's f relates to partial eta-squared as:

    f = sqrt(eta_p^2 / (1 - eta_p^2))

Cohen's conventional benchmarks:
    small   f = 0.10   (eta_p^2 ~ 0.01)
    medium  f = 0.25   (eta_p^2 ~ 0.06)
    large   f = 0.40   (eta_p^2 ~ 0.14)

Usage
-----
    python power_analysis.py                       # reference table only
    python power_analysis.py --input results.csv   # also use observed effects
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# ── Design constants ─────────────────────────────────────────────────────────
A, B, C = 3, 3, 2          # levels: augmentation, optimizer, test dataset
CELLS   = A * B * C        # 18 treatment combinations
ALPHA   = 0.05
TARGET  = 0.80

EFFECTS = {
    "A: Augmentation":          A - 1,            # df = 2
    "B: Optimizer":             B - 1,            # df = 2
    "C: Test Dataset":          C - 1,            # df = 1
    "A x B":                    (A - 1) * (B - 1),         # df = 4
    "A x C":                    (A - 1) * (C - 1),         # df = 2
    "B x C":                    (B - 1) * (C - 1),         # df = 2
    "A x B x C":                (A - 1) * (B - 1) * (C - 1), # df = 4
}

REFERENCE_F = {"small (f=0.10)": 0.10,
               "medium (f=0.25)": 0.25,
               "large (f=0.40)": 0.40}


# ── Power calculation ────────────────────────────────────────────────────────

def power_at_n(cohen_f, df_num, n, alpha=ALPHA):
    """Compute power for an F-test with n reps per cell."""
    N        = CELLS * n
    df_denom = N - CELLS         # full factorial: residual df = N - (cells)
    if df_denom <= 0:
        return 0.0
    nc       = N * cohen_f ** 2
    f_crit   = stats.f.ppf(1 - alpha, df_num, df_denom)
    return 1.0 - stats.ncf.cdf(f_crit, df_num, df_denom, nc)


def required_n(cohen_f, df_num, target=TARGET, alpha=ALPHA, n_max=200):
    """Smallest n per cell that achieves >= target power."""
    if cohen_f <= 0:
        return None
    for n in range(2, n_max + 1):
        if power_at_n(cohen_f, df_num, n, alpha) >= target:
            return n
    return None


def f_from_partial_eta_sq(eta_p2):
    return np.sqrt(eta_p2 / (1 - eta_p2)) if 0 <= eta_p2 < 1 else np.nan


# ── Reference table (no data needed) ─────────────────────────────────────────

def reference_table():
    print("\n" + "=" * 78)
    print("REQUIRED REPLICATIONS PER CELL (n) FOR 80% POWER, alpha = 0.05")
    print("Reference: Cohen's conventional effect sizes")
    print("=" * 78)

    header = f"{'Effect':<20s} {'df_num':>7s}  " + "  ".join(
        f"{label:>16s}" for label in REFERENCE_F)
    print(header)
    print("-" * len(header))

    for effect_name, df_num in EFFECTS.items():
        row = f"{effect_name:<20s} {df_num:>7d}  "
        cells = []
        for label, f in REFERENCE_F.items():
            n = required_n(f, df_num)
            if n is None:
                cells.append(f"{'>200':>16s}")
            else:
                N = CELLS * n
                cells.append(f"{f'n={n}  (N={N})':>16s}")
        print(row + "  ".join(cells))

    print()
    print("Interpretation:")
    print("  'n' is the number of seeds (replications) per treatment combination.")
    print("  'N' is the total number of observations across all 18 cells.")
    print("  Each training run contributes 2 observations (clean + adversarial),")
    print("  so total training runs = N / 2.")


# ── Observed effects from data ───────────────────────────────────────────────

REQUIRED_COLS = {"augmentation", "optimizer", "test_dataset_type", "accuracy"}

def observed_effects(path):
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"\nSkipping observed-effects analysis — '{path}' is missing columns: {missing}")
        print("(Expected schema: augmentation, optimizer, test_dataset_type, accuracy)")
        return

    print(f"\n" + "=" * 78)
    print(f"REQUIRED REPLICATIONS — based on observed effects in {path}")
    print(f"({len(df)} rows)")
    print("=" * 78)

    formula = ("accuracy ~ C(augmentation) + C(optimizer) + C(test_dataset_type)"
               " + C(augmentation):C(optimizer)"
               " + C(augmentation):C(test_dataset_type)"
               " + C(optimizer):C(test_dataset_type)"
               " + C(augmentation):C(optimizer):C(test_dataset_type)")
    model = ols(formula, data=df).fit()
    table = anova_lm(model, typ=2)
    ss_resid = table.loc["Residual", "sum_sq"]
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + ss_resid)

    rename = {
        "C(augmentation)":                                    "A: Augmentation",
        "C(optimizer)":                                       "B: Optimizer",
        "C(test_dataset_type)":                               "C: Test Dataset",
        "C(augmentation):C(optimizer)":                       "A x B",
        "C(augmentation):C(test_dataset_type)":               "A x C",
        "C(optimizer):C(test_dataset_type)":                  "B x C",
        "C(augmentation):C(optimizer):C(test_dataset_type)":  "A x B x C",
    }

    print(f"{'Effect':<20s} {'df':>4s} {'partial η²':>11s} {'Cohen f':>9s} "
          f"{'n needed':>10s} {'N total':>9s}")
    print("-" * 78)
    for src, row in table.iterrows():
        if src == "Residual":
            continue
        name  = rename.get(src, src)
        eta_p = row["partial_eta_sq"]
        f     = f_from_partial_eta_sq(eta_p)
        df_num = EFFECTS.get(name)
        if df_num is None:
            continue
        n = required_n(f, df_num)
        n_str = f"{n}" if n is not None else ">200"
        N_str = f"{CELLS * n}" if n is not None else "—"
        print(f"{name:<20s} {int(row['df']):>4d} {eta_p:>11.4f} {f:>9.4f} "
              f"{n_str:>10s} {N_str:>9s}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Optional CSV to estimate effect sizes from")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--power", type=float, default=TARGET)
    args = parser.parse_args()

    print(f"\nDesign:    3 × 3 × 2 full factorial ({CELLS} cells)")
    print(f"alpha = {args.alpha}     target power = {args.power}")

    reference_table()

    if args.input:
        if os.path.exists(args.input):
            observed_effects(args.input)
        else:
            print(f"\nInput file '{args.input}' not found.")


if __name__ == "__main__":
    main()
