#!/usr/bin/env python3
"""
ANOVA Analysis — produces everything needed for the class report.

Outputs:
  results/anova_table.csv           ANOVA table (F, p, η², partial η²)
  results/descriptive_stats.csv     Cell means, std, 95% CI
  results/tukey_resolution.csv      Tukey HSD pairwise for resolution
  results/tukey_augmentation.csv    Tukey HSD pairwise for augmentation
  results/tukey_optimizer.csv       Tukey HSD pairwise for optimizer
  results/summary.txt               Full text summary with conclusions

  plots/main_effects.png            Main effects plot
  plots/interaction_AB.png          Resolution × Augmentation
  plots/interaction_AC.png          Resolution × Optimizer
  plots/interaction_BC.png          Augmentation × Optimizer
  plots/residual_qq.png             QQ plot
  plots/residual_histogram.png      Residual distribution
  plots/residual_vs_fitted.png      Residuals vs fitted values
  plots/boxplots.png                Box plots by factor
  plots/cell_means_heatmap.png      Heatmap of all 27 cell means

Usage:
    python analysis.py
    python analysis.py --input my_results.csv
"""

import argparse, os, sys, warnings, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

PLOTS = "plots"
RESULTS = "results"


def load(path):
    df = pd.read_csv(path)
    df["resolution"] = pd.Categorical(df["resolution"].astype(str),
                                       categories=["32", "64", "96"], ordered=True)
    df["augmentation"] = pd.Categorical(df["augmentation"],
                                         categories=["none", "basic", "advanced"], ordered=True)
    df["optimizer"] = pd.Categorical(df["optimizer"],
                                      categories=["sgd", "adam", "adamw"], ordered=True)
    print(f"Loaded {len(df)} runs from {path}")
    cells = df.groupby(["resolution", "augmentation", "optimizer"]).size()
    print(f"  Treatments: {len(cells)}/27 | Reps per cell: {cells.min()}-{cells.max()}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ═════════════════════════════════════════════════════════════════════════════

def descriptive(df, out):
    resp = "test_accuracy"

    # Cell means
    cell = df.groupby(["resolution", "augmentation", "optimizer"])[resp].agg(
        ["count", "mean", "std"]).reset_index()
    cell.columns = ["resolution", "augmentation", "optimizer", "n", "mean", "std"]
    cell["se"] = cell["std"] / np.sqrt(cell["n"])
    cell["ci_lower"] = cell["mean"] - 1.96 * cell["se"]
    cell["ci_upper"] = cell["mean"] + 1.96 * cell["se"]
    cell.to_csv(os.path.join(RESULTS, "descriptive_stats.csv"), index=False)

    # Marginal means
    lines = []
    lines.append(f"Grand mean: {df[resp].mean():.2f}%  (std: {df[resp].std():.2f})")
    lines.append("")
    for factor in ["resolution", "augmentation", "optimizer"]:
        mm = df.groupby(factor)[resp].agg(["mean", "std", "count"])
        lines.append(f"--- {factor.upper()} ---")
        for level, row in mm.iterrows():
            lines.append(f"  {str(level):>10s}:  {row['mean']:.2f}%  (std={row['std']:.2f}, n={int(row['count'])})")
        lines.append("")

    out.append("DESCRIPTIVE STATISTICS")
    out.append("=" * 50)
    out.extend(lines)
    print("\n".join(lines))
    return cell


# ═════════════════════════════════════════════════════════════════════════════
# 2. ASSUMPTION CHECKS
# ═════════════════════════════════════════════════════════════════════════════

def assumptions(df, out):
    resp = "test_accuracy"
    formula = f"{resp} ~ C(resolution) * C(augmentation) * C(optimizer)"
    model = ols(formula, data=df).fit()
    resid = model.resid
    fitted = model.fittedvalues

    lines = []
    lines.append("\nASSUMPTION CHECKS")
    lines.append("=" * 50)

    # Shapiro-Wilk
    w, p = stats.shapiro(resid)
    lines.append(f"Shapiro-Wilk (normality):  W={w:.4f}, p={p:.4f}  "
                 f"{'✓ PASS' if p > 0.05 else '✗ FAIL'}")

    # Levene per factor
    for f in ["resolution", "augmentation", "optimizer"]:
        groups = [g[resp].values for _, g in df.groupby(f)]
        stat, p_lev = stats.levene(*groups)
        lines.append(f"Levene ({f:>14s}):  F={stat:.4f}, p={p_lev:.4f}  "
                     f"{'✓ PASS' if p_lev > 0.05 else '✗ FAIL'}")

    # Breusch-Pagan
    bp, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)
    lines.append(f"Breusch-Pagan:             LM={bp:.4f}, p={bp_p:.4f}  "
                 f"{'✓ PASS' if bp_p > 0.05 else '✗ FAIL'}")

    if p < 0.05:
        lines.append("\nNote: If normality is violated, ANOVA is still robust with balanced")
        lines.append("designs and n≥3 per cell (Lindman, 1974). Residual plots should be")
        lines.append("inspected visually to confirm no severe departures.")

    out.extend(lines)
    print("\n".join(lines))

    # ── Residual plots ──
    # QQ
    fig, ax = plt.subplots(figsize=(6, 5))
    stats.probplot(resid, plot=ax)
    ax.set_title("Normal Q-Q Plot of Residuals", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "residual_qq.png"), dpi=150)
    plt.close()

    # Histogram
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(resid, bins=20, edgecolor="white", color="#4393c3", alpha=0.85)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Residuals", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "residual_histogram.png"), dpi=150)
    plt.close()

    # Residuals vs fitted
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(fitted, resid, alpha=0.5, s=30, color="#2166ac")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Fitted Values", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "residual_vs_fitted.png"), dpi=150)
    plt.close()

    return model, resid


# ═════════════════════════════════════════════════════════════════════════════
# 3. THREE-WAY ANOVA
# ═════════════════════════════════════════════════════════════════════════════

def run_anova(df, out, alpha=0.05):
    resp = "test_accuracy"
    formula = f"{resp} ~ C(resolution) * C(augmentation) * C(optimizer)"
    model = ols(formula, data=df).fit()
    table = anova_lm(model, typ=2)

    ss_total = table["sum_sq"].sum()
    ss_resid = table.loc["Residual", "sum_sq"]
    table["eta_sq"] = table["sum_sq"] / ss_total
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + ss_resid)
    table["significant"] = table["PR(>F)"] < alpha

    # Readable names
    rename = {
        "C(resolution)": "Resolution (A)",
        "C(augmentation)": "Augmentation (B)",
        "C(optimizer)": "Optimizer (C)",
        "C(resolution):C(augmentation)": "A × B",
        "C(resolution):C(optimizer)": "A × C",
        "C(augmentation):C(optimizer)": "B × C",
        "C(resolution):C(augmentation):C(optimizer)": "A × B × C",
        "Residual": "Residual",
    }
    save_table = table.copy()
    save_table.index = [rename.get(i, i) for i in save_table.index]
    save_table = save_table.rename(columns={
        "sum_sq": "SS", "df": "DF", "F": "F_value",
        "PR(>F)": "p_value", "eta_sq": "eta_sq", "partial_eta_sq": "partial_eta_sq",
    })
    save_table.to_csv(os.path.join(RESULTS, "anova_table.csv"))

    lines = []
    lines.append("\nTHREE-WAY ANOVA (Type II SS)")
    lines.append("=" * 50)
    lines.append(f"α = {alpha}\n")

    # Format table for text output
    fmt = "{:<20s} {:>10s} {:>5s} {:>10s} {:>12s} {:>8s} {:>10s} {:>5s}"
    lines.append(fmt.format("Source", "SS", "DF", "F", "p-value", "η²", "partial η²", "Sig"))
    lines.append("-" * 90)
    for idx, row in table.iterrows():
        name = rename.get(idx, idx)
        sig = "*" if row.get("significant", False) and idx != "Residual" else ""
        if idx == "Residual":
            lines.append(fmt.format(
                name, f"{row['sum_sq']:.2f}", f"{row['df']:.0f}",
                "", "", f"{row['eta_sq']:.4f}", "", ""
            ))
        else:
            lines.append(fmt.format(
                name, f"{row['sum_sq']:.2f}", f"{row['df']:.0f}",
                f"{row['F']:.4f}", f"{row['PR(>F)']:.2e}",
                f"{row['eta_sq']:.4f}", f"{row['partial_eta_sq']:.4f}", sig
            ))
    lines.append("-" * 90)
    lines.append(f"\nR² = {model.rsquared:.4f}")
    lines.append(f"Adjusted R² = {model.rsquared_adj:.4f}")

    # List significant effects
    sig_effects = table[(table["significant"]) & (table.index != "Residual")]
    if len(sig_effects):
        lines.append(f"\nSignificant effects (p < {alpha}):")
        for idx, row in sig_effects.iterrows():
            name = rename.get(idx, idx)
            lines.append(f"  • {name}: F({row['df']:.0f},{table.loc['Residual','df']:.0f}) = "
                         f"{row['F']:.2f}, p = {row['PR(>F)']:.2e}, partial η² = {row['partial_eta_sq']:.4f}")
    else:
        lines.append("\nNo significant effects found.")

    out.extend(lines)
    print("\n".join(lines))
    return model, table


# ═════════════════════════════════════════════════════════════════════════════
# 4. POST-HOC (Tukey HSD)
# ═════════════════════════════════════════════════════════════════════════════

def posthoc(df, out, alpha=0.05):
    resp = "test_accuracy"
    lines = []
    lines.append("\nPOST-HOC COMPARISONS (Tukey HSD)")
    lines.append("=" * 50)

    for factor in ["resolution", "augmentation", "optimizer"]:
        tukey = pairwise_tukeyhsd(df[resp], df[factor].astype(str), alpha=alpha)

        lines.append(f"\n--- {factor.upper()} ---")
        lines.append(str(tukey.summary()))

        # Save to CSV
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                                 columns=tukey._results_table.data[0])
        tukey_df.to_csv(os.path.join(RESULTS, f"tukey_{factor}.csv"), index=False)

    out.extend(lines)
    print("\n".join(lines))


# ═════════════════════════════════════════════════════════════════════════════
# 5. PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_main_effects(df):
    resp = "test_accuracy"
    factors = ["resolution", "augmentation", "optimizer"]
    titles = ["A: Image Resolution", "B: Augmentation", "C: Optimizer"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, factor, title in zip(axes, factors, titles):
        mm = df.groupby(factor)[resp].agg(["mean", "std", "count"]).reset_index()
        mm["ci"] = 1.96 * mm["std"] / np.sqrt(mm["count"])
        x = range(len(mm))
        ax.errorbar(x, mm["mean"], yerr=mm["ci"], fmt="o-", capsize=6,
                     markersize=9, lw=2, color="#2166ac")
        ax.set_xticks(x)
        ax.set_xticklabels(mm[factor].astype(str))
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Test Accuracy (%)")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Main Effects Plot (means ± 95% CI)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "main_effects.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_interaction(df, f1, f2, label1, label2, filename):
    resp = "test_accuracy"
    means = df.groupby([f1, f2])[resp].mean().reset_index()
    palette = sns.color_palette("Set2", len(means[f2].unique()))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for j, level in enumerate(means[f2].cat.categories):
        sub = means[means[f2] == level]
        ax.plot(sub[f1].astype(str), sub[resp], "o-", label=f"{label2}={level}",
                markersize=7, lw=2, color=palette[j])

    ax.set_xlabel(label1)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Interaction: {label1} × {label2}", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, filename), dpi=150)
    plt.close()


def plot_boxplots(df):
    resp = "test_accuracy"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, f in zip(axes, ["resolution", "augmentation", "optimizer"]):
        sns.boxplot(x=f, y=resp, data=df, ax=ax, palette="Blues")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f.capitalize(), fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "boxplots.png"), dpi=150)
    plt.close()


def plot_heatmap(df):
    resp = "test_accuracy"
    pivot = df.pivot_table(values=resp, index="augmentation",
                            columns=["resolution", "optimizer"], aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title("Mean Test Accuracy (%) by All Factor Combinations", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "cell_means_heatmap.png"), dpi=150)
    plt.close()


def make_all_plots(df):
    print("\nGenerating plots...")
    plot_main_effects(df)
    plot_interaction(df, "resolution", "augmentation",
                     "Resolution", "Augmentation", "interaction_AB.png")
    plot_interaction(df, "resolution", "optimizer",
                     "Resolution", "Optimizer", "interaction_AC.png")
    plot_interaction(df, "augmentation", "optimizer",
                     "Augmentation", "Optimizer", "interaction_BC.png")
    plot_boxplots(df)
    plot_heatmap(df)
    print(f"  All plots saved to {PLOTS}/")


# ═════════════════════════════════════════════════════════════════════════════
# 6. CONCLUSIONS
# ═════════════════════════════════════════════════════════════════════════════

def conclusions(df, anova_table, out):
    resp = "test_accuracy"
    rename = {
        "C(resolution)": "Resolution",
        "C(augmentation)": "Augmentation",
        "C(optimizer)": "Optimizer",
        "C(resolution):C(augmentation)": "Resolution × Augmentation",
        "C(resolution):C(optimizer)": "Resolution × Optimizer",
        "C(augmentation):C(optimizer)": "Augmentation × Optimizer",
        "C(resolution):C(augmentation):C(optimizer)": "Three-way interaction",
    }

    lines = []
    lines.append("\nCONCLUSIONS")
    lines.append("=" * 50)

    sig = anova_table[(anova_table["PR(>F)"] < 0.05) & (anova_table.index != "Residual")]
    nonsig = anova_table[(anova_table["PR(>F)"] >= 0.05) & (anova_table.index != "Residual")]

    if len(sig):
        lines.append("\nThe following factors/interactions have a statistically significant")
        lines.append("effect on CIFAR-10 classification accuracy (α = 0.05):\n")
        for idx, row in sig.iterrows():
            name = rename.get(idx, idx)
            size = "large" if row["partial_eta_sq"] > 0.14 else \
                   "medium" if row["partial_eta_sq"] > 0.06 else "small"
            lines.append(f"  • {name}: F = {row['F']:.2f}, p = {row['PR(>F)']:.2e}, "
                         f"partial η² = {row['partial_eta_sq']:.4f} ({size} effect)")

    if len(nonsig):
        lines.append(f"\nNon-significant effects (p ≥ 0.05):")
        for idx, row in nonsig.iterrows():
            lines.append(f"  • {rename.get(idx, idx)}: p = {row['PR(>F)']:.4f}")

    # Best combination
    best = df.groupby(["resolution", "augmentation", "optimizer"])[resp].mean()
    best_idx = best.idxmax()
    lines.append(f"\nBest performing combination:")
    lines.append(f"  Resolution={best_idx[0]}, Augmentation={best_idx[1]}, "
                 f"Optimizer={best_idx[2]}")
    lines.append(f"  Mean accuracy: {best.max():.2f}%")

    worst_idx = best.idxmin()
    lines.append(f"\nWorst performing combination:")
    lines.append(f"  Resolution={worst_idx[0]}, Augmentation={worst_idx[1]}, "
                 f"Optimizer={worst_idx[2]}")
    lines.append(f"  Mean accuracy: {best.min():.2f}%")

    out.extend(lines)
    print("\n".join(lines))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiment_results.csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    report = []  # collects all text for summary.txt

    df = load(args.input)
    descriptive(df, report)
    assumptions(df, report)
    model, anova_table = run_anova(df, report, args.alpha)
    posthoc(df, report, args.alpha)
    make_all_plots(df)
    conclusions(df, anova_table, report)

    # Save full report
    with open(os.path.join(RESULTS, "summary.txt"), "w") as f:
        f.write("\n".join(report))
    print(f"\nFull report saved to {RESULTS}/summary.txt")
    print("Done.")


if __name__ == "__main__":
    main()
