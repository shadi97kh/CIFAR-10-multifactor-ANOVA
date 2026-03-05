#!/usr/bin/env python3
"""
ANOVA Analysis — Adversarial Training Edition

Analyzes THREE response variables:
  1. Clean test accuracy (%)
  2. Adversarial test accuracy (%) under FGSM attack
  3. Robustness gap (clean - adversarial)

Usage:
    python adversarial_analysis.py
    python adversarial_analysis.py --input my_results.csv
"""

import argparse, os, sys, warnings
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

PLOTS = "adversarial_plots"
RESULTS = "adversarial_results"

FACTOR_COL = "training_method"
FACTOR_LABEL = "Training Method"
FACTOR_CATEGORIES = ["standard", "fgsm", "pgd"]


def load(path, response):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} runs from {path}")
    df["resolution"] = pd.Categorical(
        df["resolution"].astype(str), categories=["32", "64", "96"], ordered=True)
    df[FACTOR_COL] = pd.Categorical(
        df[FACTOR_COL], categories=FACTOR_CATEGORIES, ordered=True)
    df["optimizer"] = pd.Categorical(
        df["optimizer"], categories=["sgd", "adam", "adamw"], ordered=True)
    cells = df.groupby(["resolution", FACTOR_COL, "optimizer"]).size()
    print(f"  Treatments: {len(cells)}/27 | Reps per cell: {cells.min()}-{cells.max()}")
    return df


def descriptive(df, resp, out):
    cell = df.groupby(["resolution", FACTOR_COL, "optimizer"])[resp].agg(
        ["count", "mean", "std"]).reset_index()
    cell.columns = ["resolution", FACTOR_COL, "optimizer", "n", "mean", "std"]
    cell["se"] = cell["std"] / np.sqrt(cell["n"])
    cell["ci_lower"] = cell["mean"] - 1.96 * cell["se"]
    cell["ci_upper"] = cell["mean"] + 1.96 * cell["se"]
    cell.to_csv(os.path.join(RESULTS, f"descriptive_stats_{resp}.csv"), index=False)

    lines = []
    lines.append(f"\nDESCRIPTIVE STATISTICS — {resp}")
    lines.append("=" * 50)
    lines.append(f"Grand mean: {df[resp].mean():.2f}%  (std: {df[resp].std():.2f})")
    lines.append("")
    for factor in ["resolution", FACTOR_COL, "optimizer"]:
        mm = df.groupby(factor)[resp].agg(["mean", "std", "count"])
        lines.append(f"--- {factor.upper()} ---")
        for level, row in mm.iterrows():
            lines.append(f"  {str(level):>10s}:  {row['mean']:.2f}%  "
                         f"(std={row['std']:.2f}, n={int(row['count'])})")
        lines.append("")
    out.extend(lines)
    print("\n".join(lines))
    return cell


def assumptions(df, resp, out):
    formula = f"{resp} ~ C(resolution) * C({FACTOR_COL}) * C(optimizer)"
    model = ols(formula, data=df).fit()
    resid = model.resid
    fitted = model.fittedvalues

    lines = []
    lines.append(f"\nASSUMPTION CHECKS — {resp}")
    lines.append("=" * 50)

    w, p = stats.shapiro(resid)
    lines.append(f"Shapiro-Wilk (normality):  W={w:.4f}, p={p:.4f}  "
                 f"{'PASS' if p > 0.05 else 'FAIL'}")

    for f in ["resolution", FACTOR_COL, "optimizer"]:
        groups = [g[resp].values for _, g in df.groupby(f)]
        stat, p_lev = stats.levene(*groups)
        lines.append(f"Levene ({f:>16s}):  F={stat:.4f}, p={p_lev:.4f}  "
                     f"{'PASS' if p_lev > 0.05 else 'FAIL'}")

    bp, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)
    lines.append(f"Breusch-Pagan:              LM={bp:.4f}, p={bp_p:.4f}  "
                 f"{'PASS' if bp_p > 0.05 else 'FAIL'}")

    if p < 0.05:
        lines.append("\nNote: ANOVA is robust to mild normality violations with balanced")
        lines.append("designs and n>=3 per cell (Lindman, 1974).")

    out.extend(lines)
    print("\n".join(lines))

    suffix = resp.replace("_", "-")

    fig, ax = plt.subplots(figsize=(6, 5))
    stats.probplot(resid, plot=ax)
    ax.set_title(f"Normal Q-Q Plot — {resp}", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"residual_qq_{suffix}.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(resid, bins=20, edgecolor="white", color="#4393c3", alpha=0.85)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Residual"); ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution — {resp}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"residual_histogram_{suffix}.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(fitted, resid, alpha=0.5, s=30, color="#2166ac")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Fitted Values"); ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals vs. Fitted — {resp}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"residual_vs_fitted_{suffix}.png"), dpi=150)
    plt.close()

    return model, resid


def run_anova(df, resp, out, alpha=0.05):
    formula = f"{resp} ~ C(resolution) * C({FACTOR_COL}) * C(optimizer)"
    model = ols(formula, data=df).fit()
    table = anova_lm(model, typ=2)

    ss_total = table["sum_sq"].sum()
    ss_resid = table.loc["Residual", "sum_sq"]
    table["eta_sq"] = table["sum_sq"] / ss_total
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + ss_resid)
    table["significant"] = table["PR(>F)"] < alpha

    rename = {
        "C(resolution)": "Resolution (A)",
        f"C({FACTOR_COL})": "Training Method (B)",
        "C(optimizer)": "Optimizer (C)",
        f"C(resolution):C({FACTOR_COL})": "A x B",
        "C(resolution):C(optimizer)": "A x C",
        f"C({FACTOR_COL}):C(optimizer)": "B x C",
        f"C(resolution):C({FACTOR_COL}):C(optimizer)": "A x B x C",
        "Residual": "Residual",
    }

    save_table = table.copy()
    save_table.index = [rename.get(i, i) for i in save_table.index]
    save_table.to_csv(os.path.join(RESULTS, f"anova_table_{resp}.csv"))

    lines = []
    lines.append(f"\nTHREE-WAY ANOVA (Type II SS) — {resp}")
    lines.append("=" * 50)
    lines.append(f"alpha = {alpha}\n")

    fmt = "{:<22s} {:>10s} {:>5s} {:>10s} {:>12s} {:>8s} {:>10s} {:>5s}"
    lines.append(fmt.format("Source", "SS", "DF", "F", "p-value",
                             "eta_sq", "p_eta_sq", "Sig"))
    lines.append("-" * 92)
    for idx, row in table.iterrows():
        name = rename.get(idx, idx)
        sig = "*" if row.get("significant", False) and idx != "Residual" else ""
        if idx == "Residual":
            lines.append(fmt.format(name, f"{row['sum_sq']:.2f}",
                         f"{row['df']:.0f}", "", "", f"{row['eta_sq']:.4f}", "", ""))
        else:
            lines.append(fmt.format(
                name, f"{row['sum_sq']:.2f}", f"{row['df']:.0f}",
                f"{row['F']:.4f}", f"{row['PR(>F)']:.2e}",
                f"{row['eta_sq']:.4f}", f"{row['partial_eta_sq']:.4f}", sig))
    lines.append("-" * 92)
    lines.append(f"\nR-squared = {model.rsquared:.4f}")
    lines.append(f"Adjusted R-squared = {model.rsquared_adj:.4f}")

    sig_effects = table[(table["significant"]) & (table.index != "Residual")]
    if len(sig_effects):
        lines.append(f"\nSignificant effects (p < {alpha}):")
        for idx, row in sig_effects.iterrows():
            name = rename.get(idx, idx)
            lines.append(f"  * {name}: F({row['df']:.0f},{table.loc['Residual','df']:.0f})"
                         f" = {row['F']:.2f}, p = {row['PR(>F)']:.2e}, "
                         f"partial eta-sq = {row['partial_eta_sq']:.4f}")

    out.extend(lines)
    print("\n".join(lines))
    return model, table


def posthoc(df, resp, out, alpha=0.05):
    lines = []
    lines.append(f"\nPOST-HOC COMPARISONS (Tukey HSD) — {resp}")
    lines.append("=" * 50)

    for factor in ["resolution", FACTOR_COL, "optimizer"]:
        tukey = pairwise_tukeyhsd(df[resp], df[factor].astype(str), alpha=alpha)
        lines.append(f"\n--- {factor.upper()} ---")
        lines.append(str(tukey.summary()))
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                                 columns=tukey._results_table.data[0])
        tukey_df.to_csv(os.path.join(RESULTS, f"tukey_{factor}_{resp}.csv"), index=False)

    out.extend(lines)
    print("\n".join(lines))


def plot_main_effects(df, resp, label):
    factors = ["resolution", FACTOR_COL, "optimizer"]
    titles = ["A: Image Resolution", "B: Training Method", "C: Optimizer"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, factor, title in zip(axes, factors, titles):
        mm = df.groupby(factor)[resp].agg(["mean", "std", "count"]).reset_index()
        mm["ci"] = 1.96 * mm["std"] / np.sqrt(mm["count"])
        x = range(len(mm))
        ax.errorbar(x, mm["mean"], yerr=mm["ci"], fmt="o-", capsize=6,
                     markersize=9, lw=2, color="#2166ac")
        ax.set_xticks(x)
        ax.set_xticklabels(mm[factor].astype(str))
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Main Effects — {label}", fontweight="bold", y=1.02)
    plt.tight_layout()
    suffix = resp.replace("_", "-")
    plt.savefig(os.path.join(PLOTS, f"main_effects_{suffix}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_interaction(df, f1, f2, label1, label2, resp, resp_label, filename):
    means = df.groupby([f1, f2])[resp].mean().reset_index()
    palette = sns.color_palette("Set2", len(means[f2].unique()))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for j, level in enumerate(means[f2].cat.categories):
        sub = means[means[f2] == level]
        ax.plot(sub[f1].astype(str), sub[resp], "o-", label=f"{label2}={level}",
                markersize=7, lw=2, color=palette[j])

    ax.set_xlabel(label1); ax.set_ylabel(resp_label)
    ax.set_title(f"Interaction: {label1} x {label2}", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, filename), dpi=150)
    plt.close()


def plot_boxplots(df, resp, label):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, f in zip(axes, ["resolution", FACTOR_COL, "optimizer"]):
        sns.boxplot(x=f, y=resp, data=df, ax=ax, palette="Blues")
        ax.set_ylabel(label)
        ax.set_title(f.replace("_", " ").title(), fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    suffix = resp.replace("_", "-")
    plt.savefig(os.path.join(PLOTS, f"boxplots_{suffix}.png"), dpi=150)
    plt.close()


def plot_heatmap(df, resp, label):
    pivot = df.pivot_table(values=resp, index=FACTOR_COL,
                            columns=["resolution", "optimizer"], aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title(f"Mean {label} by All Factor Combinations", fontweight="bold")
    plt.tight_layout()
    suffix = resp.replace("_", "-")
    plt.savefig(os.path.join(PLOTS, f"heatmap_{suffix}.png"), dpi=150)
    plt.close()


def plot_clean_vs_adversarial(df):
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = {"standard": "#e74c3c", "fgsm": "#3498db", "pgd": "#2ecc71"}
    for method in FACTOR_CATEGORIES:
        sub = df[df[FACTOR_COL] == method]
        ax.scatter(sub["clean_accuracy"], sub["adversarial_accuracy"],
                   label=method, s=60, alpha=0.7, color=palette[method])
    lims = [min(df["adversarial_accuracy"].min(), 10),
            max(df["clean_accuracy"].max() + 2, 85)]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="clean = adversarial")
    ax.set_xlabel("Clean Accuracy (%)")
    ax.set_ylabel("Adversarial Accuracy (%)")
    ax.set_title("Clean vs. Adversarial Accuracy by Training Method", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "clean_vs_adversarial.png"), dpi=150)
    plt.close()


def plot_robustness_tradeoff(df):
    methods = FACTOR_CATEGORIES
    clean = [df[df[FACTOR_COL] == m]["clean_accuracy"].mean() for m in methods]
    adv = [df[df[FACTOR_COL] == m]["adversarial_accuracy"].mean() for m in methods]
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, clean, width, label="Clean Accuracy",
                    color="#3498db", edgecolor="white")
    bars2 = ax.bar(x + width/2, adv, width, label="Adversarial Accuracy",
                    color="#e74c3c", edgecolor="white")
    ax.set_xlabel("Training Method"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy-Robustness Tradeoff by Training Method", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in methods])
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "robustness_tradeoff.png"), dpi=150)
    plt.close()


def make_all_plots(df, resp, label):
    print(f"\nGenerating plots for {resp}...")
    suffix = resp.replace("_", "-")
    plot_main_effects(df, resp, label)
    plot_interaction(df, "resolution", FACTOR_COL, "Resolution", "Training Method",
                     resp, label, f"interaction_AB_{suffix}.png")
    plot_interaction(df, "resolution", "optimizer", "Resolution", "Optimizer",
                     resp, label, f"interaction_AC_{suffix}.png")
    plot_interaction(df, FACTOR_COL, "optimizer", "Training Method", "Optimizer",
                     resp, label, f"interaction_BC_{suffix}.png")
    plot_boxplots(df, resp, label)
    plot_heatmap(df, resp, label)
    if resp == "clean_accuracy" and "adversarial_accuracy" in df.columns:
        plot_clean_vs_adversarial(df)
        plot_robustness_tradeoff(df)
    print(f"  Plots saved to {PLOTS}/")


def conclusions(df, anova_table, resp, out):
    rename = {
        "C(resolution)": "Resolution",
        f"C({FACTOR_COL})": "Training Method",
        "C(optimizer)": "Optimizer",
        f"C(resolution):C({FACTOR_COL})": "Resolution x Training Method",
        "C(resolution):C(optimizer)": "Resolution x Optimizer",
        f"C({FACTOR_COL}):C(optimizer)": "Training Method x Optimizer",
        f"C(resolution):C({FACTOR_COL}):C(optimizer)": "Three-way interaction",
    }

    lines = []
    lines.append(f"\nCONCLUSIONS — {resp}")
    lines.append("=" * 50)

    sig = anova_table[(anova_table["PR(>F)"] < 0.05) & (anova_table.index != "Residual")]
    nonsig = anova_table[(anova_table["PR(>F)"] >= 0.05) & (anova_table.index != "Residual")]

    if len(sig):
        lines.append("\nSignificant effects (alpha = 0.05):\n")
        for idx, row in sig.iterrows():
            name = rename.get(idx, idx)
            size = "large" if row["partial_eta_sq"] > 0.14 else \
                   "medium" if row["partial_eta_sq"] > 0.06 else "small"
            lines.append(f"  * {name}: F = {row['F']:.2f}, p = {row['PR(>F)']:.2e}, "
                         f"partial eta-sq = {row['partial_eta_sq']:.4f} ({size} effect)")

    if len(nonsig):
        lines.append(f"\nNon-significant effects:")
        for idx, row in nonsig.iterrows():
            lines.append(f"  * {rename.get(idx, idx)}: p = {row['PR(>F)']:.4f}")

    best = df.groupby(["resolution", FACTOR_COL, "optimizer"])[resp].mean()
    best_idx = best.idxmax()
    worst_idx = best.idxmin()
    lines.append(f"\nBest: Resolution={best_idx[0]}, Method={best_idx[1]}, "
                 f"Optimizer={best_idx[2]} -> {best.max():.2f}%")
    lines.append(f"Worst: Resolution={worst_idx[0]}, Method={worst_idx[1]}, "
                 f"Optimizer={worst_idx[2]} -> {best.min():.2f}%")

    out.extend(lines)
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="adversarial_experiment_results.csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    responses = {
        "clean_accuracy": "Clean Test Accuracy (%)",
        "adversarial_accuracy": "Adversarial Accuracy (%)",
        "robustness_gap": "Robustness Gap (clean - adv, %)",
    }

    report = []
    df = load(args.input, "clean_accuracy")

    for resp, label in responses.items():
        report.append("\n" + "#" * 60)
        report.append(f"# ANALYSIS: {label}")
        report.append("#" * 60)

        descriptive(df, resp, report)
        assumptions(df, resp, report)
        model, anova_table = run_anova(df, resp, report, args.alpha)
        posthoc(df, resp, report, args.alpha)
        make_all_plots(df, resp, label)
        conclusions(df, anova_table, resp, report)

    with open(os.path.join(RESULTS, "summary.txt"), "w") as f:
        f.write("\n".join(report))
    print(f"\nFull report saved to {RESULTS}/summary.txt")
    print("Done.")


if __name__ == "__main__":
    main()
