#!/usr/bin/env python3
"""
ANOVA Analysis — Project Update 1

Design: 3 × 3 × 2 full factorial with GPU/Computer block
  Factor A – Augmentation:   None, Basic, Advanced
  Factor B – Optimizer:      SGD, Adam, AdamW
  Factor C – Test Dataset:   Clean, Adversarial
  Block    – Computer ID:    included as fixed factor to absorb hardware variance

Response variable: accuracy (%)

Analyses performed:
  1. Descriptive statistics (mean, std, SEM, 95% CI per cell)
  2. Assumption checks (Shapiro-Wilk, Levene's per factor, Breusch-Pagan)
  3. Three-way ANOVA (Type II SS) with block term
  4. Tukey HSD post-hoc comparisons for each factor
  5. Plots: main effects, interactions (A×B, A×C, B×C), box plots, heatmap,
            residual diagnostics

Usage:
    python new_analysis.py
    python new_analysis.py --input main_results.csv --alpha 0.05
"""

import argparse
import os
import warnings

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

PLOTS   = "main_plots"
RESULTS = "main_results"
RESPONSE = "accuracy"
RESPONSE_LABEL = "Classification Accuracy (%)"


# ── Data loading ──────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")

    df["augmentation"] = pd.Categorical(
        df["augmentation"], categories=["none", "basic", "advanced"], ordered=True)
    df["optimizer"] = pd.Categorical(
        df["optimizer"], categories=["sgd", "adam", "adamw"], ordered=True)
    df["test_dataset_type"] = pd.Categorical(
        df["test_dataset_type"], categories=["clean", "adversarial"], ordered=True)

    cells = df.groupby(["augmentation", "optimizer", "test_dataset_type"]).size()
    print(f"  Treatment cells: {len(cells)}/18 | "
          f"Reps per cell: {cells.min()}–{cells.max()}")

    if "computer_id" in df.columns:
        computers = df["computer_id"].unique()
        print(f"  Computer IDs (block): {list(computers)}")
    else:
        df["computer_id"] = "unknown"
        print("  Warning: no computer_id column found; block set to 'unknown'")

    return df


# ── Descriptive statistics ────────────────────────────────────────────────────

def descriptive(df, out):
    cell = df.groupby(["augmentation", "optimizer", "test_dataset_type"])[RESPONSE].agg(
        ["count", "mean", "std"]).reset_index()
    cell.columns = ["augmentation", "optimizer", "test_dataset_type", "n", "mean", "std"]
    cell["se"]       = cell["std"] / np.sqrt(cell["n"])
    cell["ci_lower"] = cell["mean"] - 1.96 * cell["se"]
    cell["ci_upper"] = cell["mean"] + 1.96 * cell["se"]
    cell.to_csv(os.path.join(RESULTS, "descriptive_stats.csv"), index=False)

    lines = [
        "\nDESCRIPTIVE STATISTICS",
        "=" * 50,
        f"Grand mean: {df[RESPONSE].mean():.2f}%  (std: {df[RESPONSE].std():.2f})",
        f"Total observations: {len(df)}",
        "",
    ]
    for factor in ["augmentation", "optimizer", "test_dataset_type"]:
        mm = df.groupby(factor)[RESPONSE].agg(["mean", "std", "count"])
        lines.append(f"--- {factor.upper()} ---")
        for level, row in mm.iterrows():
            lines.append(f"  {str(level):>12s}:  {row['mean']:.2f}%  "
                         f"(std={row['std']:.2f}, n={int(row['count'])})")
        lines.append("")

    if df["computer_id"].nunique() > 1:
        lines.append("--- COMPUTER (Block) ---")
        mm = df.groupby("computer_id")[RESPONSE].agg(["mean", "std", "count"])
        for level, row in mm.iterrows():
            lines.append(f"  {str(level):>12s}:  {row['mean']:.2f}%  "
                         f"(std={row['std']:.2f}, n={int(row['count'])})")
        lines.append("")

    out.extend(lines)
    print("\n".join(lines))
    return cell


# ── Assumption checks ─────────────────────────────────────────────────────────

def _build_formula(include_block):
    """ANOVA formula with or without block term."""
    block = "+ C(computer_id) " if include_block else ""
    return (
        f"{RESPONSE} ~ C(augmentation) + C(optimizer) + C(test_dataset_type) "
        f"{block}"
        f"+ C(augmentation):C(optimizer) "
        f"+ C(augmentation):C(test_dataset_type) "
        f"+ C(optimizer):C(test_dataset_type) "
        f"+ C(augmentation):C(optimizer):C(test_dataset_type)"
    )


def assumptions(df, out):
    has_block = df["computer_id"].nunique() > 1
    formula = _build_formula(has_block)
    model = ols(formula, data=df).fit()
    resid  = model.resid
    fitted = model.fittedvalues

    lines = [
        "\nASSUMPTION CHECKS",
        "=" * 50,
    ]

    w, p = stats.shapiro(resid)
    lines.append(f"Shapiro-Wilk (normality):  W={w:.4f}, p={p:.4f}  "
                 f"{'PASS' if p > 0.05 else 'FAIL'}")

    for factor in ["augmentation", "optimizer", "test_dataset_type"]:
        groups = [g[RESPONSE].values for _, g in df.groupby(factor)]
        stat, p_lev = stats.levene(*groups)
        lines.append(f"Levene ({factor:>17s}):  F={stat:.4f}, p={p_lev:.4f}  "
                     f"{'PASS' if p_lev > 0.05 else 'FAIL'}")

    bp, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)
    lines.append(f"Breusch-Pagan:              LM={bp:.4f}, p={bp_p:.4f}  "
                 f"{'PASS' if bp_p > 0.05 else 'FAIL'}")

    if p < 0.05:
        lines.append(
            "\nNote: ANOVA is robust to mild normality violations in balanced "
            "designs with n≥3 per cell (Lindman, 1974).")
    lines.append("")

    out.extend(lines)
    print("\n".join(lines))

    # Residual diagnostic plots
    fig, ax = plt.subplots(figsize=(6, 5))
    stats.probplot(resid, plot=ax)
    ax.set_title("Normal Q-Q Plot — Residuals", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "residual_qq.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(resid, bins=20, edgecolor="white", color="#4393c3", alpha=0.85)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Residual"); ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "residual_histogram.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(fitted, resid, alpha=0.5, s=30, color="#2166ac")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Fitted Values"); ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Fitted", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "residual_vs_fitted.png"), dpi=150)
    plt.close()

    return model, resid


# ── Three-way ANOVA ───────────────────────────────────────────────────────────

def run_anova(df, out, alpha=0.05):
    has_block = df["computer_id"].nunique() > 1
    formula = _build_formula(has_block)
    model = ols(formula, data=df).fit()
    table = anova_lm(model, typ=2)

    ss_total = table["sum_sq"].sum()
    ss_resid = table.loc["Residual", "sum_sq"]
    table["eta_sq"]         = table["sum_sq"] / ss_total
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + ss_resid)
    table["significant"]    = table["PR(>F)"] < alpha

    rename = {
        "C(augmentation)":                                 "Augmentation (A)",
        "C(optimizer)":                                    "Optimizer (B)",
        "C(test_dataset_type)":                            "Test Dataset (C)",
        "C(computer_id)":                                  "Block (Computer)",
        "C(augmentation):C(optimizer)":                    "A × B",
        "C(augmentation):C(test_dataset_type)":            "A × C",
        "C(optimizer):C(test_dataset_type)":               "B × C",
        "C(augmentation):C(optimizer):C(test_dataset_type)": "A × B × C",
        "Residual":                                        "Residual",
    }

    save_table = table.copy()
    save_table.index = [rename.get(i, i) for i in save_table.index]
    save_table.to_csv(os.path.join(RESULTS, "anova_table.csv"))

    lines = [
        "\nTHREE-WAY ANOVA (Type II SS)",
        "=" * 92,
        f"alpha = {alpha}",
        "",
    ]

    fmt = "{:<26s} {:>10s} {:>5s} {:>10s} {:>12s} {:>8s} {:>10s} {:>5s}"
    lines.append(fmt.format("Source", "SS", "DF", "F", "p-value",
                             "eta_sq", "p_eta_sq", "Sig"))
    lines.append("-" * 92)

    for idx, row in table.iterrows():
        name = rename.get(idx, idx)
        sig  = "*" if row.get("significant", False) and idx != "Residual" else ""
        if idx == "Residual":
            lines.append(fmt.format(
                name, f"{row['sum_sq']:.2f}", f"{row['df']:.0f}",
                "", "", f"{row['eta_sq']:.4f}", "", ""))
        else:
            lines.append(fmt.format(
                name, f"{row['sum_sq']:.2f}", f"{row['df']:.0f}",
                f"{row['F']:.4f}", f"{row['PR(>F)']:.2e}",
                f"{row['eta_sq']:.4f}", f"{row['partial_eta_sq']:.4f}", sig))

    lines.append("-" * 92)
    lines.append(f"\nR-squared          = {model.rsquared:.4f}")
    lines.append(f"Adjusted R-squared = {model.rsquared_adj:.4f}")

    sig_effects = table[(table["significant"]) & (table.index != "Residual")]
    if len(sig_effects):
        lines.append(f"\nSignificant effects (p < {alpha}):")
        df_res = table.loc["Residual", "df"]
        for idx, row in sig_effects.iterrows():
            name = rename.get(idx, idx)
            size = ("large"  if row["partial_eta_sq"] > 0.14 else
                    "medium" if row["partial_eta_sq"] > 0.06 else "small")
            lines.append(
                f"  * {name}: F({row['df']:.0f},{df_res:.0f}) = {row['F']:.2f}, "
                f"p = {row['PR(>F)']:.2e}, partial η² = {row['partial_eta_sq']:.4f} ({size})")
    else:
        lines.append(f"\nNo effects significant at alpha = {alpha}.")

    lines.append("")
    out.extend(lines)
    print("\n".join(lines))
    return model, table


# ── Post-hoc comparisons ──────────────────────────────────────────────────────

def posthoc(df, out, alpha=0.05):
    lines = [
        "\nPOST-HOC COMPARISONS (Tukey HSD)",
        "=" * 50,
    ]

    for factor in ["augmentation", "optimizer", "test_dataset_type"]:
        tukey = pairwise_tukeyhsd(df[RESPONSE], df[factor].astype(str), alpha=alpha)
        lines.append(f"\n--- {factor.upper()} ---")
        lines.append(str(tukey.summary()))

        tukey_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0])
        tukey_df.to_csv(os.path.join(RESULTS, f"tukey_{factor}.csv"), index=False)

    lines.append("")
    out.extend(lines)
    print("\n".join(lines))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_main_effects(df):
    factors = ["augmentation", "optimizer", "test_dataset_type"]
    titles  = ["A: Augmentation", "B: Optimizer", "C: Test Dataset Type"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, factor, title in zip(axes, factors, titles):
        mm = df.groupby(factor)[RESPONSE].agg(["mean", "std", "count"]).reset_index()
        mm["ci"] = 1.96 * mm["std"] / np.sqrt(mm["count"])
        x = range(len(mm))
        ax.errorbar(x, mm["mean"], yerr=mm["ci"], fmt="o-", capsize=6,
                    markersize=9, lw=2, color="#2166ac")
        ax.set_xticks(x)
        ax.set_xticklabels(mm[factor].astype(str))
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(RESPONSE_LABEL)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Main Effects — {RESPONSE_LABEL}", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "main_effects.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_interaction(df, f1, f2, label1, label2, filename):
    means   = df.groupby([f1, f2])[RESPONSE].mean().reset_index()
    palette = sns.color_palette("Set2", len(means[f2].unique()))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for j, level in enumerate(means[f2].cat.categories):
        sub = means[means[f2] == level]
        ax.plot(sub[f1].astype(str), sub[RESPONSE], "o-",
                label=f"{label2}={level}", markersize=7, lw=2, color=palette[j])

    ax.set_xlabel(label1)
    ax.set_ylabel(RESPONSE_LABEL)
    ax.set_title(f"Interaction: {label1} × {label2}", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, filename), dpi=150)
    plt.close()


def plot_boxplots(df):
    factors = ["augmentation", "optimizer", "test_dataset_type"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, f in zip(axes, factors):
        sns.boxplot(x=f, y=RESPONSE, data=df, ax=ax, palette="Blues")
        ax.set_ylabel(RESPONSE_LABEL)
        ax.set_title(f.replace("_", " ").title(), fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "boxplots.png"), dpi=150)
    plt.close()


def plot_heatmap(df):
    """Cell means heatmap: augmentation × optimizer, faceted by test_dataset_type."""
    dataset_types = df["test_dataset_type"].cat.categories.tolist()
    fig, axes = plt.subplots(1, len(dataset_types),
                              figsize=(7 * len(dataset_types), 4))
    if len(dataset_types) == 1:
        axes = [axes]

    for ax, dtype in zip(axes, dataset_types):
        sub = df[df["test_dataset_type"] == dtype]
        pivot = sub.pivot_table(
            values=RESPONSE, index="augmentation", columns="optimizer", aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax,
                    linewidths=0.5, linecolor="white",
                    vmin=df[RESPONSE].min() - 1, vmax=df[RESPONSE].max() + 1)
        ax.set_title(f"Mean Accuracy — {dtype.capitalize()} Dataset",
                     fontweight="bold")

    plt.suptitle("Cell Means Heatmap (Augmentation × Optimizer)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_clean_vs_adversarial(df):
    """Scatter of clean vs. adversarial accuracy per model (seed), coloured by augmentation."""
    clean_df = df[df["test_dataset_type"] == "clean"][
        ["augmentation", "optimizer", "seed", RESPONSE]].rename(
        columns={RESPONSE: "clean_accuracy"})
    adv_df = df[df["test_dataset_type"] == "adversarial"][
        ["augmentation", "optimizer", "seed", RESPONSE]].rename(
        columns={RESPONSE: "adv_accuracy"})

    merged = clean_df.merge(adv_df, on=["augmentation", "optimizer", "seed"])
    if merged.empty:
        return

    palette = sns.color_palette("Set2", merged["augmentation"].nunique())
    fig, ax = plt.subplots(figsize=(7, 6))
    for j, aug in enumerate(merged["augmentation"].cat.categories):
        sub = merged[merged["augmentation"] == aug]
        ax.scatter(sub["clean_accuracy"], sub["adv_accuracy"],
                   label=f"aug={aug}", s=60, alpha=0.75, color=palette[j])

    lim_min = min(merged["adv_accuracy"].min() - 2, 10)
    lim_max = max(merged["clean_accuracy"].max() + 2, 85)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="gray",
            alpha=0.5, label="clean = adversarial")
    ax.set_xlabel("Clean Accuracy (%)")
    ax.set_ylabel("Adversarial Accuracy (%)")
    ax.set_title("Clean vs. Adversarial Accuracy by Augmentation", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "clean_vs_adversarial.png"), dpi=150)
    plt.close()


def make_all_plots(df):
    print("\nGenerating plots...")
    plot_main_effects(df)
    plot_interaction(df, "augmentation", "optimizer",
                     "Augmentation", "Optimizer", "interaction_AB.png")
    plot_interaction(df, "augmentation", "test_dataset_type",
                     "Augmentation", "Test Dataset", "interaction_AC.png")
    plot_interaction(df, "optimizer", "test_dataset_type",
                     "Optimizer", "Test Dataset", "interaction_BC.png")
    plot_boxplots(df)
    plot_heatmap(df)
    plot_clean_vs_adversarial(df)
    print(f"  Plots saved to {PLOTS}/")


# ── Conclusions ───────────────────────────────────────────────────────────────

def conclusions(df, anova_table, out, alpha=0.05):
    rename = {
        "C(augmentation)":                                    "Augmentation (A)",
        "C(optimizer)":                                       "Optimizer (B)",
        "C(test_dataset_type)":                               "Test Dataset (C)",
        "C(computer_id)":                                     "Block (Computer)",
        "C(augmentation):C(optimizer)":                       "A × B",
        "C(augmentation):C(test_dataset_type)":               "A × C",
        "C(optimizer):C(test_dataset_type)":                  "B × C",
        "C(augmentation):C(optimizer):C(test_dataset_type)":  "A × B × C",
    }

    sig    = anova_table[(anova_table["PR(>F)"] < alpha) & (anova_table.index != "Residual")]
    nonsig = anova_table[(anova_table["PR(>F)"] >= alpha) & (anova_table.index != "Residual")]

    lines = ["\nCONCLUSIONS", "=" * 50]

    if len(sig):
        lines.append(f"\nSignificant effects (alpha = {alpha}):\n")
        for idx, row in sig.iterrows():
            name = rename.get(idx, idx)
            size = ("large"  if row["partial_eta_sq"] > 0.14 else
                    "medium" if row["partial_eta_sq"] > 0.06 else "small")
            lines.append(
                f"  * {name}: F = {row['F']:.2f}, p = {row['PR(>F)']:.2e}, "
                f"partial η² = {row['partial_eta_sq']:.4f} ({size} effect)")

    if len(nonsig):
        lines.append("\nNon-significant effects:")
        for idx, row in nonsig.iterrows():
            lines.append(f"  * {rename.get(idx, idx)}: p = {row['PR(>F)']:.4f}")

    # Best/worst cells
    best = df.groupby(["augmentation", "optimizer", "test_dataset_type"])[RESPONSE].mean()
    best_idx  = best.idxmax()
    worst_idx = best.idxmin()
    lines.append(
        f"\nHighest mean accuracy: aug={best_idx[0]}, opt={best_idx[1]}, "
        f"dataset={best_idx[2]}  →  {best.max():.2f}%")
    lines.append(
        f"Lowest  mean accuracy: aug={worst_idx[0]}, opt={worst_idx[1]}, "
        f"dataset={worst_idx[2]}  →  {best.min():.2f}%")
    lines.append("")

    out.extend(lines)
    print("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ANOVA analysis for 3×3×2 CIFAR-10 experiment")
    parser.add_argument("--input", default="main_results.csv",
                        help="Path to the results CSV (default: main_results.csv)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found.")
        print("Run new_experiment_runner.py first to collect data.")
        return

    os.makedirs(PLOTS,   exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    report = [
        "=" * 70,
        "CIFAR-10 MULTIFACTOR ANOVA — PROJECT UPDATE 1",
        "Design: 3×3×2 (Augmentation × Optimizer × Test Dataset Type)",
        "Block:  GPU/Computer",
        f"Response: {RESPONSE_LABEL}",
        "=" * 70,
    ]

    df = load(args.input)

    descriptive(df, report)
    assumptions(df, report)
    model, anova_table = run_anova(df, report, args.alpha)
    posthoc(df, report, args.alpha)
    make_all_plots(df)
    conclusions(df, anova_table, report, args.alpha)

    summary_path = os.path.join(RESULTS, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(report))

    print(f"\nFull report saved to {summary_path}")
    print(f"ANOVA table:       {RESULTS}/anova_table.csv")
    print(f"Descriptive stats: {RESULTS}/descriptive_stats.csv")
    print(f"Tukey results:     {RESULTS}/tukey_*.csv")
    print(f"Plots:             {PLOTS}/")
    print("Done.")


if __name__ == "__main__":
    main()
