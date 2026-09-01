from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from alpha_firewall.metrics import expected_maximum_sharpe

PROMOTION_COLUMNS = (
    "naive_promote",
    "dsr_promote",
    "dsr_effective_promote",
    "bonferroni_promote",
    "bh_promote",
    "holdout_promote",
    "dual_gate_promote",
    "dual_effective_gate_promote",
)

POLICY_LABELS = {
    "blind": "blind batch",
    "adaptive_sham": "sham feedback",
    "adaptive_target": "target feedback",
}

POLICY_COLORS = {
    "blind": "#6b7280",
    "adaptive_sham": "#2563eb",
    "adaptive_target": "#dc2626",
}


def _mean_interval(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    mean = float(clean.mean())
    if clean.size < 2:
        return mean, mean, mean
    half_width = float(t.ppf(0.975, clean.size - 1) * clean.std(ddof=1) / np.sqrt(clean.size))
    return mean, mean - half_width, mean + half_width


def _wilson_interval(successes: int, n_observations: int) -> tuple[float, float]:
    if n_observations == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / n_observations
    denominator = 1.0 + z**2 / n_observations
    center = (proportion + z**2 / (2.0 * n_observations)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / n_observations + z**2 / (4.0 * n_observations**2)
        )
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (policy, budget), group in results.groupby(["policy", "budget"], sort=True):
        record: dict[str, object] = {
            "policy": policy,
            "budget": int(budget),
            "n_markets": len(group),
            "effective_trials_mean": group["effective_trials"].mean(),
        }
        for column in ("validation_sharpe", "confirmation_sharpe", "optimism_gap"):
            mean, lower, upper = _mean_interval(group[column])
            record[f"{column}_mean"] = mean
            record[f"{column}_ci_low"] = lower
            record[f"{column}_ci_high"] = upper
        for column in PROMOTION_COLUMNS:
            rate = float(group[column].mean())
            lower, upper = _wilson_interval(int(group[column].sum()), len(group))
            stem = column.removesuffix("_promote")
            record[f"{stem}_rate"] = rate
            record[f"{stem}_ci_low"] = lower
            record[f"{stem}_ci_high"] = upper
        records.append(record)
    return pd.DataFrame(records).sort_values(["policy", "budget"])


def _paired_feedback_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ("validation_sharpe", "confirmation_sharpe", "optimism_gap")
    adaptive = results.loc[results["policy"].isin(("adaptive_sham", "adaptive_target"))]
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(20260825)
    for budget, group in adaptive.groupby("budget", sort=True):
        pivot = group.pivot(index="market_seed", columns="policy", values=list(metrics))
        record: dict[str, object] = {"budget": int(budget), "n_pairs": len(pivot)}
        for metric in metrics:
            differences = (
                pivot[(metric, "adaptive_target")] - pivot[(metric, "adaptive_sham")]
            ).dropna()
            mean, lower, upper = _mean_interval(differences)
            signs = rng.choice((-1.0, 1.0), size=(20_000, len(differences)))
            null_means = signs @ differences.to_numpy() / len(differences)
            p_value = float((1 + np.sum(null_means >= mean)) / (len(null_means) + 1))
            record[f"{metric}_difference"] = mean
            record[f"{metric}_ci_low"] = lower
            record[f"{metric}_ci_high"] = upper
            record[f"{metric}_one_sided_p"] = p_value
        records.append(record)
    return pd.DataFrame(records)


def build_report(results_path: Path, output_dir: Path) -> None:
    results = pd.read_csv(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _summarize(results)
    paired = _paired_feedback_summary(results)
    summary.to_csv(output_dir / "synthetic_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_feedback_summary.csv", index=False)
    (output_dir / "synthetic_summary.tex").write_text(
        summary.to_latex(index=False, float_format="%.3f"), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    for policy, group in summary.groupby("policy"):
        color = POLICY_COLORS[policy]
        label = POLICY_LABELS[policy]
        axes[0].plot(
            group["budget"],
            group["validation_sharpe_mean"],
            marker="o",
            color=color,
            label=f"{label}: validation",
        )
        axes[0].fill_between(
            group["budget"],
            group["validation_sharpe_ci_low"],
            group["validation_sharpe_ci_high"],
            color=color,
            alpha=0.10,
        )
        axes[0].plot(
            group["budget"],
            group["confirmation_sharpe_mean"],
            marker="x",
            linestyle="--",
            color=color,
            label=f"{label}: confirmation",
        )

    blind_single = results.loc[
        (results["policy"] == "blind") & (results["budget"] == 1),
        "validation_sharpe",
    ]
    if len(blind_single) > 1:
        trial_std = float(blind_single.std(ddof=1))
        budgets = np.array(sorted(results["budget"].unique()), dtype=float)
        dummy_sharpes = np.array([-trial_std / np.sqrt(2.0), trial_std / np.sqrt(2.0)])
        theoretical = [
            expected_maximum_sharpe(dummy_sharpes, n_effective_trials=budget) for budget in budgets
        ]
        axes[0].plot(
            budgets,
            theoretical,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="iid null order statistic",
        )

    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Candidates evaluated")
    axes[0].set_ylabel("Mean annualized Sharpe")
    axes[0].set_title("Selection creates validation alpha")
    axes[0].legend(fontsize=8)

    target = summary.loc[summary["policy"] == "adaptive_target"]
    gate_styles = {
        "naive": ("single-test HAC", "#dc2626", "o", "-"),
        "dsr": ("DSR (raw N)", "#f59e0b", "s", "-"),
        "dsr_effective": ("DSR (effective N)", "#ea580c", "P", "-"),
        "bonferroni": ("Bonferroni", "#7c3aed", "^", "-"),
        "bh": ("BH", "#db2777", "D", "-"),
        "holdout": ("sealed holdout", "#059669", "o", "--"),
        "dual_gate": ("raw-N DSR + holdout", "#0f766e", "x", "--"),
    }
    for stem, (label, color, marker, linestyle) in gate_styles.items():
        axes[1].plot(
            target["budget"],
            target[f"{stem}_rate"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
        )
    axes[1].axhline(0.05, color="black", linewidth=0.8, linestyle=":", label="5% target")
    axes[1].set_xscale("log")
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_xlabel("Candidates evaluated")
    axes[1].set_ylabel("False-promotion rate")
    axes[1].set_title("Promotion-rule calibration under the null")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_main.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "synthetic_main.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sweep_path = results_path.parent / "adaptivity_sweep.csv"
    if sweep_path.exists():
        build_sweep_report(sweep_path, output_dir)

    real_path = results_path.parent / "french_49_trials.csv"
    if real_path.exists():
        build_real_data_report(real_path, output_dir)


def build_sweep_report(sweep_path: Path, output_dir: Path) -> None:
    sweep = pd.read_csv(sweep_path)
    records: list[dict[str, object]] = []
    for (exploit_rate, elite_size), group in sweep.groupby(
        ["exploit_rate", "elite_size"], sort=True
    ):
        pivot = group.pivot(
            index="market_seed",
            columns="policy",
            values=["validation_sharpe", "confirmation_sharpe", "optimism_gap"],
        )
        record: dict[str, object] = {
            "exploit_rate": exploit_rate,
            "elite_size": int(elite_size),
            "n_pairs": len(pivot),
        }
        for metric in ("validation_sharpe", "confirmation_sharpe", "optimism_gap"):
            difference = pivot[(metric, "adaptive_target")] - pivot[(metric, "adaptive_sham")]
            mean, lower, upper = _mean_interval(difference)
            record[f"{metric}_difference"] = mean
            record[f"{metric}_ci_low"] = lower
            record[f"{metric}_ci_high"] = upper
        target = group.loc[group["policy"] == "adaptive_target"]
        record["target_dsr_rate"] = target["dsr_promote"].mean()
        record["target_holdout_rate"] = target["holdout_promote"].mean()
        records.append(record)
    summary = pd.DataFrame(records).sort_values(["elite_size", "exploit_rate"])
    summary.to_csv(output_dir / "adaptivity_sweep_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    for elite_size, group in summary.groupby("elite_size"):
        label = f"elite cap {elite_size}"
        axes[0].errorbar(
            group["exploit_rate"],
            group["validation_sharpe_difference"],
            yerr=[
                group["validation_sharpe_difference"] - group["validation_sharpe_ci_low"],
                group["validation_sharpe_ci_high"] - group["validation_sharpe_difference"],
            ],
            marker="o",
            capsize=3,
            label=label,
        )
        axes[1].errorbar(
            group["exploit_rate"],
            group["confirmation_sharpe_difference"],
            yerr=[
                group["confirmation_sharpe_difference"] - group["confirmation_sharpe_ci_low"],
                group["confirmation_sharpe_ci_high"] - group["confirmation_sharpe_difference"],
            ],
            marker="o",
            capsize=3,
            label=label,
        )
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        axis.set_xlabel("Exploit probability")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Target minus sham Sharpe")
    axes[0].set_title("Validation inflation persists")
    axes[1].set_title("No confirmation improvement")
    fig.tight_layout()
    fig.savefig(output_dir / "adaptivity_sweep.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "adaptivity_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_real_data_report(real_path: Path, output_dir: Path) -> None:
    real = pd.read_csv(real_path)
    summary = (
        real.groupby(["policy", "budget"], as_index=False)
        .agg(
            n_folds=("market_seed", "size"),
            validation_sharpe_mean=("validation_sharpe", "mean"),
            confirmation_sharpe_mean=("confirmation_sharpe", "mean"),
            optimism_gap_mean=("optimism_gap", "mean"),
        )
        .sort_values(["policy", "budget"])
    )
    summary.to_csv(output_dir / "french_49_summary.csv", index=False)
