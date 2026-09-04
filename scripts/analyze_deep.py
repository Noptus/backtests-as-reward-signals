from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
OUT = ROOT / "artifacts" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

POLICY_LABELS = {
    "blind": "Blind",
    "adaptive_sham": "Sham feedback",
    "adaptive_target": "Target feedback",
    "claude_blind": "Claude: blind",
    "claude_sham": "Claude: sham",
    "claude_target": "Claude: exact",
    "claude_coarsened": "Claude: coarsened",
}
COLORS = {
    "blind": "#6B7280",
    "adaptive_sham": "#2563EB",
    "adaptive_target": "#DC2626",
    "claude_blind": "#6B7280",
    "claude_sham": "#2563EB",
    "claude_target": "#DC2626",
    "claude_coarsened": "#059669",
}


def mean_ci(values: pd.Series | np.ndarray, level: float = 0.95) -> dict[str, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    mean = float(clean.mean())
    sem = float(stats.sem(clean))
    critical = float(stats.t.ppf((1.0 + level) / 2.0, len(clean) - 1))
    return {"mean": mean, "low": mean - critical * sem, "high": mean + critical * sem}


def wilson(values: pd.Series | np.ndarray, level: float = 0.95) -> dict[str, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = len(clean)
    p = float(clean.mean())
    z = float(stats.norm.ppf((1.0 + level) / 2.0))
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return {"mean": p, "low": float(center - half), "high": float(center + half)}


def sign_flip_p(values: pd.Series | np.ndarray, draws: int = 99_999) -> float:
    """Two-sided paired sign-randomization p-value with deterministic Monte Carlo."""

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    observed = abs(float(clean.mean()))
    if len(clean) <= 20:
        assignments = np.arange(2 ** len(clean), dtype=np.uint32)[:, None]
        bits = (assignments >> np.arange(len(clean), dtype=np.uint32)) & 1
        signs = bits.astype(float) * 2.0 - 1.0
        randomized = np.abs((signs * clean).mean(axis=1))
        return float(np.mean(randomized >= observed - 1e-15))
    rng = np.random.default_rng(20260825)
    exceedances = 0
    remaining = draws
    while remaining:
        batch = min(2_000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(batch, len(clean)))
        randomized = np.abs((signs * clean).mean(axis=1))
        exceedances += int(np.sum(randomized >= observed))
        remaining -= batch
    return float((exceedances + 1) / (draws + 1))


def latex_probability(value: float) -> str:
    """Format a probability for use inside a LaTeX math environment."""

    if value < 0.001:
        exponent = int(np.floor(np.log10(value)))
        coefficient = value / (10**exponent)
        return rf"{coefficient:.1f}\times 10^{{{exponent}}}"
    return f"{value:.3f}"


def paired_contrast(
    frame: pd.DataFrame,
    *,
    index: list[str],
    column: str,
    left: str,
    right: str,
) -> pd.Series:
    pivot = frame.pivot(index=index, columns="policy", values=column)
    return pivot[left] - pivot[right]


def save_table(frame: pd.DataFrame, name: str, formats: dict[str, str] | None = None) -> None:
    formats = formats or {}
    formatted = frame.copy()
    for column, pattern in formats.items():
        formatted[column] = formatted[column].map(
            lambda value, selected_pattern=pattern: selected_pattern.format(value)
        )
    (OUT / name).write_text(
        formatted.to_latex(
            index=False, escape=True, column_format="l" + "r" * (len(frame.columns) - 1)
        ),
        encoding="utf-8",
    )


def synthetic_analysis(summary: dict[str, object]) -> None:
    data = pd.read_csv(RESULTS / "synthetic_deep_paper_trials.csv")
    maximum = int(data["budget"].max())
    final = data[data["budget"].eq(maximum)].copy()
    contrasts = {}
    for column in ("validation_sharpe", "confirmation_sharpe", "optimism_gap"):
        values = paired_contrast(
            final,
            index=["market_seed"],
            column=column,
            left="adaptive_target",
            right="adaptive_sham",
        )
        contrasts[column] = mean_ci(values)
        contrasts[column]["sign_flip_p"] = sign_flip_p(values)
    gates = [
        "naive_promote",
        "dsr_promote",
        "dsr_effective_promote",
        "bonferroni_promote",
        "bh_promote",
        "holdout_promote",
        "dual_effective_gate_promote",
    ]
    gate_summary: dict[str, object] = {}
    rows = []
    for policy in ("blind", "adaptive_sham", "adaptive_target"):
        subset = final[final["policy"].eq(policy)]
        gate_summary[policy] = {}
        row: dict[str, object] = {"Policy": POLICY_LABELS[policy]}
        for gate in gates:
            interval = wilson(subset[gate])
            gate_summary[policy][gate] = interval
            row[gate] = 100.0 * interval["mean"]
        rows.append(row)
    summary["synthetic"] = {
        "rows": len(data),
        "candidate_trials": 1_500_000,
        "markets": int(data["market_seed"].nunique()),
        "contrasts": contrasts,
        "gates": gate_summary,
    }
    save_table(
        pd.DataFrame(rows).rename(
            columns={
                "naive_promote": "Naive",
                "dsr_promote": "DSR raw N",
                "dsr_effective_promote": "DSR N-eff",
                "bonferroni_promote": "Bonf.",
                "bh_promote": "BH",
                "holdout_promote": "Sealed",
                "dual_effective_gate_promote": "N-eff + sealed",
            }
        ),
        "synthetic_gate_table.tex",
        {
            column: "{:.1f}"
            for column in [
                "Naive",
                "DSR raw N",
                "DSR N-eff",
                "Bonf.",
                "BH",
                "Sealed",
                "N-eff + sealed",
            ]
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), sharex=True)
    for policy in ("blind", "adaptive_sham", "adaptive_target"):
        grouped = data[data["policy"].eq(policy)].groupby("budget")
        for axis, column, title in zip(
            axes,
            ("validation_sharpe", "confirmation_sharpe"),
            ("Selected validation Sharpe", "Sealed confirmation Sharpe"),
            strict=True,
        ):
            means = grouped[column].mean()
            errors = 1.96 * grouped[column].sem()
            axis.plot(
                means.index, means, marker="o", label=POLICY_LABELS[policy], color=COLORS[policy]
            )
            axis.fill_between(
                means.index, means - errors, means + errors, color=COLORS[policy], alpha=0.12
            )
            axis.set_title(title)
            axis.set_xlabel("Candidate budget")
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("Annualized Sharpe")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_trajectories.pdf", bbox_inches="tight")
    fig.savefig(OUT / "synthetic_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    selected_gates = [
        "dsr_promote",
        "dsr_effective_promote",
        "bonferroni_promote",
        "bh_promote",
        "holdout_promote",
    ]
    labels = ["DSR raw N", "DSR $N_{eff}$", "Bonferroni", "BH", "Sealed"]
    x = np.arange(len(labels))
    width = 0.24
    fig, axis = plt.subplots(figsize=(7.1, 2.8))
    for offset, policy in enumerate(("blind", "adaptive_sham", "adaptive_target")):
        subset = final[final["policy"].eq(policy)]
        rates = [100.0 * subset[gate].mean() for gate in selected_gates]
        axis.bar(
            x + (offset - 1) * width,
            rates,
            width,
            label=POLICY_LABELS[policy],
            color=COLORS[policy],
        )
    axis.axhline(5.0, color="black", linestyle="--", linewidth=0.9, label="Nominal 5%")
    axis.set_xticks(x, labels)
    axis.set_ylabel("False promotions (%)")
    axis.set_ylim(0, 48)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_gates.pdf", bbox_inches="tight")
    fig.savefig(OUT / "synthetic_gates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def empirical_analysis(summary: dict[str, object]) -> None:
    path = RESULTS / "empirical_null_trials.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    maximum = int(data["budget"].max())
    final = data[data["budget"].eq(maximum)]
    cell_index = ["dataset", "block_start", "null_index"]
    val_path = paired_contrast(
        final,
        index=cell_index,
        column="validation_sharpe",
        left="adaptive_target",
        right="adaptive_sham",
    )
    conf_path = paired_contrast(
        final,
        index=cell_index,
        column="confirmation_sharpe",
        left="adaptive_target",
        right="adaptive_sham",
    )
    val_cells = val_path.groupby(level=[0, 1]).mean()
    conf_cells = conf_path.groupby(level=[0, 1]).mean()
    panel_rows = []
    panel_labels = {
        "industries_49": "49 industries",
        "size_bm_25": "Size--BM 25",
        "size_bm_100": "Size--BM 100",
        "size_momentum_25": "Size--momentum 25",
        "size_short_reversal_25": "Size--reversal 25",
    }
    for dataset in sorted(data["dataset"].unique()):
        panel_rows.append(
            {
                "Panel": panel_labels[dataset],
                "Validation diff.": val_cells.loc[dataset].mean(),
                "Confirmation diff.": conf_cells.loc[dataset].mean(),
                "Target Bonf. (%)": 100.0
                * final[(final["dataset"].eq(dataset)) & (final["policy"].eq("adaptive_target"))][
                    "bonferroni_promote"
                ].mean(),
                "Target sealed (%)": 100.0
                * final[(final["dataset"].eq(dataset)) & (final["policy"].eq("adaptive_target"))][
                    "holdout_promote"
                ].mean(),
            }
        )
    save_table(
        pd.DataFrame(panel_rows),
        "empirical_table.tex",
        {
            "Validation diff.": "{:.3f}",
            "Confirmation diff.": "{:.3f}",
            "Target Bonf. (%)": "{:.1f}",
            "Target sealed (%)": "{:.1f}",
        },
    )
    validation_interval = mean_ci(val_cells)
    validation_interval["sign_flip_p"] = sign_flip_p(val_cells)
    confirmation_interval = mean_ci(conf_cells)
    confirmation_interval["sign_flip_p"] = sign_flip_p(conf_cells)
    # The five panels share the same market calendar and underlying security
    # universe.  The primary intervals are conditional Monte Carlo summaries
    # over independently signed cells; era-collapsed intervals are a deliberately
    # conservative sensitivity that does not treat panels as independent units.
    validation_era = val_cells.groupby(level="block_start").mean()
    confirmation_era = conf_cells.groupby(level="block_start").mean()
    validation_era_interval = mean_ci(validation_era)
    validation_era_interval["sign_flip_p"] = sign_flip_p(validation_era)
    confirmation_era_interval = mean_ci(confirmation_era)
    confirmation_era_interval["sign_flip_p"] = sign_flip_p(confirmation_era)
    target_final = final[final["policy"].eq("adaptive_target")]
    target_gates = {
        gate: float(target_final[gate].mean())
        for gate in (
            "dsr_promote",
            "dsr_effective_promote",
            "bonferroni_promote",
            "holdout_promote",
        )
    }
    summary["empirical"] = {
        "rows": len(data),
        "candidate_trials": 225_000,
        "paths": int(len(val_path)),
        "cells": int(len(val_cells)),
        "validation_contrast_cell_level": validation_interval,
        "confirmation_contrast_cell_level": confirmation_interval,
        "validation_contrast_era_sensitivity": validation_era_interval,
        "confirmation_contrast_era_sensitivity": confirmation_era_interval,
        "target_gate_rates": target_gates,
    }
    labels = [f"{dataset}\n{year}" for dataset, year in val_cells.index]
    values = val_cells.to_numpy()
    fig, axis = plt.subplots(figsize=(7.1, 4.2))
    y = np.arange(len(values))
    axis.scatter(values, y, color="#DC2626", s=25)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.axvline(values.mean(), color="#DC2626", linestyle="--", linewidth=1.0)
    axis.set_yticks(y, labels, fontsize=7)
    axis.set_xlabel("Target minus sham validation Sharpe")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "empirical_forest.pdf", bbox_inches="tight")
    fig.savefig(OUT / "empirical_forest.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def agentic_analysis(summary: dict[str, object]) -> None:
    path = RESULTS / "claude_agent_trials.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    audit = pd.read_csv(RESULTS / "claude_agent_candidate_audit.csv.gz")
    metadata = json.loads((RESULTS / "claude_agent_metadata.json").read_text(encoding="utf-8"))
    maximum = int(data["budget"].max())
    final = data[data["budget"].eq(maximum)]
    index = ["dataset", "block_start", "null_index"]
    contrasts = {}
    for name, left, right in (
        ("target_minus_sham", "claude_target", "claude_sham"),
        ("target_minus_blind", "claude_target", "claude_blind"),
        ("coarsened_minus_target", "claude_coarsened", "claude_target"),
    ):
        path_values = paired_contrast(
            final,
            index=index,
            column="validation_sharpe",
            left=left,
            right=right,
        )
        cell_values = path_values.groupby(level=[0, 1]).mean()
        contrasts[name] = mean_ci(cell_values)
        contrasts[name]["sign_flip_p"] = sign_flip_p(cell_values)
        era_values = cell_values.groupby(level="block_start").mean()
        contrasts[name]["era_sensitivity"] = mean_ci(era_values)
    for name, column in (
        ("target_minus_sham_confirmation", "confirmation_sharpe"),
        ("target_minus_sham_optimism", "optimism_gap"),
    ):
        path_values = paired_contrast(
            final,
            index=index,
            column=column,
            left="claude_target",
            right="claude_sham",
        )
        cell_values = path_values.groupby(level=[0, 1]).mean()
        contrasts[name] = mean_ci(cell_values)
        contrasts[name]["sign_flip_p"] = sign_flip_p(cell_values)
    fallback_flag = audit["fallback_reason"].fillna("").astype(str).ne("")
    fallback_rates = (
        audit.assign(fallback=fallback_flag).groupby("arm")["fallback"].mean().to_dict()
    )
    contaminated = (
        audit.assign(fallback=fallback_flag)
        .groupby(["pair_id", "arm"])["fallback"]
        .any()
        .unstack("arm")
    )
    eligible_pairs = contaminated.index[
        ~contaminated.get("target", False) & ~contaminated.get("sham", False)
    ]
    clean_final = final[final["pair_id"].isin(eligible_pairs)]
    clean_interval: dict[str, float] | None = None
    if not clean_final.empty:
        clean_paths = paired_contrast(
            clean_final,
            index=index,
            column="validation_sharpe",
            left="claude_target",
            right="claude_sham",
        )
        clean_cells = clean_paths.groupby(level=[0, 1]).mean()
        clean_interval = mean_ci(clean_cells)
        clean_interval["sign_flip_p"] = sign_flip_p(clean_cells)
    model_usage = {
        model_name: sum(
            int(call.get("model_usage", {}).get(model_name, {}).get("inputTokens", 0))
            + int(call.get("model_usage", {}).get(model_name, {}).get("outputTokens", 0))
            for call in metadata["calls"]
        )
        for model_name in {
            name for call in metadata["calls"] for name in call.get("model_usage", {})
        }
    }
    summary["agentic"] = {
        "rows": len(data),
        "episodes": int(data["episode_id"].nunique()),
        "candidate_trials": int(data["episode_id"].nunique() * maximum),
        "contrasts": contrasts,
        "candidate_fallback_rates": fallback_rates,
        "clean_target_minus_sham": clean_interval,
        "clean_pairs": int(len(eligible_pairs)),
        "calls": len(metadata["calls"]),
        "total_cost_usd": float(metadata["total_cost_usd"]),
        "model_token_totals": model_usage,
        "claude_cli_version": metadata["claude_cli_version"],
    }
    rows = []
    for policy in ("claude_blind", "claude_sham", "claude_target", "claude_coarsened"):
        subset = final[final["policy"].eq(policy)]
        rows.append(
            {
                "Arm": POLICY_LABELS[policy].replace("Claude: ", "").title(),
                "Validation Sharpe": subset["validation_sharpe"].mean(),
                "Confirmation Sharpe": subset["confirmation_sharpe"].mean(),
                "Bonf. (%)": 100.0 * subset["bonferroni_promote"].mean(),
                "Sealed (%)": 100.0 * subset["holdout_promote"].mean(),
            }
        )
    save_table(
        pd.DataFrame(rows),
        "agentic_table.tex",
        {
            "Validation Sharpe": "{:.3f}",
            "Confirmation Sharpe": "{:.3f}",
            "Bonf. (%)": "{:.1f}",
            "Sealed (%)": "{:.1f}",
        },
    )
    fig, axis = plt.subplots(figsize=(5.8, 3.0))
    for policy in ("claude_blind", "claude_sham", "claude_target", "claude_coarsened"):
        grouped = data[data["policy"].eq(policy)].groupby("budget")["validation_sharpe"]
        means = grouped.mean()
        errors = 1.96 * grouped.sem()
        axis.plot(means.index, means, marker="o", label=POLICY_LABELS[policy], color=COLORS[policy])
        axis.fill_between(
            means.index, means - errors, means + errors, color=COLORS[policy], alpha=0.10
        )
    axis.set_xlabel("Candidate budget")
    axis.set_ylabel("Selected validation Sharpe")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "agentic_trajectories.pdf", bbox_inches="tight")
    fig.savefig(OUT / "agentic_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def power_analysis(summary: dict[str, object]) -> None:
    path = RESULTS / "planted_signal_trials.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    maximum = int(data["budget"].max())
    final = data[data["budget"].eq(maximum)]
    rows = []
    for amplitude in sorted(final["signal_bps"].unique()):
        for policy in ("blind", "adaptive_sham", "adaptive_target"):
            subset = final[(final["signal_bps"].eq(amplitude)) & (final["policy"].eq(policy))]
            rows.append(
                {
                    "Signal (bp/day)": amplitude,
                    "Policy": POLICY_LABELS[policy],
                    "DSR raw N (%)": 100.0 * subset["dsr_promote"].mean(),
                    "Bonf. (%)": 100.0 * subset["bonferroni_promote"].mean(),
                    "Sealed (%)": 100.0 * subset["holdout_promote"].mean(),
                    "N-eff + sealed (%)": 100.0 * subset["dual_effective_gate_promote"].mean(),
                }
            )
    save_table(
        pd.DataFrame(rows),
        "power_table.tex",
        {
            "Signal (bp/day)": "{:.1f}",
            "DSR raw N (%)": "{:.1f}",
            "Bonf. (%)": "{:.1f}",
            "Sealed (%)": "{:.1f}",
            "N-eff + sealed (%)": "{:.1f}",
        },
    )
    summary["power"] = {
        "rows": len(data),
        "markets_per_signal": int(final.groupby("signal_bps")["market_seed"].nunique().iloc[0]),
        "candidate_trials": 562_500,
    }
    target = final[final["policy"].eq("adaptive_target")]
    fig, axis = plt.subplots(figsize=(5.8, 3.0))
    for gate, label, color in (
        ("dsr_promote", "DSR raw N", "#7C3AED"),
        ("bonferroni_promote", "Bonferroni", "#D97706"),
        ("holdout_promote", "Sealed", "#059669"),
        ("dual_effective_gate_promote", "$N_{eff}$ DSR + sealed", "#2563EB"),
    ):
        rates = 100.0 * target.groupby("signal_bps")[gate].mean()
        axis.plot(rates.index, rates, marker="o", label=label, color=color)
    axis.set_xlabel("Planted signal (bp per day)")
    axis.set_ylabel("Promotion probability (%)")
    axis.set_ylim(0, 102)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "power_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT / "power_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_macros(summary: dict[str, object]) -> None:
    syn = summary["synthetic"]
    contrasts = syn["contrasts"]
    gates = syn["gates"]["adaptive_target"]
    lines = [
        rf"\newcommand{{\SynValDiff}}{{{contrasts['validation_sharpe']['mean']:.3f}}}",
        rf"\newcommand{{\SynValLow}}{{{contrasts['validation_sharpe']['low']:.3f}}}",
        rf"\newcommand{{\SynValHigh}}{{{contrasts['validation_sharpe']['high']:.3f}}}",
        rf"\newcommand{{\SynConfDiff}}{{{contrasts['confirmation_sharpe']['mean']:.3f}}}",
        rf"\newcommand{{\SynConfLow}}{{{contrasts['confirmation_sharpe']['low']:.3f}}}",
        rf"\newcommand{{\SynConfHigh}}{{{contrasts['confirmation_sharpe']['high']:.3f}}}",
        rf"\newcommand{{\RawDSRTarget}}{{{100 * gates['dsr_promote']['mean']:.2f}\%}}",
        rf"\newcommand{{\EffDSRTarget}}{{{100 * gates['dsr_effective_promote']['mean']:.1f}\%}}",
        rf"\newcommand{{\BonfTarget}}{{{100 * gates['bonferroni_promote']['mean']:.1f}\%}}",
        rf"\newcommand{{\BHTarget}}{{{100 * gates['bh_promote']['mean']:.1f}\%}}",
        rf"\newcommand{{\HoldoutTarget}}{{{100 * gates['holdout_promote']['mean']:.2f}\%}}",
    ]
    if "empirical" in summary:
        empirical = summary["empirical"]
        for prefix, key in (
            ("EmpVal", "validation_contrast_cell_level"),
            ("EmpConf", "confirmation_contrast_cell_level"),
        ):
            interval = empirical[key]
            lines.extend(
                [
                    rf"\newcommand{{\{prefix}Diff}}{{{interval['mean']:.3f}}}",
                    rf"\newcommand{{\{prefix}Low}}{{{interval['low']:.3f}}}",
                    rf"\newcommand{{\{prefix}High}}{{{interval['high']:.3f}}}",
                ]
            )
        era_interval = empirical["validation_contrast_era_sensitivity"]
        empirical_gates = empirical["target_gate_rates"]
        lines.extend(
            [
                rf"\newcommand{{\EmpValEraLow}}{{{era_interval['low']:.3f}}}",
                rf"\newcommand{{\EmpValEraHigh}}{{{era_interval['high']:.3f}}}",
                rf"\newcommand{{\EmpRawDSRTarget}}{{{100 * empirical_gates['dsr_promote']:.2f}\%}}",
                rf"\newcommand{{\EmpEffDSRTarget}}"
                rf"{{{100 * empirical_gates['dsr_effective_promote']:.2f}\%}}",
                rf"\newcommand{{\EmpBonfTarget}}"
                rf"{{{100 * empirical_gates['bonferroni_promote']:.2f}\%}}",
                rf"\newcommand{{\EmpHoldoutTarget}}"
                rf"{{{100 * empirical_gates['holdout_promote']:.2f}\%}}",
                rf"\newcommand{{\EmpValP}}"
                rf"{{{latex_probability(empirical['validation_contrast_cell_level']['sign_flip_p'])}}}",
                rf"\newcommand{{\EmpConfP}}"
                rf"{{{latex_probability(empirical['confirmation_contrast_cell_level']['sign_flip_p'])}}}",
            ]
        )
    if "agentic" in summary:
        for prefix, key in (
            ("AgentTS", "target_minus_sham"),
            ("AgentTB", "target_minus_blind"),
            ("AgentCT", "coarsened_minus_target"),
            ("AgentConfTS", "target_minus_sham_confirmation"),
            ("AgentOptTS", "target_minus_sham_optimism"),
        ):
            interval = summary["agentic"]["contrasts"][key]
            lines.extend(
                [
                    rf"\newcommand{{\{prefix}Diff}}{{{interval['mean']:.3f}}}",
                    rf"\newcommand{{\{prefix}Low}}{{{interval['low']:.3f}}}",
                    rf"\newcommand{{\{prefix}High}}{{{interval['high']:.3f}}}",
                ]
            )
        target_fallback = 100 * summary["agentic"]["candidate_fallback_rates"]["target"]
        sham_fallback = 100 * summary["agentic"]["candidate_fallback_rates"]["sham"]
        lines.extend(
            [
                rf"\newcommand{{\AgentFallbackTarget}}{{{target_fallback:.1f}\%}}",
                rf"\newcommand{{\AgentFallbackSham}}{{{sham_fallback:.1f}\%}}",
                rf"\newcommand{{\AgentCost}}{{\${summary['agentic']['total_cost_usd']:.2f}}}",
                rf"\newcommand{{\AgentCalls}}{{{summary['agentic']['calls']}}}",
            ]
        )
        target_sham_era = summary["agentic"]["contrasts"]["target_minus_sham"][
            "era_sensitivity"
        ]
        clean = summary["agentic"]["clean_target_minus_sham"]
        if clean is not None:
            lines.extend(
                [
                    rf"\newcommand{{\AgentTSEraLow}}{{{target_sham_era['low']:.3f}}}",
                    rf"\newcommand{{\AgentTSEraHigh}}{{{target_sham_era['high']:.3f}}}",
                    rf"\newcommand{{\AgentCleanDiff}}{{{clean['mean']:.3f}}}",
                    rf"\newcommand{{\AgentCleanLow}}{{{clean['low']:.3f}}}",
                    rf"\newcommand{{\AgentCleanHigh}}{{{clean['high']:.3f}}}",
                    rf"\newcommand{{\AgentCleanPairs}}{{{summary['agentic']['clean_pairs']}}}",
                    rf"\newcommand{{\AgentTSP}}{{{summary['agentic']['contrasts']['target_minus_sham']['sign_flip_p']:.4f}}}",
                    rf"\newcommand{{\AgentConfTSP}}{{{summary['agentic']['contrasts']['target_minus_sham_confirmation']['sign_flip_p']:.3f}}}",
                    rf"\newcommand{{\AgentCleanP}}{{{clean['sign_flip_p']:.3f}}}",
                ]
            )
    (OUT / "deep_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_candidates() -> list[Path]:
    source_files = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "configs" / "deep_paper.toml",
        *sorted((ROOT / "paper").glob("*.md")),
        ROOT / "paper" / "deep_main.tex",
        ROOT / "paper" / "references.bib",
        ROOT / "scripts" / "analyze_deep.py",
        *sorted((ROOT / "src" / "alpha_firewall").glob("*.py")),
        *sorted((ROOT / "tests").glob("test_*.py")),
    ]
    transcript_files = sorted((ROOT / "artifacts" / "agentic" / "transcripts").glob("*.json"))
    result_files = [
        RESULTS / "synthetic_deep_paper_trials.csv",
        RESULTS / "synthetic_deep_paper_candidate_audit.csv.gz",
        RESULTS / "synthetic_deep_paper_metadata.json",
        RESULTS / "empirical_null_trials.csv",
        RESULTS / "empirical_null_candidate_audit.csv.gz",
        RESULTS / "empirical_null_metadata.json",
        RESULTS / "planted_signal_trials.csv",
        RESULTS / "planted_signal_candidate_audit.csv.gz",
        RESULTS / "planted_signal_metadata.json",
        RESULTS / "claude_agent_trials.csv",
        RESULTS / "claude_agent_candidate_audit.csv.gz",
        RESULTS / "claude_agent_metadata.json",
    ]
    deterministic_oracles = [
        OUT / "deep_summary.json",
        OUT / "deep_macros.tex",
        OUT / "synthetic_gate_table.tex",
        OUT / "empirical_table.tex",
        OUT / "agentic_table.tex",
        OUT / "power_table.tex",
        ROOT / "artifacts" / "reproduction" / "independent_headlines.json",
        ROOT
        / "artifacts"
        / "reproduction"
        / "synthetic_generator_spotcheck.json",
    ]
    return source_files + transcript_files + result_files + deterministic_oracles


def write_manifest(path: Path) -> None:
    manifest = {
        str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in manifest_candidates()
        if path.exists()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate P1 analysis assets")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=RESULTS / "deep_artifact_manifest.json",
    )
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    global RESULTS, OUT
    args = parse_args()
    RESULTS = args.results_dir.resolve()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        write_manifest(args.manifest_output.resolve())
        return
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    summary: dict[str, object] = {}
    synthetic_analysis(summary)
    empirical_analysis(summary)
    agentic_analysis(summary)
    power_analysis(summary)
    summary["total_candidate_trials"] = int(
        sum(
            section.get("candidate_trials", 0)
            for section in summary.values()
            if isinstance(section, dict)
        )
    )
    write_macros(summary)
    if not args.skip_manifest:
        write_manifest(args.manifest_output.resolve())
    (OUT / "deep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
