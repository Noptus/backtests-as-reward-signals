from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from scipy.stats import t


class HeadlineVerificationError(RuntimeError):
    """Raised when a frozen headline cannot be independently reconstructed."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: str, *, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise HeadlineVerificationError(f"non-finite {field}: {value!r}")
    return parsed


def _mean_interval(values: Iterable[float], *, level: float = 0.95) -> dict[str, float]:
    sample = [float(value) for value in values]
    if len(sample) < 2 or not all(math.isfinite(value) for value in sample):
        raise HeadlineVerificationError("interval requires at least two finite observations")
    center = math.fsum(sample) / len(sample)
    squared = math.fsum((value - center) ** 2 for value in sample)
    standard_error = math.sqrt(squared / (len(sample) - 1)) / math.sqrt(len(sample))
    half_width = float(t.ppf((1.0 + level) / 2.0, len(sample) - 1)) * standard_error
    return {"mean": center, "low": center - half_width, "high": center + half_width}


def _exact_sign_flip_p(values: Iterable[float]) -> float:
    sample = tuple(float(value) for value in values)
    if not sample or len(sample) > 20:
        raise HeadlineVerificationError("exact sign-flip verification is limited to 1--20 units")
    observed = abs(math.fsum(sample) / len(sample))
    exceedances = 0
    for assignment in range(1 << len(sample)):
        randomized = math.fsum(
            value if assignment & (1 << index) else -value
            for index, value in enumerate(sample)
        ) / len(sample)
        exceedances += abs(randomized) >= observed - 1e-15
    return exceedances / (1 << len(sample))


def _max_budget_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    budgets = {int(row["budget"]) for row in rows}
    if not budgets:
        raise HeadlineVerificationError("trial file is empty")
    maximum = max(budgets)
    return [row for row in rows if int(row["budget"]) == maximum]


def _paired_values(
    rows: list[dict[str, str]],
    *,
    keys: tuple[str, ...],
    column: str,
    left: str,
    right: str,
) -> dict[tuple[str, ...], float]:
    cells: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        policy = row["policy"]
        if policy not in {left, right}:
            continue
        key = tuple(row[name] for name in keys)
        if policy in cells[key]:
            raise HeadlineVerificationError(f"duplicate {policy} observation for {key}")
        cells[key][policy] = _finite_float(row[column], field=column)
    incomplete = [key for key, value in cells.items() if set(value) != {left, right}]
    if incomplete:
        raise HeadlineVerificationError(f"unpaired observations for {incomplete[:3]}")
    return {key: value[left] - value[right] for key, value in cells.items()}


def _group_means(
    values: dict[tuple[str, ...], float], *, key_indexes: tuple[int, ...]
) -> dict[tuple[str, ...], float]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for key, value in values.items():
        groups[tuple(key[index] for index in key_indexes)].append(value)
    return {key: math.fsum(group) / len(group) for key, group in groups.items()}


def _parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise HeadlineVerificationError(f"ambiguous boolean value: {value!r}")


def _rate(rows: list[dict[str, str]], field: str) -> float:
    values = [_parse_boolean(row[field]) for row in rows]
    if not values:
        raise HeadlineVerificationError(f"no observations for rate {field}")
    return math.fsum(values) / len(values)


def _lookup(value: dict[str, Any], path: str) -> float:
    current: Any = value
    for component in path.split("."):
        current = current[component]
    return float(current)


def _comparison(
    checks: dict[str, float], oracle: dict[str, Any], paths: dict[str, str], *, tolerance: float
) -> list[dict[str, Any]]:
    comparisons = []
    for name, actual in checks.items():
        expected = _lookup(oracle, paths[name])
        difference = abs(actual - expected)
        comparisons.append(
            {
                "name": name,
                "actual": actual,
                "expected": expected,
                "absolute_difference": difference,
                "tolerance": tolerance,
                "status": "PASS" if difference <= tolerance else "FAIL",
            }
        )
    return comparisons


def verify_headlines(
    *, results_dir: Path, oracle_path: Path, tolerance: float = 1e-10
) -> dict[str, Any]:
    """Recompute claim-critical P1 statistics without the manuscript analysis code."""

    synthetic = _max_budget_rows(_read_csv(results_dir / "synthetic_deep_paper_trials.csv"))
    empirical = _max_budget_rows(_read_csv(results_dir / "empirical_null_trials.csv"))
    agent = _max_budget_rows(_read_csv(results_dir / "claude_agent_trials.csv"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))

    synthetic_validation = _paired_values(
        synthetic,
        keys=("market_seed",),
        column="validation_sharpe",
        left="adaptive_target",
        right="adaptive_sham",
    )
    synthetic_confirmation = _paired_values(
        synthetic,
        keys=("market_seed",),
        column="confirmation_sharpe",
        left="adaptive_target",
        right="adaptive_sham",
    )
    if len(synthetic_validation) != 2_000 or len(synthetic_confirmation) != 2_000:
        raise HeadlineVerificationError("synthetic contrasts must contain 2,000 paired markets")
    synthetic_validation_interval = _mean_interval(synthetic_validation.values())
    synthetic_confirmation_interval = _mean_interval(synthetic_confirmation.values())
    synthetic_confirmation_tost_interval = _mean_interval(
        synthetic_confirmation.values(), level=0.90
    )

    empirical_paths = _paired_values(
        empirical,
        keys=("dataset", "block_start", "null_index"),
        column="validation_sharpe",
        left="adaptive_target",
        right="adaptive_sham",
    )
    empirical_cells = _group_means(empirical_paths, key_indexes=(0, 1))
    empirical_cell_sizes: dict[tuple[str, str], int] = defaultdict(int)
    for dataset, block_start, _null_index in empirical_paths:
        empirical_cell_sizes[(dataset, block_start)] += 1
    if len(empirical_paths) != 300 or len(empirical_cells) != 15:
        raise HeadlineVerificationError("empirical analysis requires 300 paths and 15 cells")
    if set(empirical_cell_sizes.values()) != {20}:
        raise HeadlineVerificationError("each empirical cell must average exactly 20 paths")
    empirical_interval = _mean_interval(empirical_cells.values())
    empirical_eras = _group_means(empirical_cells, key_indexes=(1,))
    if len(empirical_eras) != 3:
        raise HeadlineVerificationError("empirical sensitivity requires exactly three eras")
    empirical_era_interval = _mean_interval(empirical_eras.values())

    agent_validation_paths = _paired_values(
        agent,
        keys=("dataset", "block_start", "null_index"),
        column="validation_sharpe",
        left="claude_target",
        right="claude_sham",
    )
    agent_confirmation_paths = _paired_values(
        agent,
        keys=("dataset", "block_start", "null_index"),
        column="confirmation_sharpe",
        left="claude_target",
        right="claude_sham",
    )
    agent_cells = _group_means(agent_validation_paths, key_indexes=(0, 1))
    agent_confirmation_cells = _group_means(agent_confirmation_paths, key_indexes=(0, 1))
    agent_cell_sizes: dict[tuple[str, str], int] = defaultdict(int)
    for dataset, block_start, _null_index in agent_validation_paths:
        agent_cell_sizes[(dataset, block_start)] += 1
    if len(agent_validation_paths) != 60 or len(agent_cells) != 15:
        raise HeadlineVerificationError("agent analysis requires 60 paths and 15 cells")
    if set(agent_cell_sizes.values()) != {4}:
        raise HeadlineVerificationError("each agent cell must average exactly four quartets")
    agent_interval = _mean_interval(agent_cells.values())
    agent_confirmation_interval = _mean_interval(agent_confirmation_cells.values())

    target_synthetic = [row for row in synthetic if row["policy"] == "adaptive_target"]
    target_empirical = [row for row in empirical if row["policy"] == "adaptive_target"]
    checks = {
        "synthetic_validation_mean": synthetic_validation_interval["mean"],
        "synthetic_validation_low": synthetic_validation_interval["low"],
        "synthetic_validation_high": synthetic_validation_interval["high"],
        "synthetic_confirmation_mean": synthetic_confirmation_interval["mean"],
        "synthetic_confirmation_low": synthetic_confirmation_interval["low"],
        "synthetic_confirmation_high": synthetic_confirmation_interval["high"],
        "synthetic_raw_dsr_rate": _rate(target_synthetic, "dsr_promote"),
        "synthetic_effective_dsr_rate": _rate(target_synthetic, "dsr_effective_promote"),
        "synthetic_bonferroni_rate": _rate(target_synthetic, "bonferroni_promote"),
        "synthetic_bh_rate": _rate(target_synthetic, "bh_promote"),
        "synthetic_sealed_rate": _rate(target_synthetic, "holdout_promote"),
        "empirical_validation_mean": empirical_interval["mean"],
        "empirical_validation_low": empirical_interval["low"],
        "empirical_validation_high": empirical_interval["high"],
        "empirical_era_low": empirical_era_interval["low"],
        "empirical_era_high": empirical_era_interval["high"],
        "empirical_sealed_rate": _rate(target_empirical, "holdout_promote"),
        "agent_validation_mean": agent_interval["mean"],
        "agent_validation_low": agent_interval["low"],
        "agent_validation_high": agent_interval["high"],
        "agent_confirmation_mean": agent_confirmation_interval["mean"],
        "agent_confirmation_low": agent_confirmation_interval["low"],
        "agent_confirmation_high": agent_confirmation_interval["high"],
    }
    paths = {
        "synthetic_validation_mean": "synthetic.contrasts.validation_sharpe.mean",
        "synthetic_validation_low": "synthetic.contrasts.validation_sharpe.low",
        "synthetic_validation_high": "synthetic.contrasts.validation_sharpe.high",
        "synthetic_confirmation_mean": "synthetic.contrasts.confirmation_sharpe.mean",
        "synthetic_confirmation_low": "synthetic.contrasts.confirmation_sharpe.low",
        "synthetic_confirmation_high": "synthetic.contrasts.confirmation_sharpe.high",
        "synthetic_raw_dsr_rate": "synthetic.gates.adaptive_target.dsr_promote.mean",
        "synthetic_effective_dsr_rate": (
            "synthetic.gates.adaptive_target.dsr_effective_promote.mean"
        ),
        "synthetic_bonferroni_rate": "synthetic.gates.adaptive_target.bonferroni_promote.mean",
        "synthetic_bh_rate": "synthetic.gates.adaptive_target.bh_promote.mean",
        "synthetic_sealed_rate": "synthetic.gates.adaptive_target.holdout_promote.mean",
        "empirical_validation_mean": "empirical.validation_contrast_cell_level.mean",
        "empirical_validation_low": "empirical.validation_contrast_cell_level.low",
        "empirical_validation_high": "empirical.validation_contrast_cell_level.high",
        "empirical_era_low": "empirical.validation_contrast_era_sensitivity.low",
        "empirical_era_high": "empirical.validation_contrast_era_sensitivity.high",
        "empirical_sealed_rate": "empirical.target_gate_rates.holdout_promote",
        "agent_validation_mean": "agentic.contrasts.target_minus_sham.mean",
        "agent_validation_low": "agentic.contrasts.target_minus_sham.low",
        "agent_validation_high": "agentic.contrasts.target_minus_sham.high",
        "agent_confirmation_mean": (
            "agentic.contrasts.target_minus_sham_confirmation.mean"
        ),
        "agent_confirmation_low": "agentic.contrasts.target_minus_sham_confirmation.low",
        "agent_confirmation_high": "agentic.contrasts.target_minus_sham_confirmation.high",
    }
    comparisons = _comparison(checks, oracle, paths, tolerance=tolerance)
    failures = [row["name"] for row in comparisons if row["status"] != "PASS"]
    equivalence_pass = (
        synthetic_confirmation_tost_interval["low"] > -0.10
        and synthetic_confirmation_tost_interval["high"] < 0.10
    )
    if not equivalence_pass:
        failures.append("synthetic_confirmation_90pct_equivalence_band")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "scope": "same-data independent implementation; not an independent replication",
        "implementation_boundary": (
            "stdlib CSV/math plus scipy t quantiles; no import or call to scripts/analyze_deep.py"
        ),
        "unit_counts": {
            "synthetic_markets": len(synthetic_validation),
            "empirical_paths": len(empirical_paths),
            "empirical_cells": len(empirical_cells),
            "empirical_paths_per_cell": sorted(set(empirical_cell_sizes.values())),
            "empirical_eras": len(empirical_eras),
            "agent_paths": len(agent_validation_paths),
            "agent_cells": len(agent_cells),
            "agent_paths_per_cell": sorted(set(agent_cell_sizes.values())),
        },
        "derived_checks": {
            "synthetic_confirmation_90pct_interval": synthetic_confirmation_tost_interval,
            "synthetic_confirmation_inside_equivalence_band": equivalence_pass,
            "empirical_validation_exact_sign_flip_p": _exact_sign_flip_p(
                empirical_cells.values()
            ),
            "agent_validation_exact_sign_flip_p": _exact_sign_flip_p(agent_cells.values()),
        },
        "human_review_values": {
            "empirical_validation_cells": [
                {"dataset": key[0], "block_start": int(key[1]), "contrast": value}
                for key, value in sorted(empirical_cells.items())
            ],
            "empirical_validation_eras": [
                {"block_start": int(key[0]), "contrast": value}
                for key, value in sorted(empirical_eras.items())
            ],
            "agent_validation_cells": [
                {"dataset": key[0], "block_start": int(key[1]), "contrast": value}
                for key, value in sorted(agent_cells.items())
            ],
        },
        "comparisons": comparisons,
        "failures": failures,
    }
    if failures:
        raise HeadlineVerificationError(f"headline verification failed: {', '.join(failures)}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent same-data P1 headline verifier")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_headlines(
        results_dir=args.results_dir.resolve(),
        oracle_path=args.oracle.resolve(),
        tolerance=args.tolerance,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
