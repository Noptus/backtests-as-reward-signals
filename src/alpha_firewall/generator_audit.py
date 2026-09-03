"""Regenerate three frozen synthetic searches and compare them with the archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_firewall.data import simulate_null_returns
from alpha_firewall.experiment import evaluate_trials
from alpha_firewall.search import SearchPolicy, run_search

AUDIT_SEEDS = (50_000, 50_999, 51_999)
CANDIDATE_KEYS = ("market_seed", "policy", "trial_index")
RESULT_KEYS = ("market_seed", "policy", "budget")
CANDIDATE_FLOAT_FIELDS = ("validation_sharpe", "feedback_sharpe")
RESULT_FLOAT_FIELDS = (
    "effective_trials",
    "validation_sharpe",
    "confirmation_sharpe",
    "optimism_gap",
    "validation_psr",
    "validation_dsr",
    "validation_dsr_effective",
    "validation_hac_probability",
    "confirmation_psr",
    "confirmation_hac_probability",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regenerate_seed(
    *,
    market_seed: int,
    base_seed: int,
    n_assets: int,
    validation_days: int,
    confirmation_days: int,
    budgets: tuple[int, ...],
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild one market without calling the experiment batch wrapper."""
    seed_offset = market_seed - base_seed
    if seed_offset < 0:
        raise ValueError("market_seed precedes the configured base seed")
    total_days = validation_days + confirmation_days + 504
    returns = simulate_null_returns(total_days, n_assets, market_seed)
    validation = returns.iloc[: validation_days + 252]
    confirmation = returns.iloc[validation_days + 252 :]
    sham_validation = simulate_null_returns(
        total_days, n_assets, market_seed + 1_000_000
    ).iloc[: validation_days + 252]
    policies: tuple[SearchPolicy, ...] = (
        "blind",
        "adaptive_sham",
        "adaptive_target",
    )
    candidate_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    for policy_index, policy in enumerate(policies):
        adaptive_offset = 1 if policy.startswith("adaptive") else policy_index
        search_seed = base_seed * 100_000 + seed_offset * 10 + adaptive_offset
        trials = run_search(
            validation,
            budget=max(budgets),
            policy=policy,
            seed=search_seed,
            cost_bps=cost_bps,
            feedback_returns=sham_validation if policy == "adaptive_sham" else None,
        )
        for trial in trials:
            candidate_rows.append(
                {
                    "market_seed": market_seed,
                    "search_seed": search_seed,
                    "policy": policy,
                    "cost_bps": cost_bps,
                    "trial_index": trial.trial_index,
                    **trial.spec.to_dict(),
                    "strategy_id": trial.spec.identifier,
                    "validation_sharpe": trial.validation_sharpe,
                    "feedback_sharpe": trial.feedback_sharpe,
                }
            )
        for budget in budgets:
            result_rows.append(
                evaluate_trials(
                    trials,
                    confirmation,
                    budget=budget,
                    policy=policy,
                    search_seed=search_seed,
                    market_seed=market_seed,
                    cost_bps=cost_bps,
                ).to_dict()
            )
    return pd.DataFrame(candidate_rows), pd.DataFrame(result_rows)


def compare_frames(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    float_fields: tuple[str, ...],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Compare exact discrete fields and tightly tolerant float64 fields."""
    if set(expected.columns) != set(actual.columns):
        missing = sorted(set(expected.columns) - set(actual.columns))
        extra = sorted(set(actual.columns) - set(expected.columns))
        raise AssertionError(f"column mismatch: missing={missing}, extra={extra}")
    left = expected.sort_values(list(keys)).reset_index(drop=True)
    right = actual.sort_values(list(keys)).reset_index(drop=True)
    if len(left) != len(right):
        raise AssertionError(f"row-count mismatch: expected {len(left)}, got {len(right)}")

    float_set = set(float_fields)
    exact_fields = [field for field in left.columns if field not in float_set]
    mismatches: list[str] = []
    for field in exact_fields:
        left_values = left[field].tolist()
        right_values = right[field].tolist()
        if left_values != right_values:
            first = next(
                index
                for index, (expected_value, actual_value) in enumerate(
                    zip(left_values, right_values, strict=True)
                )
                if expected_value != actual_value
            )
            mismatches.append(
                f"{field} differs at sorted row {first}: "
                f"expected {left_values[first]!r}, got {right_values[first]!r}"
            )

    maximum_absolute_differences: dict[str, float] = {}
    for field in float_fields:
        expected_values = left[field].to_numpy(dtype=float)
        actual_values = right[field].to_numpy(dtype=float)
        finite_difference = np.abs(expected_values - actual_values)
        maximum_absolute_differences[field] = float(
            np.nanmax(finite_difference) if finite_difference.size else 0.0
        )
        if not np.allclose(
            expected_values,
            actual_values,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        ):
            mismatches.append(f"{field} exceeds atol={atol}, rtol={rtol}")
    if mismatches:
        raise AssertionError("; ".join(mismatches))
    return {
        "rows": len(left),
        "exact_fields": exact_fields,
        "float_fields": list(float_fields),
        "maximum_absolute_differences": maximum_absolute_differences,
    }


def verify_prefix_selection(candidates: pd.DataFrame, results: pd.DataFrame) -> None:
    for row in results.itertuples(index=False):
        prefix = candidates.loc[
            (candidates["market_seed"] == row.market_seed)
            & (candidates["policy"] == row.policy)
            & (candidates["trial_index"] < row.budget)
        ].sort_values("trial_index")
        if len(prefix) != row.budget:
            raise AssertionError("archive does not contain an exact search prefix")
        selected = prefix.loc[prefix["validation_sharpe"].idxmax(), "strategy_id"]
        if selected != row.selected_spec:
            raise AssertionError("selected_spec is not the first validation argmax")


def run_audit(
    *,
    config_path: Path,
    candidate_path: Path,
    result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)["experiment"]
    base_seed = int(config["base_seed"])
    total_seeds = int(config["seeds"])
    budgets = tuple(int(value) for value in config["budgets"])
    if base_seed != AUDIT_SEEDS[0] or base_seed + total_seeds - 1 != AUDIT_SEEDS[-1]:
        raise ValueError("audit seeds no longer equal the first, midpoint, and last market")
    if budgets != (1, 10, 25, 50, 100, 250):
        raise ValueError("audit requires the frozen six-prefix budget schedule")

    regenerated_candidates = []
    regenerated_results = []
    for market_seed in AUDIT_SEEDS:
        candidates, results = regenerate_seed(
            market_seed=market_seed,
            base_seed=base_seed,
            n_assets=int(config["n_assets"]),
            validation_days=int(config["validation_days"]),
            confirmation_days=int(config["confirmation_days"]),
            budgets=budgets,
            cost_bps=float(config["cost_bps"]),
        )
        regenerated_candidates.append(candidates)
        regenerated_results.append(results)
    actual_candidates = pd.concat(regenerated_candidates, ignore_index=True)
    actual_results = pd.concat(regenerated_results, ignore_index=True)

    archived_candidates = pd.read_csv(candidate_path)
    archived_results = pd.read_csv(result_path)
    expected_candidates = archived_candidates.loc[
        archived_candidates["market_seed"].isin(AUDIT_SEEDS)
    ].copy()
    expected_results = archived_results.loc[
        archived_results["market_seed"].isin(AUDIT_SEEDS)
    ].copy()
    atol = rtol = 1e-12
    candidate_comparison = compare_frames(
        expected_candidates,
        actual_candidates,
        keys=CANDIDATE_KEYS,
        float_fields=CANDIDATE_FLOAT_FIELDS,
        atol=atol,
        rtol=rtol,
    )
    result_comparison = compare_frames(
        expected_results,
        actual_results,
        keys=RESULT_KEYS,
        float_fields=RESULT_FLOAT_FIELDS,
        atol=atol,
        rtol=rtol,
    )
    verify_prefix_selection(expected_candidates, expected_results)
    report = {
        "status": "PASS",
        "classification": "post-release bounded provenance audit; not replication",
        "seeds": list(AUDIT_SEEDS),
        "selection_rule": "first, midpoint, and last of 2,000 consecutive markets",
        "markets_regenerated": 3,
        "markets_not_regenerated": total_seeds - 3,
        "candidate_comparison": candidate_comparison,
        "result_comparison": result_comparison,
        "structural_checks": {
            "all_six_budgets_are_prefixes": True,
            "selected_spec_is_first_validation_argmax": True,
        },
        "tolerance": {"absolute": atol, "relative": rtol},
        "inputs": {
            "config": {
                "path": "configs/deep_paper.toml",
                "sha256": sha256(config_path),
            },
            "candidate_archive": {
                "path": "artifacts/results/synthetic_deep_paper_candidate_audit.csv.gz",
                "sha256": sha256(candidate_path),
            },
            "result_archive": {
                "path": "artifacts/results/synthetic_deep_paper_trials.csv",
                "sha256": sha256(result_path),
            },
        },
        "hosted_model_calls": 0,
        "network_downloads": 0,
        "claim_boundary": (
            "Confirms code-to-archive determinism for three fixed markets only; "
            "does not validate the method or independently replicate results."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/deep_paper.toml")
    parser.add_argument(
        "--candidate-audit",
        type=Path,
        default=root / "artifacts/results/synthetic_deep_paper_candidate_audit.csv.gz",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=root / "artifacts/results/synthetic_deep_paper_trials.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts/reproduction/synthetic_generator_spotcheck.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(
                config_path=args.config,
                candidate_path=args.candidate_audit,
                result_path=args.results,
                output_path=args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
