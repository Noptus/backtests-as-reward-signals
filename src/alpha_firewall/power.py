from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from alpha_firewall.data import simulate_null_returns
from alpha_firewall.experiment import evaluate_trials
from alpha_firewall.search import SearchPolicy, run_search


def simulate_planted_momentum_returns(
    n_days: int,
    n_assets: int,
    seed: int,
    *,
    signal_bps: float,
    lookback: int = 21,
) -> pd.DataFrame:
    """Add a known lagged cross-sectional momentum premium to the null process."""

    base = simulate_null_returns(n_days, n_assets, seed)
    if signal_bps == 0.0:
        return base
    lagged = base.rolling(lookback, min_periods=lookback).sum().shift(1)
    centered = lagged.sub(lagged.mean(axis=1), axis=0)
    scale = centered.std(axis=1, ddof=0).replace(0.0, np.nan)
    standardized = centered.div(scale, axis=0).fillna(0.0)
    return base + signal_bps * 1e-4 * standardized


def run_power_experiment(
    *,
    seeds: int,
    signal_bps: list[float],
    n_assets: int,
    validation_days: int,
    confirmation_days: int,
    budgets: list[int],
    cost_bps: float,
    base_seed: int,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [
        (
            signal_index,
            amplitude,
            seed_offset,
            n_assets,
            validation_days,
            confirmation_days,
            budgets,
            cost_bps,
            base_seed,
        )
        for signal_index, amplitude in enumerate(signal_bps)
        for seed_offset in range(seeds)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(_run_power_seed, tasks))
    else:
        nested = [_run_power_seed(task) for task in tasks]
    result_rows = [row for seed_rows, _ in nested for row in seed_rows]
    audit_rows = [row for _, seed_audit in nested for row in seed_audit]
    return pd.DataFrame(result_rows), pd.DataFrame(audit_rows)


def _run_power_seed(
    task: tuple[int, float, int, int, int, int, list[int], float, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    (
        signal_index,
        amplitude,
        seed_offset,
        n_assets,
        validation_days,
        confirmation_days,
        budgets,
        cost_bps,
        base_seed,
    ) = task
    total_days = validation_days + confirmation_days + 504
    market_seed = base_seed + signal_index * 1_000_000 + seed_offset
    returns = simulate_planted_momentum_returns(
        total_days,
        n_assets,
        market_seed,
        signal_bps=amplitude,
    )
    validation = returns.iloc[: validation_days + 252]
    confirmation = returns.iloc[validation_days + 252 :]
    development = simulate_planted_momentum_returns(
        total_days,
        n_assets,
        market_seed + 10_000_000,
        signal_bps=amplitude,
    ).iloc[: validation_days + 252]
    policies: tuple[SearchPolicy, ...] = ("blind", "adaptive_sham", "adaptive_target")
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for policy_index, policy in enumerate(policies):
        adaptive_seed_offset = 1 if policy.startswith("adaptive") else policy_index
        search_seed = base_seed * 100_000 + market_seed * 10 + adaptive_seed_offset
        trials = run_search(
            validation,
            budget=max(budgets),
            policy=policy,
            seed=search_seed,
            cost_bps=cost_bps,
            feedback_returns=development if policy == "adaptive_sham" else None,
        )
        for trial in trials:
            audit_rows.append(
                {
                    "signal_bps": amplitude,
                    "market_seed": market_seed,
                    "search_seed": search_seed,
                    "policy": policy,
                    "trial_index": trial.trial_index,
                    **trial.spec.to_dict(),
                    "strategy_id": trial.spec.identifier,
                    "validation_sharpe": trial.validation_sharpe,
                    "feedback_sharpe": trial.feedback_sharpe,
                }
            )
        for budget in budgets:
            row = evaluate_trials(
                trials,
                confirmation,
                budget=budget,
                policy=policy,
                search_seed=search_seed,
                market_seed=market_seed,
                cost_bps=cost_bps,
            ).to_dict()
            row["signal_bps"] = amplitude
            rows.append(row)
    return rows, audit_rows
