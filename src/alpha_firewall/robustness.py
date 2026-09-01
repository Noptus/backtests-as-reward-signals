from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from alpha_firewall.data import simulate_null_returns
from alpha_firewall.experiment import evaluate_trials
from alpha_firewall.search import SearchPolicy, run_search

SweepTask = tuple[int, float, int, int, int, int, int, float, int]


def run_adaptivity_sweep(
    *,
    seeds: int,
    exploit_rates: list[float],
    elite_sizes: list[int],
    n_assets: int,
    validation_days: int,
    confirmation_days: int,
    budget: int,
    cost_bps: float,
    base_seed: int,
    workers: int = 1,
) -> pd.DataFrame:
    tasks: list[SweepTask] = [
        (
            seed_offset,
            exploit_rate,
            elite_size,
            n_assets,
            validation_days,
            confirmation_days,
            budget,
            cost_bps,
            base_seed,
        )
        for exploit_rate in exploit_rates
        for elite_size in elite_sizes
        for seed_offset in range(seeds)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(_run_sweep_task, tasks))
    else:
        nested = [_run_sweep_task(task) for task in tasks]
    return pd.DataFrame([row for task_rows in nested for row in task_rows])


def _run_sweep_task(task: SweepTask) -> list[dict[str, object]]:
    (
        seed_offset,
        exploit_rate,
        elite_size,
        n_assets,
        validation_days,
        confirmation_days,
        budget,
        cost_bps,
        base_seed,
    ) = task
    total_days = validation_days + confirmation_days + 504
    market_seed = base_seed + seed_offset
    target = simulate_null_returns(total_days, n_assets, market_seed)
    sham = simulate_null_returns(total_days, n_assets, market_seed + 1_000_000)
    validation = target.iloc[: validation_days + 252]
    confirmation = target.iloc[validation_days + 252 :]
    sham_validation = sham.iloc[: validation_days + 252]
    search_seed = base_seed * 100_000 + seed_offset * 10 + 1
    rows: list[dict[str, object]] = []

    for policy in ("adaptive_sham", "adaptive_target"):
        typed_policy: SearchPolicy = policy
        trials = run_search(
            validation,
            budget=budget,
            policy=typed_policy,
            seed=search_seed,
            cost_bps=cost_bps,
            feedback_returns=sham_validation if policy == "adaptive_sham" else None,
            exploit_rate=exploit_rate,
            max_elite=elite_size,
        )
        result = evaluate_trials(
            trials,
            confirmation,
            budget=budget,
            policy=typed_policy,
            search_seed=search_seed,
            market_seed=market_seed,
            cost_bps=cost_bps,
        ).to_dict()
        result.update({"exploit_rate": exploit_rate, "elite_size": elite_size})
        rows.append(result)
    return rows
