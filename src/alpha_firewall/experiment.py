from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from alpha_firewall.backtest import run_backtest
from alpha_firewall.data import rademacher_null_returns, simulate_null_returns
from alpha_firewall.metrics import (
    deflated_sharpe_ratio,
    effective_trial_count,
    hac_positive_mean_probability,
    probabilistic_sharpe_ratio,
)
from alpha_firewall.search import SearchPolicy, Trial, run_search


@dataclass(frozen=True, slots=True)
class ExperimentRow:
    market_seed: int
    search_seed: int
    policy: str
    budget: int
    cost_bps: float
    effective_trials: float
    selected_spec: str
    validation_sharpe: float
    confirmation_sharpe: float
    optimism_gap: float
    validation_psr: float
    validation_dsr: float
    validation_dsr_effective: float
    validation_hac_probability: float
    confirmation_psr: float
    confirmation_hac_probability: float
    naive_promote: bool
    dsr_promote: bool
    dsr_effective_promote: bool
    bonferroni_promote: bool
    bh_promote: bool
    holdout_promote: bool
    dual_gate_promote: bool
    dual_effective_gate_promote: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_search(
    validation: pd.DataFrame,
    confirmation: pd.DataFrame,
    *,
    policy: SearchPolicy,
    budget: int,
    search_seed: int,
    market_seed: int,
    cost_bps: float,
) -> ExperimentRow:
    trials = run_search(
        validation,
        budget=budget,
        policy=policy,
        seed=search_seed,
        cost_bps=cost_bps,
    )
    return evaluate_trials(
        trials,
        confirmation,
        budget=budget,
        policy=policy,
        search_seed=search_seed,
        market_seed=market_seed,
        cost_bps=cost_bps,
    )


def evaluate_trials(
    trials: list[Trial],
    confirmation: pd.DataFrame,
    *,
    budget: int,
    policy: SearchPolicy,
    search_seed: int,
    market_seed: int,
    cost_bps: float,
) -> ExperimentRow:
    """Select from a search prefix, then open the sealed confirmation sample."""

    if budget < 1 or budget > len(trials):
        raise ValueError("budget must select a non-empty prefix of trials")
    trials = trials[:budget]
    selected = max(trials, key=lambda trial: trial.validation_sharpe)
    confirmation_result = run_backtest(confirmation, selected.spec, cost_bps=cost_bps)
    confirmation_returns = confirmation_result.returns.iloc[252:]
    trial_sharpes = np.array([trial.validation_sharpe for trial in trials])
    trial_return_frame = pd.concat(
        [trial.validation_returns.rename(str(index)) for index, trial in enumerate(trials)],
        axis=1,
    )
    n_effective_trials = effective_trial_count(trial_return_frame)
    validation_psr = probabilistic_sharpe_ratio(selected.validation_returns)
    validation_dsr = deflated_sharpe_ratio(
        selected.validation_returns,
        trial_sharpes,
    )
    validation_dsr_effective = deflated_sharpe_ratio(
        selected.validation_returns,
        trial_sharpes,
        n_effective_trials=n_effective_trials,
    )
    validation_hac_probability = hac_positive_mean_probability(selected.validation_returns)
    confirmation_psr = probabilistic_sharpe_ratio(confirmation_returns)
    confirmation_hac_probability = hac_positive_mean_probability(confirmation_returns)
    confirmation_sharpe = float(
        np.sqrt(252.0) * confirmation_returns.mean() / confirmation_returns.std(ddof=1)
    )
    selected_p_value = 1.0 - validation_hac_probability
    trial_p_values = np.array(
        [1.0 - hac_positive_mean_probability(trial.validation_returns) for trial in trials]
    )
    sorted_p_values = np.sort(trial_p_values[np.isfinite(trial_p_values)])
    bh_thresholds = 0.05 * np.arange(1, len(sorted_p_values) + 1) / len(sorted_p_values)
    bh_promote = bool(np.any(sorted_p_values <= bh_thresholds))
    return ExperimentRow(
        market_seed=market_seed,
        search_seed=search_seed,
        policy=policy,
        budget=budget,
        cost_bps=cost_bps,
        effective_trials=n_effective_trials,
        selected_spec=selected.spec.identifier,
        validation_sharpe=selected.validation_sharpe,
        confirmation_sharpe=confirmation_sharpe,
        optimism_gap=selected.validation_sharpe - confirmation_sharpe,
        validation_psr=validation_psr,
        validation_dsr=validation_dsr,
        validation_dsr_effective=validation_dsr_effective,
        validation_hac_probability=validation_hac_probability,
        confirmation_psr=confirmation_psr,
        confirmation_hac_probability=confirmation_hac_probability,
        naive_promote=validation_hac_probability >= 0.95,
        dsr_promote=validation_dsr >= 0.95,
        dsr_effective_promote=validation_dsr_effective >= 0.95,
        bonferroni_promote=selected_p_value <= 0.05 / budget,
        bh_promote=bh_promote,
        holdout_promote=confirmation_hac_probability >= 0.95,
        dual_gate_promote=(validation_dsr >= 0.95 and confirmation_hac_probability >= 0.95),
        dual_effective_gate_promote=(
            validation_dsr_effective >= 0.95 and confirmation_hac_probability >= 0.95
        ),
    )


def run_synthetic_experiment(
    *,
    seeds: int,
    n_assets: int,
    validation_days: int,
    confirmation_days: int,
    budgets: list[int],
    cost_bps: float,
    base_seed: int,
    workers: int = 1,
) -> pd.DataFrame:
    results, _ = run_synthetic_experiment_with_audit(
        seeds=seeds,
        n_assets=n_assets,
        validation_days=validation_days,
        confirmation_days=confirmation_days,
        budgets=budgets,
        cost_bps=cost_bps,
        base_seed=base_seed,
        workers=workers,
    )
    return results


def run_synthetic_experiment_with_audit(
    *,
    seeds: int,
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
            seed_offset,
            n_assets,
            validation_days,
            confirmation_days,
            budgets,
            cost_bps,
            base_seed,
        )
        for seed_offset in range(seeds)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested_outputs = list(executor.map(_run_synthetic_seed, tasks))
    else:
        nested_outputs = [_run_synthetic_seed(task) for task in tasks]
    result_rows = [row for seed_rows, _ in nested_outputs for row in seed_rows]
    audit_rows = [row for _, seed_audit in nested_outputs for row in seed_audit]
    return pd.DataFrame(result_rows), pd.DataFrame(audit_rows)


def _run_synthetic_seed(
    task: tuple[int, int, int, int, list[int], float, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    (
        seed_offset,
        n_assets,
        validation_days,
        confirmation_days,
        budgets,
        cost_bps,
        base_seed,
    ) = task
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    total_days = validation_days + confirmation_days + 504
    market_seed = base_seed + seed_offset
    returns = simulate_null_returns(total_days, n_assets, market_seed)
    validation = returns.iloc[: validation_days + 252]
    confirmation = returns.iloc[validation_days + 252 :]
    sham_returns = simulate_null_returns(total_days, n_assets, market_seed + 1_000_000)
    sham_validation = sham_returns.iloc[: validation_days + 252]
    policies: tuple[SearchPolicy, ...] = (
        "blind",
        "adaptive_sham",
        "adaptive_target",
    )
    for policy_index, policy in enumerate(policies):
        # Target and sham adaptivity start from identical candidates and random
        # choices; only the feedback scores can make them diverge.
        adaptive_seed_offset = 1 if policy.startswith("adaptive") else policy_index
        search_seed = base_seed * 100_000 + seed_offset * 10 + adaptive_seed_offset
        trials = run_search(
            validation,
            budget=max(budgets),
            policy=policy,
            seed=search_seed,
            cost_bps=cost_bps,
            feedback_returns=sham_validation if policy == "adaptive_sham" else None,
        )
        for trial in trials:
            audit_rows.append(
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
            row = evaluate_trials(
                trials,
                confirmation,
                budget=budget,
                policy=policy,
                search_seed=search_seed,
                market_seed=market_seed,
                cost_bps=cost_bps,
            )
            rows.append(row.to_dict())
    return rows, audit_rows


def run_real_data_experiment(
    returns: pd.DataFrame,
    *,
    start_year: int,
    budgets: list[int],
    cost_bps: float,
    base_seed: int,
) -> pd.DataFrame:
    """Run pre-specified rolling 10-year/5-year industry-portfolio folds."""

    if returns.empty:
        raise ValueError("real-data returns are empty")
    latest = returns.index.max()
    latest_full_year = latest.year if latest.month == 12 else latest.year - 1
    confirmation_starts = range(start_year + 11, latest_full_year - 3, 5)
    rows: list[dict[str, object]] = []

    for fold_index, confirmation_start in enumerate(confirmation_starts):
        confirmation_end = confirmation_start + 5
        if confirmation_end - 1 > latest_full_year:
            continue
        validation_start = confirmation_start - 11
        gap_year = confirmation_start - 1
        validation_eval = returns.loc[
            (returns.index.year >= validation_start) & (returns.index.year < gap_year)
        ]
        validation_warmup = returns.loc[returns.index < pd.Timestamp(validation_start, 1, 1)].tail(
            252
        )
        confirmation_eval = returns.loc[
            (returns.index.year >= confirmation_start) & (returns.index.year < confirmation_end)
        ]
        confirmation_warmup = returns.loc[
            returns.index < pd.Timestamp(confirmation_start, 1, 1)
        ].tail(252)
        if len(validation_warmup) < 252 or len(confirmation_warmup) < 252:
            continue
        validation = pd.concat([validation_warmup, validation_eval])
        confirmation = pd.concat([confirmation_warmup, confirmation_eval])

        rng = np.random.default_rng(base_seed + fold_index)
        sham_validation = validation.copy()
        sham_validation.iloc[:, :] = validation.to_numpy()[rng.permutation(len(validation))]
        policies: tuple[SearchPolicy, ...] = (
            "blind",
            "adaptive_sham",
            "adaptive_target",
        )
        for policy_index, policy in enumerate(policies):
            adaptive_seed_offset = 1 if policy.startswith("adaptive") else policy_index
            search_seed = base_seed * 100_000 + fold_index * 10 + adaptive_seed_offset
            trials = run_search(
                validation,
                budget=max(budgets),
                policy=policy,
                seed=search_seed,
                cost_bps=cost_bps,
                feedback_returns=sham_validation if policy == "adaptive_sham" else None,
            )
            for budget in budgets:
                result = evaluate_trials(
                    trials,
                    confirmation,
                    budget=budget,
                    policy=policy,
                    search_seed=search_seed,
                    market_seed=fold_index,
                    cost_bps=cost_bps,
                ).to_dict()
                result.update(
                    {
                        "validation_start": validation_start,
                        "validation_end": gap_year - 1,
                        "gap_year": gap_year,
                        "confirmation_start": confirmation_start,
                        "confirmation_end": confirmation_end - 1,
                    }
                )
                rows.append(result)
    return pd.DataFrame(rows)


def _empirical_window(
    returns: pd.DataFrame,
    block_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a nonoverlapping 1y warmup/5y validation/1y gap/5y confirmation block."""

    validation_warmup = returns.loc[returns.index.year == block_start].tail(252)
    validation_eval = returns.loc[
        (returns.index.year >= block_start + 1) & (returns.index.year <= block_start + 5)
    ]
    confirmation_warmup = returns.loc[returns.index.year == block_start + 6].tail(252)
    confirmation_eval = returns.loc[
        (returns.index.year >= block_start + 7) & (returns.index.year <= block_start + 11)
    ]
    validation = pd.concat([validation_warmup, validation_eval]).dropna(how="any")
    confirmation = pd.concat([confirmation_warmup, confirmation_eval]).dropna(how="any")
    if len(validation_warmup) < 252 or len(confirmation_warmup) < 252:
        raise ValueError(f"insufficient warmup for empirical block {block_start}")
    if len(validation_eval) < 1_200 or len(confirmation_eval) < 1_200:
        raise ValueError(f"insufficient scored observations for empirical block {block_start}")
    return validation, confirmation


def run_empirical_null_benchmark(
    panels: dict[str, pd.DataFrame],
    *,
    block_starts: list[int],
    null_paths_per_cell: int,
    budgets: list[int],
    cost_bps: float,
    base_seed: int,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run exact sign-randomized nulls over fixed, canonical return panels."""

    tasks: list[tuple[object, ...]] = []
    for panel_index, (panel_name, returns) in enumerate(sorted(panels.items())):
        for block_index, block_start in enumerate(block_starts):
            validation, confirmation = _empirical_window(returns, block_start)
            tasks.append(
                (
                    panel_index,
                    panel_name,
                    block_index,
                    block_start,
                    validation,
                    confirmation,
                    null_paths_per_cell,
                    budgets,
                    cost_bps,
                    base_seed,
                )
            )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(_run_empirical_cell, tasks))
    else:
        nested = [_run_empirical_cell(task) for task in tasks]
    result_rows = [row for cell_rows, _ in nested for row in cell_rows]
    audit_rows = [row for _, cell_audit in nested for row in cell_audit]
    return pd.DataFrame(result_rows), pd.DataFrame(audit_rows)


def _run_empirical_cell(
    task: tuple[object, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    (
        panel_index,
        panel_name,
        block_index,
        block_start,
        validation_raw,
        confirmation_raw,
        null_paths_per_cell,
        budgets,
        cost_bps,
        base_seed,
    ) = task
    assert isinstance(validation_raw, pd.DataFrame)
    assert isinstance(confirmation_raw, pd.DataFrame)
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for null_index in range(int(null_paths_per_cell)):
        market_seed = (
            int(base_seed)
            + int(panel_index) * 1_000_000
            + int(block_index) * 10_000
            + null_index
        )
        validation = rademacher_null_returns(validation_raw, market_seed)
        confirmation = rademacher_null_returns(confirmation_raw, market_seed + 5_000_000)
        sham_validation = rademacher_null_returns(validation_raw, market_seed + 10_000_000)
        policies: tuple[SearchPolicy, ...] = (
            "blind",
            "adaptive_sham",
            "adaptive_target",
        )
        for policy_index, policy in enumerate(policies):
            adaptive_seed_offset = 1 if policy.startswith("adaptive") else policy_index
            search_seed = int(base_seed) * 100_000 + market_seed * 10 + adaptive_seed_offset
            trials = run_search(
                validation,
                budget=max(budgets),
                policy=policy,
                seed=search_seed,
                cost_bps=float(cost_bps),
                feedback_returns=sham_validation if policy == "adaptive_sham" else None,
            )
            for trial in trials:
                audit_rows.append(
                    {
                        "dataset": panel_name,
                        "block_start": block_start,
                        "null_index": null_index,
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
                row = evaluate_trials(
                    trials,
                    confirmation,
                    budget=budget,
                    policy=policy,
                    search_seed=search_seed,
                    market_seed=market_seed,
                    cost_bps=float(cost_bps),
                ).to_dict()
                row.update(
                    {
                        "dataset": panel_name,
                        "block_start": block_start,
                        "null_index": null_index,
                    }
                )
                rows.append(row)
    return rows, audit_rows
