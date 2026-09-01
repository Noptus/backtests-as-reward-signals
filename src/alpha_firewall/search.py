from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from alpha_firewall.backtest import PreparedBacktester
from alpha_firewall.metrics import annualized_sharpe
from alpha_firewall.specs import (
    FAST_WINDOWS,
    REBALANCE_DAYS,
    SLOW_WINDOWS,
    TOP_K,
    VOL_WINDOWS,
    StrategySpec,
)

SearchPolicy = Literal["blind", "adaptive_sham", "adaptive_target"]


@dataclass(frozen=True, slots=True)
class Trial:
    trial_index: int
    spec: StrategySpec
    validation_sharpe: float
    validation_returns: pd.Series
    feedback_sharpe: float


def random_spec(rng: np.random.Generator, n_assets: int) -> StrategySpec:
    feasible_k = [value for value in TOP_K if value * 2 <= n_assets]
    while True:
        fast = int(rng.choice(FAST_WINDOWS))
        slow = int(rng.choice(SLOW_WINDOWS))
        if fast < slow:
            break
    return StrategySpec(
        fast_window=fast,
        slow_window=slow,
        mix=float(np.round(rng.uniform(-1.0, 1.0), 3)),
        top_k=int(rng.choice(feasible_k)),
        rebalance_days=int(rng.choice(REBALANCE_DAYS)),
        vol_window=int(rng.choice(VOL_WINDOWS)),
        volatility_scale=bool(rng.integers(0, 2)),
    )


def mutate_spec(
    parent: StrategySpec,
    rng: np.random.Generator,
    n_assets: int,
) -> StrategySpec:
    fields = [
        "fast_window",
        "slow_window",
        "mix",
        "top_k",
        "rebalance_days",
        "vol_window",
        "volatility_scale",
    ]
    feasible_k = [value for value in TOP_K if value * 2 <= n_assets]
    for _ in range(50):
        field = str(rng.choice(fields))
        choices = {
            "fast_window": FAST_WINDOWS,
            "slow_window": SLOW_WINDOWS,
            "mix": (),
            "top_k": feasible_k,
            "rebalance_days": REBALANCE_DAYS,
            "vol_window": VOL_WINDOWS,
            "volatility_scale": (False, True),
        }[field]
        try:
            if field == "mix":
                new_value = float(
                    np.round(np.clip(parent.mix + rng.normal(0.0, 0.20), -1.0, 1.0), 3)
                )
            else:
                new_value = rng.choice(choices).item()
            candidate = replace(parent, **{field: new_value})
        except ValueError:
            continue
        if candidate != parent:
            return candidate
    return random_spec(rng, n_assets)


def run_search(
    validation_returns: pd.DataFrame,
    *,
    budget: int,
    policy: SearchPolicy,
    seed: int,
    cost_bps: float,
    burn_in: int = 252,
    feedback_returns: pd.DataFrame | None = None,
    exploit_rate: float = 0.8,
    max_elite: int = 5,
) -> list[Trial]:
    """Evaluate exactly ``budget`` unique candidates using validation only."""

    if budget < 1:
        raise ValueError("budget must be positive")
    if policy not in ("blind", "adaptive_sham", "adaptive_target"):
        raise ValueError(f"unknown search policy {policy}")
    if feedback_returns is not None and feedback_returns.shape != validation_returns.shape:
        raise ValueError("feedback and validation frames must have identical shapes")
    if not 0.0 <= exploit_rate <= 1.0:
        raise ValueError("exploit_rate must lie in [0, 1]")
    if max_elite < 1:
        raise ValueError("max_elite must be positive")

    rng = np.random.default_rng(seed)
    validation_engine = PreparedBacktester(validation_returns, cost_bps=cost_bps)
    feedback_engine = (
        PreparedBacktester(feedback_returns, cost_bps=cost_bps)
        if feedback_returns is not None
        else None
    )
    trials: list[Trial] = []
    seen: set[StrategySpec] = set()
    attempts = 0
    while len(trials) < budget:
        attempts += 1
        if attempts > budget * 500:
            raise RuntimeError("could not draw enough unique strategy specifications")
        if policy.startswith("adaptive") and len(trials) >= 5 and rng.random() < exploit_rate:
            elite_count = max(1, min(max_elite, len(trials) // 5))
            elite = sorted(trials, key=lambda trial: trial.feedback_sharpe, reverse=True)[
                :elite_count
            ]
            parent = elite[int(rng.integers(0, len(elite)))].spec
            spec = mutate_spec(parent, rng, validation_returns.shape[1])
        else:
            spec = random_spec(rng, validation_returns.shape[1])
        if spec in seen:
            continue
        seen.add(spec)
        result = validation_engine.run(spec)
        evaluation_returns = result.returns.iloc[burn_in:]
        validation_sharpe = annualized_sharpe(evaluation_returns)
        if feedback_returns is None:
            feedback_sharpe = validation_sharpe
        else:
            feedback_result = feedback_engine.run(spec)
            feedback_sharpe = annualized_sharpe(feedback_result.returns.iloc[burn_in:])
        trials.append(
            Trial(
                trial_index=len(trials),
                spec=spec,
                validation_sharpe=validation_sharpe,
                validation_returns=evaluation_returns,
                feedback_sharpe=feedback_sharpe,
            )
        )
    return trials
