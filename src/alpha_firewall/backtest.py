from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_firewall.specs import StrategySpec


@dataclass(frozen=True, slots=True)
class BacktestResult:
    returns: pd.Series
    turnover: pd.Series


def _row_zscore(values: pd.DataFrame) -> pd.DataFrame:
    centered = values.sub(values.mean(axis=1), axis=0)
    scale = values.std(axis=1, ddof=0).replace(0.0, np.nan)
    return centered.div(scale, axis=0)


class PreparedBacktester:
    """Cache lagged features while evaluating many strategies on one sample."""

    def __init__(self, returns: pd.DataFrame, *, cost_bps: float = 5.0) -> None:
        if returns.empty or returns.shape[1] < 2:
            raise ValueError("returns must contain at least two assets")
        if cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        self.returns = returns
        self.cost_bps = cost_bps
        windows = (5, 10, 21, 42, 63, 126, 252)
        self.momentum = {
            window: _row_zscore(returns.rolling(window, min_periods=window).sum())
            for window in windows
        }
        self.inverse_volatility = {
            window: 1.0 / returns.rolling(window, min_periods=window).std().clip(lower=1e-6)
            for window in (10, 21, 42, 63)
        }

    def _target_weights(self, spec: StrategySpec) -> pd.DataFrame:
        fast = self.momentum[spec.fast_window]
        slow = self.momentum[spec.slow_window]
        # Positive fast weight expresses momentum; a negative weight expresses
        # short-horizon reversal. The residual weight preserves a slower trend.
        signal = spec.mix * fast + (1.0 - abs(spec.mix)) * slow

        ranks = signal.rank(axis=1, method="first")
        valid_count = signal.notna().sum(axis=1)
        k_by_row = np.minimum(spec.top_k, np.maximum(valid_count // 2, 0))
        long_mask = ranks.gt(valid_count.sub(k_by_row).to_numpy()[:, None])
        short_mask = ranks.le(k_by_row.to_numpy()[:, None])

        if spec.volatility_scale:
            inverse_vol = self.inverse_volatility[spec.vol_window]
            long_raw = inverse_vol.where(long_mask, 0.0)
            short_raw = inverse_vol.where(short_mask, 0.0)
        else:
            long_raw = long_mask.astype(float)
            short_raw = short_mask.astype(float)

        long_total = long_raw.sum(axis=1).replace(0.0, np.nan)
        short_total = short_raw.sum(axis=1).replace(0.0, np.nan)
        weights = 0.5 * long_raw.div(long_total, axis=0) - 0.5 * short_raw.div(short_total, axis=0)
        weights = weights.fillna(0.0)

        rebalance = np.arange(len(weights)) % spec.rebalance_days == 0
        weights.loc[~rebalance, :] = np.nan
        return weights.ffill().fillna(0.0)

    def run(self, spec: StrategySpec) -> BacktestResult:
        if spec.top_k * 2 > self.returns.shape[1]:
            raise ValueError("top_k is too large for the asset universe")
        targets = self._target_weights(spec)
        held = targets.shift(1).fillna(0.0)
        # The target chosen after close t-1 earns return t. Its associated
        # target[t-1]-target[t-2] turnover is charged on that same return row.
        turnover = targets.diff().abs().sum(axis=1).shift(1).fillna(0.0)
        gross_returns = (held * self.returns).sum(axis=1)
        net_returns = gross_returns - turnover * (self.cost_bps / 10_000.0)
        net_returns.name = spec.identifier
        turnover.name = spec.identifier
        return BacktestResult(returns=net_returns, turnover=turnover)


def run_backtest(
    returns: pd.DataFrame,
    spec: StrategySpec,
    *,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """Run a close-to-close backtest with a strict one-session signal lag."""

    return PreparedBacktester(returns, cost_bps=cost_bps).run(spec)
