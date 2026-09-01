from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class Performance:
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    average_turnover: float
    n_observations: int


def annualized_sharpe(returns: pd.Series | np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    volatility = values.std(ddof=1)
    if volatility <= 0:
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * values.mean() / volatility)


def performance(returns: pd.Series, turnover: pd.Series | None = None) -> Performance:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    wealth = (1.0 + clean).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    annual_return = float((wealth.iloc[-1] ** (TRADING_DAYS / len(clean))) - 1.0)
    annual_volatility = float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS))
    average_turnover = float(turnover.reindex(clean.index).mean()) if turnover is not None else 0.0
    return Performance(
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        sharpe=annualized_sharpe(clean),
        max_drawdown=float(drawdown.min()),
        average_turnover=average_turnover,
        n_observations=len(clean),
    )


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Bailey--López de Prado PSR using non-excess Pearson kurtosis."""

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan")
    daily_sr = values.mean() / values.std(ddof=1)
    benchmark_daily = benchmark_sharpe / np.sqrt(TRADING_DAYS)
    sample_skew = float(skew(values, bias=False))
    sample_kurtosis = float(kurtosis(values, fisher=False, bias=False))
    denominator = np.sqrt(
        max(
            1e-12,
            1.0 - sample_skew * daily_sr + ((sample_kurtosis - 1.0) / 4.0) * daily_sr**2,
        )
    )
    statistic = (daily_sr - benchmark_daily) * np.sqrt(values.size - 1) / denominator
    return float(norm.cdf(statistic))


def hac_positive_mean_probability(
    returns: pd.Series | np.ndarray,
    lags: int | None = None,
) -> float:
    """One-sided asymptotic probability using a Newey--West long-run variance."""

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    n_observations = values.size
    if n_observations < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(int(lags), n_observations - 2))
    residuals = values - values.mean()
    long_run_variance = float(residuals @ residuals / n_observations)
    for lag in range(1, lags + 1):
        covariance = float(residuals[lag:] @ residuals[:-lag] / n_observations)
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0:
        return float("nan")
    statistic = values.mean() / np.sqrt(long_run_variance / n_observations)
    return float(norm.cdf(statistic))


def effective_trial_count(trial_returns: pd.DataFrame) -> float:
    """Participation-ratio effective dimension of correlated trial returns."""

    if trial_returns.shape[1] <= 1:
        return float(trial_returns.shape[1])
    correlation = trial_returns.corr().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(correlation, 1.0)
    squared_eigenvalue_sum = float(np.square(correlation).sum())
    if squared_eigenvalue_sum <= 0:
        return 1.0
    n_trials = float(trial_returns.shape[1])
    return float(np.clip(n_trials**2 / squared_eigenvalue_sum, 1.0, n_trials))


def expected_maximum_sharpe(
    sharpes: np.ndarray,
    n_effective_trials: float | None = None,
) -> float:
    """Expected maximum zero-skill Sharpe under Gaussian trial statistics."""

    clean = np.asarray(sharpes, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= 1:
        return 0.0
    trial_std = clean.std(ddof=1)
    if trial_std <= 0:
        return 0.0
    euler_gamma = 0.5772156649015329
    n_trials = float(clean.size if n_effective_trials is None else n_effective_trials)
    if n_trials <= 1.0:
        return 0.0
    first = norm.ppf(1.0 - 1.0 / n_trials)
    second = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(trial_std * ((1.0 - euler_gamma) * first + euler_gamma * second))


def deflated_sharpe_ratio(
    selected_returns: pd.Series | np.ndarray,
    trial_sharpes: np.ndarray,
    n_effective_trials: float | None = None,
) -> float:
    hurdle = expected_maximum_sharpe(trial_sharpes, n_effective_trials)
    return probabilistic_sharpe_ratio(selected_returns, benchmark_sharpe=hurdle)
