from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENDPOINT_CONTRASTS = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])


@dataclass(frozen=True)
class WaldBootstrapResult:
    statistic: float
    p_value: float
    draws: int
    markets: int
    contrast_means: tuple[float, float]


def _wald_statistic(contrasts: np.ndarray) -> float:
    markets = contrasts.shape[0]
    mean = contrasts.mean(axis=0)
    if np.allclose(mean, 0.0, atol=1e-15, rtol=0.0):
        return 0.0
    covariance = np.cov(contrasts, rowvar=False, ddof=1)
    return float(markets * mean @ np.linalg.pinv(covariance, hermitian=True) @ mean)


def wild_cluster_wald_test(
    endpoint_effects: np.ndarray,
    *,
    draws: int = 99_999,
    seed: int = 20260828,
    batch_size: int = 2_000,
) -> WaldBootstrapResult:
    """Test equality of three endpoint means using market-level wild bootstrap.

    Rows are independent markets and columns are the three endpoint-specific
    target-minus-sham effects after within-market replicate averaging.
    """

    effects = np.asarray(endpoint_effects, dtype=float)
    if effects.ndim != 2 or effects.shape[1] != 3:
        raise ValueError("endpoint_effects must have shape (markets, 3)")
    if effects.shape[0] < 3:
        raise ValueError("at least three markets are required")
    if draws < 1 or batch_size < 1:
        raise ValueError("draws and batch_size must be positive")
    if not np.isfinite(effects).all():
        raise ValueError("endpoint_effects must be finite")

    market_contrasts = effects @ ENDPOINT_CONTRASTS.T
    observed = _wald_statistic(market_contrasts)
    null_residuals = market_contrasts - market_contrasts.mean(axis=0)
    rng = np.random.default_rng(seed)
    exceedances = 0
    remaining = draws
    while remaining:
        current = min(batch_size, remaining)
        weights = rng.choice((-1.0, 1.0), size=(current, effects.shape[0], 1))
        bootstrapped = null_residuals[None, :, :] * weights
        means = bootstrapped.mean(axis=1)
        centered = bootstrapped - means[:, None, :]
        covariance = np.einsum("bni,bnj->bij", centered, centered) / (effects.shape[0] - 1)
        inverse = np.linalg.pinv(covariance, hermitian=True)
        statistics = effects.shape[0] * np.einsum("bi,bij,bj->b", means, inverse, means)
        exceedances += int(np.sum(statistics >= observed - 1e-15))
        remaining -= current
    p_value = (exceedances + 1.0) / (draws + 1.0)
    means = market_contrasts.mean(axis=0)
    return WaldBootstrapResult(
        statistic=observed,
        p_value=float(p_value),
        draws=draws,
        markets=effects.shape[0],
        contrast_means=(float(means[0]), float(means[1])),
    )
