from __future__ import annotations

import numpy as np

from alpha_firewall.p3_inference import wild_cluster_wald_test


def test_wild_cluster_test_is_invariant_to_common_market_shift() -> None:
    rng = np.random.default_rng(11)
    effects = rng.normal(size=(40, 3))
    common_shift = rng.normal(size=(40, 1))
    baseline = wild_cluster_wald_test(effects, draws=999, seed=7)
    shifted = wild_cluster_wald_test(effects + common_shift, draws=999, seed=7)
    assert shifted.statistic == baseline.statistic
    assert shifted.p_value == baseline.p_value


def test_wild_cluster_test_detects_large_endpoint_mean_difference() -> None:
    rng = np.random.default_rng(12)
    effects = rng.normal(scale=0.15, size=(80, 3))
    effects[:, 2] += 0.25
    result = wild_cluster_wald_test(effects, draws=4_999, seed=9)
    assert result.statistic > 20.0
    assert result.p_value < 0.01


def test_wild_cluster_test_requires_three_endpoint_columns() -> None:
    with np.testing.assert_raises_regex(ValueError, "shape"):
        wild_cluster_wald_test(np.ones((10, 2)), draws=99)
