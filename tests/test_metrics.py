import numpy as np
import pandas as pd

from alpha_firewall.metrics import (
    annualized_sharpe,
    effective_trial_count,
    expected_maximum_sharpe,
    hac_positive_mean_probability,
    probabilistic_sharpe_ratio,
)


def test_psr_is_monotone_in_mean_return() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 0.01, size=1_000)
    assert probabilistic_sharpe_ratio(base + 0.001) > probabilistic_sharpe_ratio(base)


def test_sharpe_sign_matches_mean() -> None:
    returns = np.array([0.01, 0.02, -0.005, 0.01])
    assert annualized_sharpe(returns) > 0


def test_hac_probability_is_monotone_in_mean_return() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(0.0, 0.01, size=1_000)
    assert hac_positive_mean_probability(base + 0.001) > hac_positive_mean_probability(base)


def test_expected_maximum_sharpe_uses_zero_skill_mean() -> None:
    # A location shift in the observed trial Sharpes must not shift the
    # zero-skill hurdle; only cross-trial dispersion and trial count matter.
    base = np.array([-0.5, 0.0, 0.5, 1.0])
    assert np.isclose(
        expected_maximum_sharpe(base),
        expected_maximum_sharpe(base + 10.0),
    )


def test_effective_trials_shrink_for_duplicate_returns() -> None:
    rng = np.random.default_rng(17)
    independent = rng.normal(size=(500, 3))
    duplicated = np.column_stack([independent[:, 0], independent[:, 0], independent[:, 0]])
    assert effective_trial_count(pd.DataFrame(duplicated)) == 1.0
    assert effective_trial_count(pd.DataFrame(independent)) > 2.5
