import numpy as np
import pandas as pd

from alpha_firewall.search import run_search
from alpha_firewall.specs import StrategySpec


def test_target_and_sham_arms_share_initial_candidates() -> None:
    rng = np.random.default_rng(23)
    index = pd.bdate_range("2010-01-01", periods=600)
    target = pd.DataFrame(rng.normal(0, 0.01, size=(600, 12)), index=index)
    sham = pd.DataFrame(rng.normal(0, 0.01, size=(600, 12)), index=index)

    target_trials = run_search(
        target,
        budget=5,
        policy="adaptive_target",
        seed=99,
        cost_bps=0.0,
    )
    sham_trials = run_search(
        target,
        feedback_returns=sham,
        budget=5,
        policy="adaptive_sham",
        seed=99,
        cost_bps=0.0,
    )

    assert [trial.spec for trial in target_trials] == [trial.spec for trial in sham_trials]
    assert [trial.validation_sharpe for trial in target_trials] == [
        trial.validation_sharpe for trial in sham_trials
    ]


def test_equal_weight_specs_canonicalize_irrelevant_vol_window() -> None:
    first = StrategySpec(5, 21, 0.5, 2, 5, 10, False)
    second = StrategySpec(5, 21, 0.5, 2, 5, 63, False)
    assert first == second
    assert first.identifier == second.identifier
