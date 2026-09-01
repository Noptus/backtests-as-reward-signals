import numpy as np
import pandas as pd

from alpha_firewall.backtest import run_backtest
from alpha_firewall.specs import StrategySpec

SPEC = StrategySpec(
    fast_window=5,
    slow_window=21,
    mix=0.5,
    top_k=2,
    rebalance_days=5,
    vol_window=21,
    volatility_scale=False,
)


def test_future_return_change_does_not_affect_past_pnl() -> None:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=100)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(100, 8)), index=index)
    perturbed = returns.copy()
    perturbed.iloc[70:] += rng.normal(0.0, 0.25, size=(30, 8))

    base = run_backtest(returns, SPEC).returns
    changed = run_backtest(perturbed, SPEC).returns

    pd.testing.assert_series_equal(base.iloc[:70], changed.iloc[:70])
    assert not np.allclose(base.iloc[70:], changed.iloc[70:])


def test_costs_never_improve_returns() -> None:
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(200, 10)))
    free = run_backtest(returns, SPEC, cost_bps=0.0).returns
    costly_result = run_backtest(returns, SPEC, cost_bps=10.0)
    assert (costly_result.returns <= free + 1e-15).all()
    expected_cost = costly_result.turnover * 10.0 / 10_000.0
    pd.testing.assert_series_equal(
        (free - costly_result.returns).rename(SPEC.identifier),
        expected_cost,
        check_exact=False,
        atol=1e-15,
    )
