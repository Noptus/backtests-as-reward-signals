import pandas as pd

from alpha_firewall.data import simulate_null_returns
from alpha_firewall.power import simulate_planted_momentum_returns


def test_zero_signal_reproduces_null_generator_exactly() -> None:
    expected = simulate_null_returns(300, 10, 77)
    actual = simulate_planted_momentum_returns(300, 10, 77, signal_bps=0.0)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_planted_signal_changes_returns_only_after_lag() -> None:
    baseline = simulate_null_returns(300, 10, 77)
    planted = simulate_planted_momentum_returns(300, 10, 77, signal_bps=1.0)
    pd.testing.assert_frame_equal(planted.iloc[:21], baseline.iloc[:21], check_exact=True)
    assert not planted.iloc[21:].equals(baseline.iloc[21:])
