import numpy as np

from alpha_firewall.data import rademacher_null_returns, simulate_null_returns


def test_null_simulation_is_reproducible_and_finite() -> None:
    first = simulate_null_returns(300, 12, 42)
    second = simulate_null_returns(300, 12, 42)
    assert first.equals(second)
    assert np.isfinite(first.to_numpy()).all()
    assert first.shape == (300, 12)


def test_rademacher_null_preserves_cross_section_outer_products() -> None:
    original = simulate_null_returns(50, 8, 7)
    signed = rademacher_null_returns(original, 11)
    assert np.allclose(np.abs(signed.to_numpy()), np.abs(original.to_numpy()))
    for row in range(len(original)):
        assert np.allclose(
            np.outer(signed.iloc[row], signed.iloc[row]),
            np.outer(original.iloc[row], original.iloc[row]),
        )
