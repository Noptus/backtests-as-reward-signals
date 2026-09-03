from __future__ import annotations

import pandas as pd
import pytest

from alpha_firewall.generator_audit import compare_frames, verify_prefix_selection


def test_compare_frames_separates_exact_and_tolerant_fields() -> None:
    expected = pd.DataFrame(
        {
            "market_seed": [1],
            "policy": ["blind"],
            "trial_index": [0],
            "strategy_id": ["x"],
            "validation_sharpe": [0.25],
        }
    )
    actual = expected.copy()
    actual.loc[0, "validation_sharpe"] += 5e-13
    report = compare_frames(
        expected,
        actual,
        keys=("market_seed", "policy", "trial_index"),
        float_fields=("validation_sharpe",),
        atol=1e-12,
        rtol=1e-12,
    )
    assert report["rows"] == 1
    actual.loc[0, "strategy_id"] = "changed"
    with pytest.raises(AssertionError, match="strategy_id"):
        compare_frames(
            expected,
            actual,
            keys=("market_seed", "policy", "trial_index"),
            float_fields=("validation_sharpe",),
            atol=1e-12,
            rtol=1e-12,
        )


def test_boolean_threshold_changes_fail_exactly() -> None:
    expected = pd.DataFrame({"id": [1], "score": [0.95], "promote": [True]})
    actual = pd.DataFrame({"id": [1], "score": [0.95 + 1e-13], "promote": [False]})
    with pytest.raises(AssertionError, match="promote"):
        compare_frames(
            expected,
            actual,
            keys=("id",),
            float_fields=("score",),
            atol=1e-12,
            rtol=1e-12,
        )


def test_prefix_selection_fails_on_a_different_winner() -> None:
    candidates = pd.DataFrame(
        {
            "market_seed": [1, 1],
            "policy": ["blind", "blind"],
            "trial_index": [0, 1],
            "strategy_id": ["a", "b"],
            "validation_sharpe": [0.1, 0.2],
        }
    )
    results = pd.DataFrame(
        {
            "market_seed": [1],
            "policy": ["blind"],
            "budget": [2],
            "selected_spec": ["a"],
        }
    )
    with pytest.raises(AssertionError, match="first validation argmax"):
        verify_prefix_selection(candidates, results)
