from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_firewall.headline_verify import (
    HeadlineVerificationError,
    _parse_boolean,
    verify_headlines,
)

ROOT = Path(__file__).resolve().parents[1]


def test_independent_headlines_match_frozen_oracle() -> None:
    report = verify_headlines(
        results_dir=ROOT / "artifacts/results",
        oracle_path=ROOT / "artifacts/paper/deep_summary.json",
    )
    assert report["status"] == "PASS"
    assert report["unit_counts"] == {
        "synthetic_markets": 2_000,
        "empirical_paths": 300,
        "empirical_cells": 15,
        "empirical_paths_per_cell": [20],
        "empirical_eras": 3,
        "agent_paths": 60,
        "agent_cells": 15,
        "agent_paths_per_cell": [4],
    }
    assert report["derived_checks"][
        "synthetic_confirmation_inside_equivalence_band"
    ] is True


def test_verifier_fails_on_perturbed_oracle(tmp_path: Path) -> None:
    oracle = json.loads((ROOT / "artifacts/paper/deep_summary.json").read_text())
    oracle["empirical"]["validation_contrast_cell_level"]["mean"] += 0.01
    oracle_path = tmp_path / "perturbed.json"
    oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
    with pytest.raises(HeadlineVerificationError, match="empirical_validation_mean"):
        verify_headlines(results_dir=ROOT / "artifacts/results", oracle_path=oracle_path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("True", True), ("false", False), ("1", True), ("0", False)],
)
def test_boolean_parser_is_explicit(value: str, expected: bool) -> None:
    assert _parse_boolean(value) is expected


def test_boolean_parser_rejects_ambiguity() -> None:
    with pytest.raises(HeadlineVerificationError, match="ambiguous"):
        _parse_boolean("yes")
