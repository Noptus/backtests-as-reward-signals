from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from alpha_firewall import repro_check
from alpha_firewall.repro_check import json_differences, verify_manifest


def test_verify_manifest_fails_closed_on_hash_mismatch(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("original", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "payload.txt": {
                    "bytes": payload.stat().st_size,
                    "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "pass"
    payload.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_manifest(tmp_path, manifest)


def test_json_differences_uses_explicit_numeric_tolerance() -> None:
    expected = {"estimate": 0.25, "nested": [1, "fixed"]}
    assert json_differences(expected, {"estimate": 0.25 + 1e-14, "nested": [1, "fixed"]}) == []
    assert json_differences(expected, {"estimate": 0.26, "nested": [1, "fixed"]})


def test_missing_executable_version_is_unavailable() -> None:
    assert repro_check.command_version(["/nonexistent/portfolio-test-tool"]) is None


def test_missing_tools_still_produce_a_failure_report(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "saved-report.json"
    monkeypatch.setattr("sys.argv", ["repro_check", "--root", str(tmp_path),
                                   "--report-output", str(report_path)])
    monkeypatch.setattr(repro_check, "collect_environment",
                        lambda: {"uv": "available", "tectonic": None, "pdftotext": None})
    with pytest.raises(SystemExit) as error:
        repro_check.main()
    assert error.value.code == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "FAIL"
    assert "Missing required tools: tectonic, pdftotext" in report["error"]
    assert report["hosted_model_calls"] == 0
