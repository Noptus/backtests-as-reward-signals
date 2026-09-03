from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TEXT_OUTPUTS = (
    "deep_macros.tex",
    "synthetic_gate_table.tex",
    "empirical_table.tex",
    "agentic_table.tex",
    "power_table.tex",
)
FIGURE_OUTPUTS = (
    "synthetic_trajectories.pdf",
    "synthetic_gates.pdf",
    "empirical_forest.pdf",
    "agentic_trajectories.pdf",
    "power_curves.pdf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("manifest must be a non-empty object")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for relative, expected in manifest.items():
        path = root / relative
        row: dict[str, Any] = {"path": relative, "exists": path.is_file()}
        if not path.is_file():
            row["status"] = "missing"
            failures.append(f"missing: {relative}")
        else:
            row["bytes"] = path.stat().st_size
            row["sha256"] = sha256(path)
            row["status"] = (
                "pass"
                if row["bytes"] == expected.get("bytes")
                and row["sha256"] == expected.get("sha256")
                else "mismatch"
            )
            if row["status"] != "pass":
                failures.append(f"hash/size mismatch: {relative}")
        rows.append(row)
    if failures:
        raise ValueError("; ".join(failures))
    return rows


def json_differences(
    expected: Any,
    actual: Any,
    *,
    path: str = "$",
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        if expected.keys() != actual.keys():
            differences.append(f"{path}: key sets differ")
        for key in expected.keys() & actual.keys():
            differences.extend(
                json_differences(
                    expected[key],
                    actual[key],
                    path=f"{path}.{key}",
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
            )
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: list lengths differ"]
        differences = []
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual, strict=True)):
            differences.extend(
                json_differences(
                    expected_value,
                    actual_value,
                    path=f"{path}[{index}]",
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
            )
        return differences
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if abs(float(expected) - float(actual)) <= absolute_tolerance + relative_tolerance * abs(
            float(expected)
        ):
            return []
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return [] if expected == actual else [f"{path}: expected {expected!r}, got {actual!r}"]


def run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "wall_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def command_version(command: list[str]) -> str | None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else None


def collect_environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "pytest", "ruff"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "uv": command_version(["uv", "--version"]),
        "tectonic": command_version(["tectonic", "--version"]),
        "pdftotext": command_version(["pdftotext", "-v"]),
    }


def copy_manifest_inputs(root: Path, manifest_path: Path, staging: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative in manifest:
        source = root / relative
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed P1 artifact reproduction check")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = (
        args.manifest or root / "artifacts/results/deep_artifact_manifest.json"
    ).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    staging = (args.staging or root / "tmp/reproduction" / stamp).resolve()
    if staging.exists():
        raise SystemExit(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    report: dict[str, Any] = {
        "status": "FAIL",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "root": str(root),
        "staging": str(staging),
        "manifest": str(manifest_path),
        "hosted_model_calls": 0,
        "network_downloads": 0,
        "environment": collect_environment(),
        "steps": {},
    }
    started = time.monotonic()
    try:
        rows = verify_manifest(root, manifest_path)
        report["steps"]["manifest"] = {"status": "PASS", "files": rows}
        copy_manifest_inputs(root, manifest_path, staging)

        ruff = run_command(
            [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"], cwd=root
        )
        report["steps"]["ruff"] = ruff
        if ruff["returncode"] != 0:
            raise RuntimeError("Ruff failed")

        pytest = run_command([sys.executable, "-m", "pytest"], cwd=root)
        report["steps"]["pytest"] = pytest
        if pytest["returncode"] != 0:
            raise RuntimeError("pytest failed")

        staged_results = staging / "artifacts/results"
        staged_generator_report = (
            staging / "artifacts/reproduction/synthetic_generator_spotcheck.json"
        )
        generator_audit = run_command(
            [
                sys.executable,
                "-m",
                "alpha_firewall.generator_audit",
                "--config",
                str(staging / "configs/deep_paper.toml"),
                "--candidate-audit",
                str(staged_results / "synthetic_deep_paper_candidate_audit.csv.gz"),
                "--results",
                str(staged_results / "synthetic_deep_paper_trials.csv"),
                "--output",
                str(staged_generator_report),
            ],
            cwd=root,
        )
        report["steps"]["synthetic_generator_spotcheck"] = generator_audit
        if generator_audit["returncode"] != 0:
            raise RuntimeError("synthetic generator-to-archive audit failed")
        expected_generator_report = json.loads(
            (
                root
                / "artifacts/reproduction/synthetic_generator_spotcheck.json"
            ).read_text(encoding="utf-8")
        )
        actual_generator_report = json.loads(
            staged_generator_report.read_text(encoding="utf-8")
        )
        generator_differences = json_differences(
            expected_generator_report, actual_generator_report
        )
        report["steps"]["synthetic_generator_spotcheck_comparison"] = {
            "status": "PASS" if not generator_differences else "FAIL",
            "differences": generator_differences,
            "seeds": actual_generator_report["seeds"],
            "candidate_rows": actual_generator_report["candidate_comparison"]["rows"],
            "result_rows": actual_generator_report["result_comparison"]["rows"],
            "claim_boundary": actual_generator_report["claim_boundary"],
        }
        if generator_differences:
            raise RuntimeError("synthetic generator audit report differs")

        staged_assets = staging / "artifacts/paper"
        analysis = run_command(
            [
                sys.executable,
                str(root / "scripts/analyze_deep.py"),
                "--results-dir",
                str(staged_results),
                "--output-dir",
                str(staged_assets),
                "--skip-manifest",
            ],
            cwd=root,
        )
        report["steps"]["analysis"] = analysis
        if analysis["returncode"] != 0:
            raise RuntimeError("analysis regeneration failed")

        staged_independent_report = (
            staging / "artifacts/reproduction/independent_headlines.json"
        )
        independent = run_command(
            [
                sys.executable,
                str(root / "src/alpha_firewall/headline_verify.py"),
                "--results-dir",
                str(staged_results),
                "--oracle",
                str(staged_assets / "deep_summary.json"),
                "--output",
                str(staged_independent_report),
            ],
            cwd=root,
        )
        report["steps"]["independent_headline_verification"] = independent
        if independent["returncode"] != 0:
            raise RuntimeError("independent headline verification failed")
        expected_independent = json.loads(
            (root / "artifacts/reproduction/independent_headlines.json").read_text(
                encoding="utf-8"
            )
        )
        actual_independent = json.loads(staged_independent_report.read_text(encoding="utf-8"))
        independent_differences = json_differences(expected_independent, actual_independent)
        report["steps"]["independent_headline_comparison"] = {
            "status": "PASS" if not independent_differences else "FAIL",
            "differences": independent_differences,
            "claim_components": len(actual_independent["comparisons"]),
            "scope": actual_independent["scope"],
            "unit_counts": actual_independent["unit_counts"],
        }
        if independent_differences:
            raise RuntimeError("independent headline report differs")

        expected_summary = json.loads(
            (root / "artifacts/paper/deep_summary.json").read_text(encoding="utf-8")
        )
        actual_summary = json.loads(
            (staged_assets / "deep_summary.json").read_text(encoding="utf-8")
        )
        differences = json_differences(expected_summary, actual_summary)
        text_checks = {}
        for name in TEXT_OUTPUTS:
            expected_text = (root / "artifacts/paper" / name).read_text(encoding="utf-8")
            actual_text = (staged_assets / name).read_text(encoding="utf-8")
            text_checks[name] = {"exact_match": expected_text == actual_text}
            if expected_text != actual_text:
                differences.append(f"{name}: text differs")
        report["steps"]["numeric_comparison"] = {
            "status": "PASS" if not differences else "FAIL",
            "differences": differences,
            "text_outputs": text_checks,
            "headline_values": {
                "synthetic_validation_difference": actual_summary["synthetic"]["contrasts"][
                    "validation_sharpe"
                ],
                "synthetic_confirmation_difference": actual_summary["synthetic"]["contrasts"][
                    "confirmation_sharpe"
                ],
                "empirical_validation_difference": actual_summary["empirical"][
                    "validation_contrast_cell_level"
                ],
                "agent_target_minus_sham": actual_summary["agentic"]["contrasts"][
                    "target_minus_sham"
                ],
            },
        }
        if differences:
            raise RuntimeError("regenerated numeric/text outputs differ")

        for name in FIGURE_OUTPUTS:
            if not (staged_assets / name).is_file():
                raise RuntimeError(f"missing regenerated figure: {name}")

        staged_paper = staging / "paper"
        staged_paper.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "paper/deep_main.tex", staged_paper / "deep_main.tex")
        shutil.copy2(root / "paper/references.bib", staged_paper / "references.bib")
        staged_pdf_dir = staging / "output/pdf"
        staged_pdf_dir.mkdir(parents=True, exist_ok=True)
        compile_result = run_command(
            ["tectonic", "paper/deep_main.tex", "--outdir", str(staged_pdf_dir)], cwd=staging
        )
        report["steps"]["paper_compile"] = compile_result
        if compile_result["returncode"] != 0:
            raise RuntimeError("paper compilation failed")
        staged_pdf = staged_pdf_dir / "deep_main.pdf"
        pdfinfo = run_command(["pdfinfo", str(staged_pdf)], cwd=staging)
        report["steps"]["pdfinfo"] = pdfinfo
        if pdfinfo["returncode"] != 0 or "Pages:           8" not in pdfinfo["stdout_tail"]:
            raise RuntimeError("compiled PDF is missing or does not have eight pages")
        pdf_text_started = time.monotonic()
        pdf_text_result = subprocess.run(
            ["pdftotext", str(staged_pdf), "-"],
            cwd=staging,
            capture_output=True,
            text=True,
            check=False,
        )
        report["steps"]["pdf_text"] = {
            "command": ["pdftotext", str(staged_pdf), "-"],
            "returncode": pdf_text_result.returncode,
            "wall_seconds": round(time.monotonic() - pdf_text_started, 3),
            "stdout_tail": pdf_text_result.stdout[-1000:],
            "stderr_tail": pdf_text_result.stderr[-1000:],
        }
        if pdf_text_result.returncode != 0:
            raise RuntimeError("PDF text extraction failed")
        lowered = pdf_text_result.stdout.lower()
        if any(marker in lowered for marker in ("undefined control sequence", " tbd", "todo")):
            raise RuntimeError("compiled PDF contains a placeholder or undefined marker")

        report["staged_output_hashes"] = {
            str(path.relative_to(staging)): sha256(path)
            for path in sorted(staged_assets.iterdir())
            if path.is_file()
        }
        report["staged_output_hashes"][
            "artifacts/reproduction/independent_headlines.json"
        ] = sha256(staged_independent_report)
        report["staged_output_hashes"][
            "artifacts/reproduction/synthetic_generator_spotcheck.json"
        ] = sha256(staged_generator_report)
        report["staged_pdf"] = {
            "path": str(staged_pdf),
            "sha256": sha256(staged_pdf),
            "bytes": staged_pdf.stat().st_size,
            "pages": 8,
            "byte_hash_is_provenance_only": True,
        }
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        report["wall_seconds"] = round(time.monotonic() - started, 3)
        report_path = staging / "repro_report.json"
        report_text = json.dumps(report, indent=2) + "\n"
        report_path.write_text(report_text, encoding="utf-8")
        preserved_report = None
        if args.report_output:
            preserved_report = args.report_output.resolve()
            preserved_report.parent.mkdir(parents=True, exist_ok=True)
            preserved_report.write_text(report_text, encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(report_path),
                    "preserved_report": str(preserved_report) if preserved_report else None,
                },
                indent=2,
            )
        )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
