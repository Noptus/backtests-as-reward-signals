from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from pathlib import Path

from alpha_firewall.agentic import run_claude_agent_benchmark
from alpha_firewall.data import (
    FRENCH_DAILY_PANELS,
    download_french_49_daily,
    download_french_daily_panel,
    file_sha256,
)
from alpha_firewall.experiment import (
    run_empirical_null_benchmark,
    run_real_data_experiment,
    run_synthetic_experiment_with_audit,
)
from alpha_firewall.power import run_power_experiment
from alpha_firewall.report import build_report
from alpha_firewall.robustness import run_adaptivity_sweep

ROOT = Path(__file__).resolve().parents[2]


def _load_config(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _synthetic(config_path: Path) -> None:
    config = _load_config(config_path)
    experiment = config["experiment"]
    results, audit = run_synthetic_experiment_with_audit(
        seeds=int(experiment["seeds"]),
        n_assets=int(experiment["n_assets"]),
        validation_days=int(experiment["validation_days"]),
        confirmation_days=int(experiment["confirmation_days"]),
        budgets=[int(value) for value in experiment["budgets"]],
        cost_bps=float(experiment["cost_bps"]),
        base_seed=int(experiment["base_seed"]),
        workers=int(experiment.get("workers", 1)),
    )
    output_dir = ROOT / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"synthetic_{config_path.stem}_trials.csv"
    results.to_csv(output_path, index=False)
    audit_path = output_dir / f"synthetic_{config_path.stem}_candidate_audit.csv.gz"
    audit.to_csv(audit_path, index=False, compression="gzip")
    metadata = {
        "config": str(config_path),
        "rows": len(results),
        "candidate_rows": len(audit),
        "columns": list(results.columns),
        "candidate_audit": str(audit_path),
        "result_sha256": file_sha256(output_path),
        "audit_sha256": file_sha256(audit_path),
    }
    (output_dir / f"synthetic_{config_path.stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(results)} rows to {output_path}")


def _real_data(config_path: Path) -> None:
    config = _load_config(config_path)
    real_config = config["real_data"]
    experiment = config["experiment"]
    returns = download_french_49_daily(ROOT / "data" / "raw")
    results = run_real_data_experiment(
        returns,
        start_year=int(real_config["start_year"]),
        budgets=[int(value) for value in real_config["budgets"]],
        cost_bps=float(real_config["cost_bps"]),
        base_seed=int(experiment["base_seed"]),
    )
    output_dir = ROOT / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "french_49_trials.csv"
    results.to_csv(output_path, index=False)
    print(f"wrote {len(results)} rows to {output_path}")


def _empirical_null(config_path: Path) -> None:
    config = _load_config(config_path)
    empirical = config["empirical_null"]
    experiment = config["experiment"]
    cache_dir = ROOT / "data" / "raw"
    panel_names = [str(value) for value in empirical["panels"]]
    panels = {name: download_french_daily_panel(cache_dir, name) for name in panel_names}
    results, audit = run_empirical_null_benchmark(
        panels,
        block_starts=[int(value) for value in empirical["block_starts"]],
        null_paths_per_cell=int(empirical["null_paths_per_cell"]),
        budgets=[int(value) for value in experiment["budgets"]],
        cost_bps=float(empirical["cost_bps"]),
        base_seed=int(experiment["base_seed"]),
        workers=int(experiment.get("workers", 1)),
    )
    output_dir = ROOT / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "empirical_null_trials.csv"
    audit_path = output_dir / "empirical_null_candidate_audit.csv.gz"
    results.to_csv(output_path, index=False)
    audit.to_csv(audit_path, index=False, compression="gzip")
    raw_files = {
        name: {
            "filename": FRENCH_DAILY_PANELS[name],
            "sha256": file_sha256(cache_dir / FRENCH_DAILY_PANELS[name]),
        }
        for name in panel_names
    }
    metadata = {
        "config": str(config_path),
        "rows": len(results),
        "candidate_rows": len(audit),
        "raw_files": raw_files,
        "result_sha256": file_sha256(output_path),
        "audit_sha256": file_sha256(audit_path),
    }
    (output_dir / "empirical_null_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(results)} rows to {output_path}")


def _agentic(config_path: Path, claude_path: Path) -> None:
    config = _load_config(config_path)
    agentic = config["agentic"]
    experiment = config["experiment"]
    cache_dir = ROOT / "data" / "raw"
    panel_names = [str(value) for value in agentic["panels"]]
    panels = {name: download_french_daily_panel(cache_dir, name) for name in panel_names}
    artifact_dir = ROOT / "artifacts" / "agentic"
    results, audit, metadata = run_claude_agent_benchmark(
        panels,
        block_starts=[int(value) for value in agentic["block_starts"]],
        quartets_per_cell=int(agentic["quartets_per_cell"]),
        initial_candidates=int(agentic["initial_candidates"]),
        rounds=int(agentic["rounds"]),
        candidates_per_round=int(agentic["candidates_per_round"]),
        batch_pairs=int(agentic["batch_pairs"]),
        model=str(agentic["model"]),
        claude_path=claude_path,
        cost_bps=float(agentic["cost_bps"]),
        base_seed=int(experiment["base_seed"]),
        artifact_dir=artifact_dir,
    )
    output_dir = ROOT / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "claude_agent_trials.csv"
    audit_path = output_dir / "claude_agent_candidate_audit.csv.gz"
    results.to_csv(output_path, index=False)
    audit.to_csv(audit_path, index=False, compression="gzip")
    metadata.update(
        {
            "config": str(config_path),
            "claude_cli_version": subprocess.run(
                [str(claude_path), "--version"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "result_sha256": file_sha256(output_path),
            "audit_sha256": file_sha256(audit_path),
        }
    )
    (output_dir / "claude_agent_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(results)} rows to {output_path}")


def _power(config_path: Path) -> None:
    config = _load_config(config_path)
    experiment = config["experiment"]
    power = config["power"]
    results, audit = run_power_experiment(
        seeds=int(power["seeds"]),
        signal_bps=[float(value) for value in power["signal_bps"]],
        n_assets=int(experiment["n_assets"]),
        validation_days=int(experiment["validation_days"]),
        confirmation_days=int(experiment["confirmation_days"]),
        budgets=[int(value) for value in experiment["budgets"]],
        cost_bps=float(power["cost_bps"]),
        base_seed=int(power["base_seed"]),
        workers=int(experiment.get("workers", 1)),
    )
    output_dir = ROOT / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "planted_signal_trials.csv"
    audit_path = output_dir / "planted_signal_candidate_audit.csv.gz"
    results.to_csv(output_path, index=False)
    audit.to_csv(audit_path, index=False, compression="gzip")
    metadata = {
        "config": str(config_path),
        "rows": len(results),
        "candidate_rows": len(audit),
        "result_sha256": file_sha256(output_path),
        "audit_sha256": file_sha256(audit_path),
    }
    (output_dir / "planted_signal_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(results)} rows to {output_path}")


def _sweep(config_path: Path) -> None:
    config = _load_config(config_path)["experiment"]
    results = run_adaptivity_sweep(
        seeds=int(config["seeds"]),
        exploit_rates=[float(value) for value in config["exploit_rates"]],
        elite_sizes=[int(value) for value in config["elite_sizes"]],
        n_assets=int(config["n_assets"]),
        validation_days=int(config["validation_days"]),
        confirmation_days=int(config["confirmation_days"]),
        budget=int(config["budget"]),
        cost_bps=float(config["cost_bps"]),
        base_seed=int(config["base_seed"]),
        workers=int(config.get("workers", 1)),
    )
    output_dir = ROOT / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "adaptivity_sweep.csv"
    results.to_csv(output_path, index=False)
    print(f"wrote {len(results)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="alpha-firewall")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.toml")

    real_data = subparsers.add_parser("real-data")
    real_data.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.toml")

    empirical_null = subparsers.add_parser("empirical-null")
    empirical_null.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "deep_paper.toml"
    )

    agentic = subparsers.add_parser("agentic")
    agentic.add_argument("--config", type=Path, default=ROOT / "configs" / "deep_paper.toml")
    agentic.add_argument("--claude", type=Path, default=Path("claude"))

    power = subparsers.add_parser("power")
    power.add_argument("--config", type=Path, default=ROOT / "configs" / "deep_paper.toml")

    sweep = subparsers.add_parser("sweep")
    sweep.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "adaptivity_sweep.toml",
    )

    report = subparsers.add_parser("report")
    report.add_argument(
        "--results",
        type=Path,
        default=ROOT / "artifacts" / "results" / "synthetic_paper_trials.csv",
    )

    args = parser.parse_args()
    if args.command == "synthetic":
        _synthetic(args.config)
    elif args.command == "real-data":
        _real_data(args.config)
    elif args.command == "empirical-null":
        _empirical_null(args.config)
    elif args.command == "agentic":
        _agentic(args.config, args.claude)
    elif args.command == "power":
        _power(args.config)
    elif args.command == "sweep":
        _sweep(args.config)
    elif args.command == "report":
        build_report(args.results, ROOT / "artifacts" / "paper")


if __name__ == "__main__":
    main()
