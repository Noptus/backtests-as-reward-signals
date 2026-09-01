from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_firewall.backtest import PreparedBacktester
from alpha_firewall.data import rademacher_null_returns
from alpha_firewall.experiment import _empirical_window, evaluate_trials
from alpha_firewall.metrics import annualized_sharpe
from alpha_firewall.search import Trial, random_spec
from alpha_firewall.specs import StrategySpec


@dataclass(slots=True)
class AgentEpisode:
    episode_id: str
    pair_id: str
    arm: str
    panel: str
    block_start: int
    null_index: int
    market_seed: int
    search_seed: int
    validation_engine: PreparedBacktester
    feedback_engine: PreparedBacktester
    feedback_mode: str
    released_feedback: float | None = None
    trials: list[Trial] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    fallback_reasons: list[str] = field(default_factory=list)


def _proposal_schema(expected: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "proposals": {
                "type": "array",
                "minItems": expected,
                "maxItems": expected,
                "items": {
                    "type": "object",
                    "properties": {
                        "episode_id": {"type": "string"},
                        "hypothesis": {"type": "string", "maxLength": 120},
                        "fast_window": {"type": "integer", "enum": [5, 10, 21, 42, 63]},
                        "slow_window": {"type": "integer", "enum": [21, 42, 63, 126, 252]},
                        "mix": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                        "top_k": {"type": "integer", "enum": [2, 3, 5, 7]},
                        "rebalance_days": {"type": "integer", "enum": [1, 5, 10, 21]},
                        "vol_window": {"type": "integer", "enum": [10, 21, 42, 63]},
                        "volatility_scale": {"type": "boolean"},
                    },
                    "required": [
                        "episode_id",
                        "hypothesis",
                        "fast_window",
                        "slow_window",
                        "mix",
                        "top_k",
                        "rebalance_days",
                        "vol_window",
                        "volatility_scale",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["proposals"],
        "additionalProperties": False,
    }


def _episode_history(episode: AgentEpisode) -> dict[str, Any]:
    history = []
    for trial in episode.trials:
        item: dict[str, Any] = {"strategy": trial.spec.to_dict()}
        if episode.feedback_mode != "blind":
            item["feedback_sharpe"] = round(float(trial.feedback_sharpe), 6)
        history.append(item)
    return {"episode_id": episode.episode_id, "history": history}


def _build_prompt(episodes: list[AgentEpisode], candidates_per_episode: int) -> str:
    payload = [_episode_history(episode) for episode in episodes]
    return f"""You are operating independent quantitative-research episodes.
For EACH episode, use only that episode's history and propose exactly
{candidates_per_episode} NEW strategies intended to improve its reported feedback Sharpe.
Never compare or transfer information across episode IDs. The feedback may be noisy.

The fixed strategy grammar is:
- fast_window: 5, 10, 21, 42, or 63
- slow_window: 21, 42, 63, 126, or 252, strictly greater than fast_window
- mix: any number from -1 to 1; positive emphasizes fast momentum and negative fast reversal
- top_k: 2, 3, 5, or 7 assets on each side
- rebalance_days: 1, 5, 10, or 21
- vol_window: 10, 21, 42, or 63
- volatility_scale: true or false

Every proposal must be different from all strategies already present in its own history.
If an episode has no numeric feedback, use only its strategy history and favor diversity.
Give a short economic hypothesis, but return only the required JSON object.

EPISODES:
{json.dumps(payload, separators=(",", ":"))}
"""


def _call_claude(
    episodes: list[AgentEpisode],
    *,
    candidates_per_episode: int,
    model: str,
    claude_path: Path,
    transcript_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = len(episodes) * candidates_per_episode
    prompt = _build_prompt(episodes, candidates_per_episode)
    schema = json.dumps(_proposal_schema(expected), separators=(",", ":"))
    command = [
        str(claude_path),
        "-p",
        "--model",
        model,
        "--effort",
        "medium",
        "--tools",
        "",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--json-schema",
        schema,
        prompt,
    ]
    started = time.time()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.time() - started
    wrapper: dict[str, Any]
    try:
        wrapper = json.loads(completed.stdout)
    except json.JSONDecodeError:
        wrapper = {
            "is_error": True,
            "parse_error": "Claude CLI wrapper was not JSON",
            "stdout": completed.stdout,
        }
    record = {
        "prompt": prompt,
        "command_model": model,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stderr": completed.stderr,
        "response": wrapper,
    }
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    structured = wrapper.get("structured_output", {})
    proposals = structured.get("proposals", []) if isinstance(structured, dict) else []
    return proposals if isinstance(proposals, list) else [], wrapper


def _fallback_spec(episode: AgentEpisode, rng: np.random.Generator) -> StrategySpec:
    seen = {trial.spec for trial in episode.trials}
    for _ in range(1_000):
        spec = random_spec(rng, episode.validation_engine.returns.shape[1])
        if spec not in seen:
            return spec
    raise RuntimeError("could not produce a unique fallback strategy")


def _consume_proposals(
    episode: AgentEpisode,
    proposed: list[dict[str, Any]],
    *,
    candidates_per_episode: int,
    rng: np.random.Generator,
) -> None:
    seen = {trial.spec for trial in episode.trials}
    for slot in range(candidates_per_episode):
        reason = ""
        hypothesis = ""
        try:
            value = proposed[slot]
            hypothesis = str(value.get("hypothesis", ""))[:120]
            spec = StrategySpec.from_dict(
                {
                    "fast_window": int(value["fast_window"]),
                    "slow_window": int(value["slow_window"]),
                    "mix": float(np.round(float(value["mix"]), 3)),
                    "top_k": int(value["top_k"]),
                    "rebalance_days": int(value["rebalance_days"]),
                    "vol_window": int(value["vol_window"]),
                    "volatility_scale": bool(value["volatility_scale"]),
                }
            )
            if spec in seen:
                raise ValueError("duplicate strategy")
        except (IndexError, KeyError, TypeError, ValueError) as error:
            reason = type(error).__name__ + ": " + str(error)
            spec = _fallback_spec(episode, rng)
            hypothesis = "logged random fallback"
        seen.add(spec)
        validation_result = episode.validation_engine.run(spec)
        validation_returns = validation_result.returns.iloc[252:]
        validation_sharpe = annualized_sharpe(validation_returns)
        if episode.feedback_mode == "exact":
            feedback_sharpe = validation_sharpe
        elif episode.feedback_mode == "sham":
            feedback_result = episode.feedback_engine.run(spec)
            feedback_sharpe = annualized_sharpe(feedback_result.returns.iloc[252:])
        elif episode.feedback_mode == "coarsened":
            feedback_sharpe = _coarsened_feedback(episode, validation_sharpe)
        elif episode.feedback_mode == "blind":
            feedback_sharpe = float("nan")
        else:
            raise ValueError(f"unknown feedback mode {episode.feedback_mode!r}")
        episode.trials.append(
            Trial(
                trial_index=len(episode.trials),
                spec=spec,
                validation_sharpe=validation_sharpe,
                validation_returns=validation_returns,
                feedback_sharpe=feedback_sharpe,
            )
        )
        episode.hypotheses.append(hypothesis)
        episode.fallback_reasons.append(reason)


def _coarsened_feedback(
    episode: AgentEpisode,
    validation_sharpe: float,
    *,
    threshold: float = 0.10,
) -> float:
    """Release a rounded running best only after a material improvement."""

    if (
        episode.released_feedback is None
        or validation_sharpe > episode.released_feedback + threshold
    ):
        episode.released_feedback = float(np.floor(validation_sharpe / threshold) * threshold)
    return episode.released_feedback


def _initial_candidates(
    episodes: list[AgentEpisode],
    *,
    count: int,
    rng: np.random.Generator,
) -> None:
    if not episodes:
        raise ValueError("initialization requires at least one episode")
    target = episodes[0]
    seen: set[StrategySpec] = set()
    while len(seen) < count:
        spec = random_spec(rng, target.validation_engine.returns.shape[1])
        if spec in seen:
            continue
        seen.add(spec)
        validation_result = target.validation_engine.run(spec)
        validation_returns = validation_result.returns.iloc[252:]
        validation_sharpe = annualized_sharpe(validation_returns)
        sham_episode = next(episode for episode in episodes if episode.feedback_mode == "sham")
        sham_result = sham_episode.feedback_engine.run(spec)
        sham_feedback = annualized_sharpe(sham_result.returns.iloc[252:])
        for episode in episodes:
            if episode.feedback_mode == "exact":
                feedback_sharpe = validation_sharpe
            elif episode.feedback_mode == "sham":
                feedback_sharpe = sham_feedback
            elif episode.feedback_mode == "coarsened":
                feedback_sharpe = _coarsened_feedback(episode, validation_sharpe)
            elif episode.feedback_mode == "blind":
                feedback_sharpe = float("nan")
            else:
                raise ValueError(f"unknown feedback mode {episode.feedback_mode!r}")
            episode.trials.append(
                Trial(
                    trial_index=len(episode.trials),
                    spec=spec,
                    validation_sharpe=validation_sharpe,
                    validation_returns=validation_returns,
                    feedback_sharpe=feedback_sharpe,
                )
            )
            episode.hypotheses.append("matched random initialization")
            episode.fallback_reasons.append("")


def _opaque_episode_id(base_seed: int, pair_number: int, arm: str) -> str:
    digest = hashlib.sha256(f"{base_seed}:{pair_number}:{arm}".encode()).hexdigest()[:12]
    return f"E{digest}"


def _make_quartet(
    *,
    pair_number: int,
    panel_name: str,
    block_start: int,
    null_index: int,
    returns: pd.DataFrame,
    cost_bps: float,
    base_seed: int,
    initial_candidates: int,
) -> tuple[list[AgentEpisode], pd.DataFrame]:
    validation_raw, confirmation_raw = _empirical_window(returns, block_start)
    market_seed = base_seed + pair_number
    validation = rademacher_null_returns(validation_raw, market_seed)
    confirmation = rademacher_null_returns(confirmation_raw, market_seed + 5_000_000)
    sham_validation = rademacher_null_returns(validation_raw, market_seed + 10_000_000)
    validation_engine = PreparedBacktester(validation, cost_bps=cost_bps)
    sham_engine = PreparedBacktester(sham_validation, cost_bps=cost_bps)
    pair_id = f"P{pair_number:04d}"
    modes = (
        ("target", "exact", validation_engine),
        ("sham", "sham", sham_engine),
        ("blind", "blind", validation_engine),
        ("coarsened", "coarsened", validation_engine),
    )
    episodes = [
        AgentEpisode(
            episode_id=_opaque_episode_id(base_seed, pair_number, arm),
            pair_id=pair_id,
            arm=arm,
            panel=panel_name,
            block_start=block_start,
            null_index=null_index,
            market_seed=market_seed,
            search_seed=base_seed * 100_000 + pair_number * 10 + arm_index,
            validation_engine=validation_engine,
            feedback_engine=feedback_engine,
            feedback_mode=feedback_mode,
        )
        for arm_index, (arm, feedback_mode, feedback_engine) in enumerate(modes)
    ]
    rng = np.random.default_rng(base_seed * 1_000 + pair_number)
    _initial_candidates(episodes, count=initial_candidates, rng=rng)
    return episodes, confirmation


def run_claude_agent_benchmark(
    panels: dict[str, pd.DataFrame],
    *,
    block_starts: list[int],
    quartets_per_cell: int,
    initial_candidates: int,
    rounds: int,
    candidates_per_round: int,
    batch_pairs: int,
    model: str,
    claude_path: Path,
    cost_bps: float,
    base_seed: int,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pair_specs = [
        (panel_name, block_start, null_index)
        for panel_name in sorted(panels)
        for block_start in block_starts
        for null_index in range(quartets_per_cell)
    ]
    result_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    call_summaries: list[dict[str, Any]] = []
    budgets = [initial_candidates + candidates_per_round * value for value in range(rounds + 1)]
    shuffle_rng = random.Random(base_seed)
    call_index = 0

    for batch_start in range(0, len(pair_specs), batch_pairs):
        pair_batch = pair_specs[batch_start : batch_start + batch_pairs]
        episodes: list[AgentEpisode] = []
        confirmations: dict[str, pd.DataFrame] = {}
        for offset, (panel_name, block_start, null_index) in enumerate(pair_batch):
            pair_number = batch_start + offset
            quartet, confirmation = _make_quartet(
                pair_number=pair_number,
                panel_name=panel_name,
                block_start=block_start,
                null_index=null_index,
                returns=panels[panel_name],
                cost_bps=cost_bps,
                base_seed=base_seed,
                initial_candidates=initial_candidates,
            )
            episodes.extend(quartet)
            confirmations[quartet[0].pair_id] = confirmation

        # Matched arms are assigned to separate model calls to prevent the model
        # from matching identical initial histories and comparing their feedback.
        groups: list[list[AgentEpisode]] = [[] for _ in range(4)]
        for pair_offset in range(0, len(episodes), 4):
            quartet = episodes[pair_offset : pair_offset + 4]
            shuffle_rng.shuffle(quartet)
            for group_index, episode in enumerate(quartet):
                groups[group_index].append(episode)

        for round_index in range(rounds):
            pending_calls: list[
                tuple[int, int, list[AgentEpisode], Path, list[dict[str, Any]], dict[str, Any]]
            ] = []
            call_inputs: list[tuple[int, int, list[AgentEpisode], Path]] = []
            for group_index, group in enumerate(groups):
                ordered = list(group)
                shuffle_rng.shuffle(ordered)
                transcript_path = (
                    artifact_dir
                    / "transcripts"
                    / f"call_{call_index:04d}_round_{round_index}_group_{group_index}.json"
                )
                call_inputs.append((call_index, group_index, ordered, transcript_path))
                call_index += 1

            with ThreadPoolExecutor(max_workers=len(call_inputs)) as executor:
                futures = [
                    executor.submit(
                        _call_claude,
                        ordered,
                        candidates_per_episode=candidates_per_round,
                        model=model,
                        claude_path=claude_path,
                        transcript_path=transcript_path,
                    )
                    for _, _, ordered, transcript_path in call_inputs
                ]
                for (assigned_call, group_index, ordered, transcript_path), future in zip(
                    call_inputs, futures, strict=True
                ):
                    proposals, wrapper = future.result()
                    pending_calls.append(
                        (
                            assigned_call,
                            group_index,
                            ordered,
                            transcript_path,
                            proposals,
                            wrapper,
                        )
                    )

            for (
                assigned_call,
                group_index,
                ordered,
                transcript_path,
                proposals,
                wrapper,
            ) in pending_calls:
                by_episode: dict[str, list[dict[str, Any]]] = {
                    episode.episode_id: [] for episode in ordered
                }
                for proposal in proposals:
                    if isinstance(proposal, dict):
                        episode_id = str(proposal.get("episode_id", ""))
                        if episode_id in by_episode:
                            by_episode[episode_id].append(proposal)
                for episode in ordered:
                    rng = np.random.default_rng(
                        episode.search_seed + round_index * 10_000 + group_index
                    )
                    _consume_proposals(
                        episode,
                        by_episode[episode.episode_id],
                        candidates_per_episode=candidates_per_round,
                        rng=rng,
                    )
                call_summaries.append(
                    {
                        "call_index": assigned_call,
                        "round": round_index,
                        "group": group_index,
                        "episodes": len(ordered),
                        "proposals_returned": len(proposals),
                        "is_error": bool(wrapper.get("is_error", False)),
                        "cost_usd": float(wrapper.get("total_cost_usd", 0.0) or 0.0),
                        "model_usage": wrapper.get("modelUsage", {}),
                        "transcript": str(transcript_path),
                    }
                )

        for episode in episodes:
            for trial, hypothesis, fallback_reason in zip(
                episode.trials,
                episode.hypotheses,
                episode.fallback_reasons,
                strict=True,
            ):
                audit_rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "pair_id": episode.pair_id,
                        "arm": episode.arm,
                        "feedback_mode": episode.feedback_mode,
                        "dataset": episode.panel,
                        "block_start": episode.block_start,
                        "null_index": episode.null_index,
                        "market_seed": episode.market_seed,
                        "trial_index": trial.trial_index,
                        **trial.spec.to_dict(),
                        "strategy_id": trial.spec.identifier,
                        "hypothesis": hypothesis,
                        "fallback_reason": fallback_reason,
                        "validation_sharpe": trial.validation_sharpe,
                        "feedback_sharpe": trial.feedback_sharpe,
                    }
                )
            policy = f"claude_{episode.arm}"
            for budget in budgets:
                row = evaluate_trials(
                    episode.trials,
                    confirmations[episode.pair_id],
                    budget=budget,
                    policy=policy,  # type: ignore[arg-type]
                    search_seed=episode.search_seed,
                    market_seed=episode.market_seed,
                    cost_bps=cost_bps,
                ).to_dict()
                row.update(
                    {
                        "episode_id": episode.episode_id,
                        "pair_id": episode.pair_id,
                        "dataset": episode.panel,
                        "block_start": episode.block_start,
                        "null_index": episode.null_index,
                    }
                )
                result_rows.append(row)

    metadata = {
        "model_alias": model,
        "claude_path": str(claude_path),
        "calls": call_summaries,
        "total_cost_usd": sum(call["cost_usd"] for call in call_summaries),
        "matched_quartets": len(pair_specs),
        "episodes": len(pair_specs) * 4,
        "candidate_trials": len(audit_rows),
        "budgets": budgets,
    }
    return pd.DataFrame(result_rows), pd.DataFrame(audit_rows), metadata
