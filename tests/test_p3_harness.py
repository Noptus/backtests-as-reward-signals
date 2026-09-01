from __future__ import annotations

import copy
from pathlib import Path

import pytest

from alpha_firewall.p3_harness import (
    BurnedConfirmation,
    ConfirmationReceipt,
    ConfirmationService,
    CostCeilingError,
    CostLedger,
    EpisodeKey,
    HashChainLedger,
    HistoryItem,
    InformationFlowError,
    MockProvider,
    ProvenanceError,
    ProviderResult,
    SecretRedactor,
    build_prompt,
    execute_candidate,
    strategy_content,
)
from alpha_firewall.specs import StrategySpec

MODEL = "frozen-mock-model"
SPEC = StrategySpec(5, 21, 0.25, 2, 5, 10, True)
KEY = EpisodeKey("opaque-a7f3", 810000, 0, 1, 0)


def result(
    status: str,
    content: str | None = None,
    *,
    model: str = MODEL,
    cost: float = 0.05,
) -> ProviderResult:
    return ProviderResult(status, content, model, cost, latency_ms=7)  # type: ignore[arg-type]


def invoke(
    provider: MockProvider,
    *,
    ceiling: float = 2.0,
    secret: str = "test-secret",
    ledger: HashChainLedger | None = None,
):
    selected_ledger = ledger or HashChainLedger(redactor=SecretRedactor((secret,)))
    outcome = execute_candidate(
        provider=provider,
        requested_model=MODEL,
        frozen_canonical_model=MODEL,
        frozen_instruction="Propose one new strategy from this frozen grammar.",
        history=(),
        key=KEY,
        base_seed=20260828,
        cost_ledger=CostLedger(ceiling),
        audit_ledger=selected_ledger,
        headers={"Authorization": f"Bearer {secret}"},
        executable_hash="f" * 64,
        price_schedule="mock-2026-08-29",
    )
    return outcome, selected_ledger


def test_retry_once_only_for_empty_transport_failure() -> None:
    provider = MockProvider([result("transport_empty"), result("ok", strategy_content(SPEC))])
    outcome, ledger = invoke(provider)
    assert outcome.source == "model"
    assert outcome.attempts == 2
    assert len(provider.calls) == 2
    assert ledger.entries[1]["record"]["retry_reason"] == "transport_empty"


def test_empty_transport_without_model_metadata_still_gets_one_retry() -> None:
    provider = MockProvider(
        [
            result("transport_empty", model=""),
            result("ok", strategy_content(SPEC)),
        ]
    )
    outcome, ledger = invoke(provider)
    assert outcome.source == "model"
    assert len(provider.calls) == 2
    assert ledger.entries[0]["record"]["canonical_model"] == ""


def test_invalid_content_does_not_retry_and_uses_deterministic_fallback() -> None:
    first, _ = invoke(MockProvider([result("ok", "not-json")]))
    second, _ = invoke(MockProvider([result("ok", "not-json")]))
    assert first.source == "fallback"
    assert first.attempts == 1
    assert first.strategy == second.strategy


def test_two_empty_transport_failures_do_not_get_a_third_call() -> None:
    provider = MockProvider(
        [result("transport_empty"), result("transport_empty"), result("ok", "{}")]
    )
    outcome, _ = invoke(provider)
    assert outcome.source == "fallback"
    assert outcome.attempts == 2
    assert len(provider.calls) == 2


def test_cost_ceiling_fails_before_provider_call() -> None:
    provider = MockProvider([result("ok", strategy_content(SPEC))], maximum_call_cost_usd=0.25)
    with pytest.raises(CostCeilingError):
        invoke(provider, ceiling=0.24)
    assert provider.calls == []


def test_cumulative_reservations_never_cross_cost_ceiling() -> None:
    ledger = CostLedger(0.50)
    first = ledger.reserve(0.25)
    ledger.settle(first, 0.20)
    second = ledger.reserve(0.25)
    ledger.settle(second, 0.25)
    assert ledger.committed_usd == pytest.approx(0.45)
    with pytest.raises(CostCeilingError):
        ledger.reserve(0.06)
    assert ledger.committed_usd + ledger.reserved_usd <= ledger.ceiling_usd


def test_secret_redaction_precedes_persistence(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    secret = "never-write-this-token"
    ledger = HashChainLedger(redactor=SecretRedactor((secret,)), path=path)
    provider = MockProvider(
        [
            ProviderResult(
                "ok",
                strategy_content(SPEC),
                MODEL,
                0.05,
                response_metadata=(("echo", secret),),
            )
        ]
    )
    invoke(provider, secret=secret, ledger=ledger)
    stored = path.read_text(encoding="utf-8")
    assert secret not in stored
    assert "<REDACTED>" in stored


def test_hash_chain_detects_tampering() -> None:
    _outcome, ledger = invoke(MockProvider([result("ok", strategy_content(SPEC))]))
    ledger.append({"event": "second"})
    ledger.verify()
    altered = copy.deepcopy(ledger.entries)
    altered[0]["record"]["provider"] = "tampered"
    with pytest.raises(ValueError, match="entry 0"):
        ledger.verify(altered)


def test_canonical_model_mismatch_is_fatal() -> None:
    with pytest.raises(ProvenanceError):
        invoke(MockProvider([result("ok", strategy_content(SPEC), model="moving-alias")]))


def test_duplicate_model_proposal_does_not_retry() -> None:
    ledger = HashChainLedger(redactor=SecretRedactor())
    provider = MockProvider([result("ok", strategy_content(SPEC))])
    outcome = execute_candidate(
        provider=provider,
        requested_model=MODEL,
        frozen_canonical_model=MODEL,
        frozen_instruction="frozen",
        history=(HistoryItem(SPEC, 0.1),),
        key=KEY,
        base_seed=20260828,
        cost_ledger=CostLedger(1.0),
        audit_ledger=ledger,
        headers={},
        executable_hash="f" * 64,
        price_schedule="mock",
    )
    assert outcome.source == "fallback"
    assert outcome.attempts == 1
    assert len(provider.calls) == 1
    assert ledger.entries[0]["record"]["duplicate_status"] == "duplicate"


def test_provider_invalid_status_does_not_retry() -> None:
    provider = MockProvider([result("invalid"), result("ok", strategy_content(SPEC))])
    outcome, _ledger = invoke(provider)
    assert outcome.source == "fallback"
    assert outcome.attempts == 1
    assert len(provider.calls) == 1


def test_confirmation_is_write_once_and_cannot_enter_prompt() -> None:
    service = ConfirmationService({(810000, 0): 0.42})
    first = service.query((810000, 0))
    second = service.query((810000, 0))
    assert isinstance(first, ConfirmationReceipt)
    assert isinstance(second, BurnedConfirmation)
    assert second.intent_to_treat_failure is True
    with pytest.raises(InformationFlowError):
        build_prompt("frozen", (first,))  # type: ignore[arg-type]


def test_prompt_contains_only_structured_same_episode_history() -> None:
    prompt = build_prompt("frozen", (HistoryItem(SPEC, 0.1234567),))
    assert "opaque-a7f3" not in prompt
    assert "confirmation" not in prompt.lower()
    assert '"feedback_sharpe":0.123457' in prompt


def test_mock_run_has_reproducible_hash_chain() -> None:
    first, first_ledger = invoke(MockProvider([result("invalid")]))
    second, second_ledger = invoke(MockProvider([result("invalid")]))
    assert first.strategy == second.strategy
    assert [entry["entry_hash"] for entry in first_ledger.entries] == [
        entry["entry_hash"] for entry in second_ledger.entries
    ]
