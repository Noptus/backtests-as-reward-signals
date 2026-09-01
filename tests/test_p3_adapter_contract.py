from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from alpha_firewall.p3_adapter_contract import (
    RETRIEVED_ON,
    SPECS,
    build_request,
    maximum_cost_reservation,
    parse_response,
    request_sha256,
    schema_fingerprint,
    to_provider_result,
)
from alpha_firewall.p3_harness import ProvenanceError, proposal_schema

CONTENT = json.dumps(
    {
        "fast_window": 5,
        "slow_window": 21,
        "mix": 0.25,
        "top_k": 2,
        "rebalance_days": 5,
        "vol_window": 10,
        "volatility_scale": True,
    }
)


def test_request_dialects_are_exact_secret_free_and_canonical() -> None:
    schema = proposal_schema()
    anthropic = build_request("anthropic", prompt="frozen", schema=schema)
    openai = build_request("openai", prompt="frozen", schema=schema)
    gemini = build_request("gemini", prompt="frozen", schema=schema)
    assert anthropic["model"] == "claude-sonnet-5"
    assert anthropic["output_config"]["format"]["schema"] == schema
    assert "thinking" not in anthropic and "temperature" not in anthropic
    assert openai["model"] == "gpt-5.4-2026-03-05"
    assert openai["reasoning"] == {"effort": "none"}
    assert openai["text"]["format"]["schema"] == schema
    assert gemini["model"] == "gemini-3.6-flash"
    assert gemini["response_format"]["schema"] == schema
    assert gemini["generation_config"] == {
        "max_output_tokens": 1200,
        "thinking_level": "minimal",
        "thinking_summaries": "none",
        "tool_choice": "none",
    }
    assert "temperature" not in gemini and "top_p" not in gemini
    assert request_sha256(anthropic) == request_sha256(anthropic)
    assert "secret" not in json.dumps([anthropic, openai, gemini]).lower()


def test_schema_fingerprint_is_provider_independent() -> None:
    schema = proposal_schema()
    embedded = [
        build_request(provider, prompt="frozen", schema=schema) for provider in SPECS
    ]
    assert len({schema_fingerprint(schema) for _body in embedded}) == 1


def test_anthropic_ignores_thinking_and_parses_exactly_one_text_block() -> None:
    raw = {
        "id": "msg_1",
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "private chain"},
            {"type": "text", "text": CONTENT},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    parsed = parse_response("anthropic", raw)
    assert parsed.status == "ok"
    assert parsed.content == CONTENT
    assert "private chain" not in (parsed.content or "")
    assert parsed.estimated_cost_usd == pytest.approx(0.0004)


def test_anthropic_http_200_refusal_is_not_success() -> None:
    raw = {
        "id": "msg_refusal",
        "model": "claude-sonnet-5",
        "stop_reason": "refusal",
        "content": [{"type": "text", "text": "I cannot comply"}],
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    parsed = parse_response("anthropic", raw)
    assert parsed.status == "refused"
    assert parsed.content is None


def test_openai_ignores_reasoning_item_and_reasoning_is_not_double_billed() -> None:
    raw = {
        "id": "resp_1",
        "model": "gpt-5.4-2026-03-05",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": CONTENT}],
            },
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 8},
        },
    }
    parsed = parse_response("openai", raw)
    assert parsed.status == "ok"
    assert parsed.content == CONTENT
    assert parsed.thought_tokens == 8
    assert parsed.estimated_cost_usd == pytest.approx(0.00055)


def test_openai_refusal_and_incomplete_fail_closed() -> None:
    refusal = {
        "model": "gpt-5.4-2026-03-05",
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}
        ],
    }
    incomplete = {"model": "gpt-5.4-2026-03-05", "status": "incomplete"}
    assert parse_response("openai", refusal).status == "refused"
    assert parse_response("openai", incomplete).status == "incomplete"


def test_gemini_extracts_model_output_and_bills_thought_tokens() -> None:
    raw = {
        "id": "interaction_1",
        "model": "gemini-3.6-flash",
        "status": "completed",
        "steps": [
            {"type": "thought", "content": [{"type": "text", "text": "private"}]},
            {"type": "model_output", "content": [{"type": "text", "text": CONTENT}]},
        ],
        "usage": {
            "total_input_tokens": 100,
            "total_output_tokens": 20,
            "total_thought_tokens": 30,
        },
    }
    parsed = parse_response("gemini", raw)
    assert parsed.status == "ok"
    assert parsed.content == CONTENT
    assert parsed.thought_tokens == 30
    assert parsed.estimated_cost_usd == pytest.approx(0.0002625)


@pytest.mark.parametrize(
    ("provider", "wrong_model"),
    [
        ("anthropic", "claude-sonnet-5.1"),
        ("openai", "gpt-5.4"),
        ("gemini", "gemini-3.6-pro"),
    ],
)
def test_exact_model_mismatch_is_fatal(provider: str, wrong_model: str) -> None:
    with pytest.raises(ProvenanceError):
        parse_response(provider, {"model": wrong_model, "content": ["x"]})


def test_empty_transport_without_model_is_retry_class_not_provenance_error() -> None:
    parsed = parse_response("anthropic", {"_transport_empty": True})
    assert parsed.status == "transport_empty"
    assert parsed.provenance.canonical_model == ""


def test_missing_model_with_content_is_invalid() -> None:
    parsed = parse_response("gemini", {"steps": [{"type": "model_output"}]})
    assert parsed.status == "invalid"


def test_more_than_one_typed_text_block_is_invalid() -> None:
    raw = {
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "content": [
            {"type": "text", "text": CONTENT},
            {"type": "text", "text": CONTENT},
        ],
    }
    assert parse_response("anthropic", raw).status == "invalid"


def test_cost_reservations_use_dated_schedules() -> None:
    assert maximum_cost_reservation(
        "anthropic", maximum_input_tokens=1_000, maximum_output_tokens=1_200
    ) == pytest.approx(0.014)
    assert maximum_cost_reservation(
        "openai", maximum_input_tokens=1_000, maximum_output_tokens=1_200
    ) == pytest.approx(0.0205)
    assert maximum_cost_reservation(
        "gemini", maximum_input_tokens=1_000, maximum_output_tokens=1_200
    ) == pytest.approx(0.00525)
    assert all(spec.price.schedule_id and spec.price.provenance_url for spec in SPECS.values())


def test_gemini_reservation_can_explicitly_bound_thought_tokens() -> None:
    reservation = maximum_cost_reservation(
        "gemini",
        maximum_input_tokens=1_000,
        maximum_output_tokens=1_200,
        maximum_thought_tokens=400,
    )
    assert reservation == pytest.approx(0.00675)
    with pytest.raises(ValueError, match="only to Gemini"):
        maximum_cost_reservation(
            "openai",
            maximum_input_tokens=1_000,
            maximum_output_tokens=1_200,
            maximum_thought_tokens=400,
        )


def test_provenance_records_date_headers_id_and_raw_hash() -> None:
    raw = {
        "id": "msg_2",
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": CONTENT}],
        "_version_headers": {"anthropic-version": "2023-06-01"},
    }
    provenance = parse_response("anthropic", raw).provenance
    assert provenance.retrieved_on == RETRIEVED_ON
    assert provenance.response_id == "msg_2"
    assert provenance.version_headers == (("anthropic-version", "2023-06-01"),)
    assert len(provenance.raw_response_sha256) == 64


def test_parser_is_total_on_wrong_typed_garbage() -> None:
    assert parse_response("openai", {}).status == "invalid"
    assert parse_response(
        "openai", {"model": "gpt-5.4-2026-03-05", "output": "wrong", "usage": []}
    ).status == "invalid"


@pytest.mark.parametrize("adapter_status", ["refused", "incomplete", "invalid"])
def test_non_success_adapter_statuses_bridge_to_non_retryable_invalid(
    adapter_status: str,
) -> None:
    raw = {
        "model": "claude-sonnet-5",
        "stop_reason": {
            "refused": "refusal",
            "incomplete": "max_tokens",
            "invalid": "pause_turn",
        }[adapter_status],
        "content": [{"type": "text", "text": CONTENT}],
    }
    normalized = parse_response("anthropic", raw)
    assert normalized.status == adapter_status
    bridged = to_provider_result(normalized, latency_ms=11)
    assert bridged.status == "invalid"
    assert bridged.content is None
    assert dict(bridged.response_metadata)["adapter_status"] == adapter_status


def test_only_transport_empty_bridge_retains_retryable_status() -> None:
    normalized = parse_response("anthropic", {"_transport_empty": True})
    bridged = to_provider_result(normalized, latency_ms=3)
    assert bridged.status == "transport_empty"
    assert bridged.content is None


def test_adapter_module_has_no_network_sdk_or_secret_lookup_imports() -> None:
    path = Path(__file__).parents[1] / "src/alpha_firewall/p3_adapter_contract.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = {"anthropic", "google", "httpx", "openai", "os", "requests", "socket"}
    assert imported.isdisjoint(forbidden)
