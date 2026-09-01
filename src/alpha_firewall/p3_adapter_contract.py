"""Offline provider-dialect contract for the prospective P3 benchmark.

This module deliberately contains no transport, SDK, credential, environment, or
filesystem code.  It maps frozen prompt/schema inputs to JSON request bodies and
maps hand-supplied response dictionaries to one provider-neutral record.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from alpha_firewall.p3_harness import ProvenanceError, ProviderResult, canonical_json

RETRIEVED_ON = "2026-08-30"


@dataclass(frozen=True, slots=True)
class PriceSchedule:
    provider: str
    model: str
    input_usd_per_million: float
    output_usd_per_million: float
    schedule_id: str
    provenance_url: str

    def estimate(
        self, *, input_tokens: int, output_tokens: int, thought_tokens: int = 0
    ) -> float:
        if min(input_tokens, output_tokens, thought_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        billable_output = output_tokens + thought_tokens
        return (
            input_tokens * self.input_usd_per_million
            + billable_output * self.output_usd_per_million
        ) / 1_000_000.0


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    provider: Literal["anthropic", "openai", "gemini"]
    endpoint: str
    requested_model: str
    provenance_url: str
    price: PriceSchedule


@dataclass(frozen=True, slots=True)
class ResponseProvenance:
    provider: str
    endpoint: str
    requested_model: str
    canonical_model: str
    retrieved_on: str
    documentation_url: str
    response_id: str | None
    version_headers: tuple[tuple[str, str], ...]
    raw_response_sha256: str


@dataclass(frozen=True, slots=True)
class NormalizedResponse:
    status: Literal["ok", "refused", "incomplete", "invalid", "transport_empty"]
    content: str | None
    input_tokens: int
    output_tokens: int
    thought_tokens: int
    estimated_cost_usd: float
    finish_reason: str
    provenance: ResponseProvenance


SPECS: dict[str, AdapterSpec] = {
    "anthropic": AdapterSpec(
        provider="anthropic",
        endpoint="/v1/messages",
        requested_model="claude-sonnet-5",
        provenance_url="https://platform.claude.com/docs/en/api/messages/create",
        price=PriceSchedule(
            provider="anthropic",
            model="claude-sonnet-5",
            input_usd_per_million=2.0,
            output_usd_per_million=10.0,
            schedule_id="anthropic-standard-2026-08-30",
            provenance_url="https://platform.claude.com/docs/en/about-claude/pricing",
        ),
    ),
    "openai": AdapterSpec(
        provider="openai",
        endpoint="/v1/responses",
        requested_model="gpt-5.4-2026-03-05",
        provenance_url="https://developers.openai.com/api/docs/models/gpt-5.4",
        price=PriceSchedule(
            provider="openai",
            model="gpt-5.4-2026-03-05",
            input_usd_per_million=2.5,
            output_usd_per_million=15.0,
            schedule_id="openai-standard-2026-08-30",
            provenance_url="https://developers.openai.com/api/docs/models/gpt-5.4",
        ),
    ),
    "gemini": AdapterSpec(
        provider="gemini",
        endpoint="/v1beta/interactions",
        requested_model="gemini-3.6-flash",
        provenance_url="https://ai.google.dev/api/interactions-api",
        price=PriceSchedule(
            provider="gemini",
            model="gemini-3.6-flash",
            input_usd_per_million=0.75,
            output_usd_per_million=3.75,
            schedule_id="gemini-3.6-through-2026-12-31-retrieved-2026-08-30",
            provenance_url="https://ai.google.dev/gemini-api/docs/pricing",
        ),
    ),
}


def schema_fingerprint(schema: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(schema).encode("utf-8")).hexdigest()


def build_request(
    provider: str, *, prompt: str, schema: dict[str, Any], max_output_tokens: int = 1200
) -> dict[str, Any]:
    """Build a secret-free JSON body for a frozen endpoint dialect."""
    if provider not in SPECS:
        raise ValueError(f"unsupported provider: {provider}")
    if not prompt or max_output_tokens <= 0:
        raise ValueError("prompt must be non-empty and max_output_tokens positive")
    spec = SPECS[provider]
    if provider == "anthropic":
        # Sonnet 5 uses adaptive thinking by default. Manual thinking and sampling
        # controls are intentionally omitted because this endpoint rejects them.
        return {
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "model": spec.requested_model,
            "output_config": {
                "format": {"type": "json_schema", "schema": schema}
            },
        }
    if provider == "openai":
        return {
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "model": spec.requested_model,
            "reasoning": {"effort": "none"},
            "store": False,
            "text": {
                "format": {
                    "name": "p3_strategy",
                    "schema": schema,
                    "strict": True,
                    "type": "json_schema",
                }
            },
        }
    # Gemini 3.6 exposes reasoning controls through generation_config. Minimal
    # reasoning preserves the frozen no-extended-thinking intent; summaries are
    # omitted so no reasoning text can enter the candidate channel.
    return {
        "generation_config": {
            "max_output_tokens": max_output_tokens,
            "thinking_level": "minimal",
            "thinking_summaries": "none",
            "tool_choice": "none",
        },
        "input": prompt,
        "model": spec.requested_model,
        "response_format": {
            "mime_type": "application/json",
            "schema": schema,
            "type": "text",
        },
        "store": False,
    }


def request_sha256(body: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def maximum_cost_reservation(
    provider: str,
    *,
    maximum_input_tokens: int,
    maximum_output_tokens: int,
    maximum_thought_tokens: int = 0,
) -> float:
    """Conservative uncached price reservation from the dated public schedule."""
    if provider not in SPECS:
        raise ValueError(f"unsupported provider: {provider}")
    if provider != "gemini" and maximum_thought_tokens:
        raise ValueError("separate thought-token reservation applies only to Gemini")
    return SPECS[provider].price.estimate(
        input_tokens=maximum_input_tokens,
        output_tokens=maximum_output_tokens,
        thought_tokens=maximum_thought_tokens,
    )


def to_provider_result(response: NormalizedResponse, *, latency_ms: int) -> ProviderResult:
    """Bridge parsed dialect output into the frozen harness failure machine.

    Refusal, incomplete, and invalid responses deliberately collapse to the
    harness's non-retryable ``invalid`` class. Only a content-free transport
    failure retains ``transport_empty`` and can receive the one allowed retry.
    """
    status: Literal["ok", "transport_empty", "invalid"]
    if response.status == "ok":
        status = "ok"
    elif response.status == "transport_empty":
        status = "transport_empty"
    else:
        status = "invalid"
    metadata = (
        ("adapter_status", response.status),
        ("finish_reason", response.finish_reason),
        ("thought_tokens", str(response.thought_tokens)),
        ("raw_response_sha256", response.provenance.raw_response_sha256),
        ("retrieved_on", response.provenance.retrieved_on),
    )
    return ProviderResult(
        status=status,
        content=response.content if status == "ok" else None,
        canonical_model=response.provenance.canonical_model,
        actual_cost_usd=response.estimated_cost_usd,
        latency_ms=latency_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens + response.thought_tokens,
        version_headers=response.provenance.version_headers,
        response_metadata=metadata,
    )


def _integer(mapping: Any, key: str) -> int:
    if not isinstance(mapping, dict):
        return 0
    value = mapping.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _version_headers(raw: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    value = raw.get("_version_headers", {})
    if not isinstance(value, dict):
        return ()
    return tuple(sorted((str(key), str(item)) for key, item in value.items()))


def _has_returned_content(raw: dict[str, Any]) -> bool:
    return any(raw.get(key) not in (None, "", [], {}) for key in ("content", "output", "steps"))


def _provenance(provider: str, raw: dict[str, Any], canonical_model: str) -> ResponseProvenance:
    spec = SPECS[provider]
    return ResponseProvenance(
        provider=provider,
        endpoint=spec.endpoint,
        requested_model=spec.requested_model,
        canonical_model=canonical_model,
        retrieved_on=RETRIEVED_ON,
        documentation_url=spec.provenance_url,
        response_id=str(raw["id"]) if raw.get("id") is not None else None,
        version_headers=_version_headers(raw),
        raw_response_sha256=hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest(),
    )


def _model_or_status(provider: str, raw: dict[str, Any]) -> tuple[str, str | None]:
    model = raw.get("model")
    if isinstance(model, str) and model:
        if model != SPECS[provider].requested_model:
            raise ProvenanceError(
                f"{provider} returned {model!r}, expected {SPECS[provider].requested_model!r}"
            )
        return model, None
    if raw.get("_transport_empty") is True and not _has_returned_content(raw):
        return "", "transport_empty"
    return "", "invalid"


def _single_text(blocks: Any, *, text_type: str) -> tuple[str | None, bool]:
    if not isinstance(blocks, list):
        return None, False
    texts = [
        block.get("text")
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == text_type
        and isinstance(block.get("text"), str)
    ]
    return (texts[0], True) if len(texts) == 1 else (None, False)


def _normalized(
    provider: str,
    raw: dict[str, Any],
    *,
    canonical_model: str,
    status: Literal["ok", "refused", "incomplete", "invalid", "transport_empty"],
    content: str | None,
    input_tokens: int,
    output_tokens: int,
    thought_tokens: int,
    finish_reason: str,
) -> NormalizedResponse:
    price = SPECS[provider].price
    return NormalizedResponse(
        status=status,
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thought_tokens=thought_tokens,
        estimated_cost_usd=price.estimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thought_tokens=thought_tokens if provider == "gemini" else 0,
        ),
        finish_reason=finish_reason,
        provenance=_provenance(provider, raw, canonical_model),
    )


def parse_response(provider: str, raw: dict[str, Any]) -> NormalizedResponse:
    """Normalize one raw REST response; only an exact-model mismatch raises."""
    if provider not in SPECS:
        raise ValueError(f"unsupported provider: {provider}")
    if not isinstance(raw, dict):
        raw = {}
    model, early = _model_or_status(provider, raw)
    if early is not None:
        return _normalized(
            provider,
            raw,
            canonical_model=model,
            status=early,  # type: ignore[arg-type]
            content=None,
            input_tokens=0,
            output_tokens=0,
            thought_tokens=0,
            finish_reason=early,
        )
    if provider == "anthropic":
        return _parse_anthropic(raw, model)
    if provider == "openai":
        return _parse_openai(raw, model)
    return _parse_gemini(raw, model)


def _parse_anthropic(raw: dict[str, Any], model: str) -> NormalizedResponse:
    reason = raw.get("stop_reason") if isinstance(raw.get("stop_reason"), str) else ""
    usage = raw.get("usage", {})
    input_tokens = _integer(usage, "input_tokens")
    output_tokens = _integer(usage, "output_tokens")
    text, unique = _single_text(raw.get("content"), text_type="text")
    status: Literal["ok", "refused", "incomplete", "invalid", "transport_empty"]
    if reason == "refusal":
        status, text = "refused", None
    elif reason == "max_tokens":
        status, text = "incomplete", None
    elif reason == "end_turn" and unique:
        status = "ok"
    else:
        status, text = "invalid", None
    return _normalized(
        "anthropic",
        raw,
        canonical_model=model,
        status=status,
        content=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thought_tokens=0,
        finish_reason=reason,
    )


def _parse_openai(raw: dict[str, Any], model: str) -> NormalizedResponse:
    response_status = raw.get("status") if isinstance(raw.get("status"), str) else ""
    output = raw.get("output")
    text_blocks: list[dict[str, Any]] = []
    refused = False
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            refused = refused or any(
                isinstance(block, dict) and block.get("type") == "refusal" for block in content
            )
            text_blocks.extend(
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "output_text"
            )
    text, unique = _single_text(text_blocks, text_type="output_text")
    usage = raw.get("usage", {})
    input_tokens = _integer(usage, "input_tokens")
    output_tokens = _integer(usage, "output_tokens")
    details = usage.get("output_tokens_details", {}) if isinstance(usage, dict) else {}
    thought_tokens = _integer(details, "reasoning_tokens")
    if refused:
        status, text = "refused", None
    elif response_status in {"incomplete", "failed"}:
        status, text = "incomplete", None
    elif response_status == "completed" and unique:
        status = "ok"
    else:
        status, text = "invalid", None
    return _normalized(
        "openai",
        raw,
        canonical_model=model,
        status=status,
        content=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thought_tokens=thought_tokens,
        finish_reason=response_status,
    )


def _parse_gemini(raw: dict[str, Any], model: str) -> NormalizedResponse:
    response_status = raw.get("status") if isinstance(raw.get("status"), str) else ""
    steps = raw.get("steps")
    text_blocks: list[dict[str, Any]] = []
    refused = False
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            content = step.get("content")
            if not isinstance(content, list):
                continue
            refused = refused or any(
                isinstance(block, dict) and block.get("type") in {"refusal", "safety"}
                for block in content
            )
            text_blocks.extend(
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    text, unique = _single_text(text_blocks, text_type="text")
    usage = raw.get("usage", {})
    input_tokens = _integer(usage, "total_input_tokens")
    output_tokens = _integer(usage, "total_output_tokens")
    thought_tokens = _integer(usage, "total_thought_tokens")
    if refused:
        status, text = "refused", None
    elif response_status in {"incomplete", "failed"}:
        status, text = "incomplete", None
    elif response_status == "completed" and unique:
        status = "ok"
    else:
        status, text = "invalid", None
    return _normalized(
        "gemini",
        raw,
        canonical_model=model,
        status=status,
        content=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thought_tokens=thought_tokens,
        finish_reason=response_status,
    )
