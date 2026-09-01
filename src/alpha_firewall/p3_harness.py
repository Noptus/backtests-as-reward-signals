from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from alpha_firewall.specs import (
    FAST_WINDOWS,
    REBALANCE_DAYS,
    SLOW_WINDOWS,
    TOP_K,
    VOL_WINDOWS,
    StrategySpec,
)

GENESIS_HASH = "0" * 64
SECRET_FIELD_FRAGMENTS = ("authorization", "api_key", "apikey", "secret", "token")


class CostCeilingError(RuntimeError):
    """Raised before a call whose reservation would exceed the frozen ceiling."""


class ProvenanceError(RuntimeError):
    """Raised when a provider does not return the frozen canonical model ID."""


class InformationFlowError(RuntimeError):
    """Raised when non-history information is offered to a model prompt."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HistoryItem:
    strategy: StrategySpec
    feedback_sharpe: float | None


@dataclass(frozen=True, slots=True)
class EpisodeKey:
    opaque_arm_id: str
    market_seed: int
    replicate: int
    round_index: int
    slot: int


@dataclass(frozen=True, slots=True)
class ProviderResult:
    status: Literal["ok", "transport_empty", "invalid"]
    content: str | None
    canonical_model: str
    actual_cost_usd: float
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    version_headers: tuple[tuple[str, str], ...] = ()
    response_metadata: tuple[tuple[str, str], ...] = ()


class Provider(Protocol):
    provider_name: str
    maximum_call_cost_usd: float

    def call(
        self,
        *,
        requested_model: str,
        prompt: str,
        schema: dict[str, Any],
        headers: dict[str, str],
    ) -> ProviderResult: ...


class MockProvider:
    """A deterministic scripted provider. It contains no network implementation."""

    def __init__(
        self,
        results: list[ProviderResult],
        *,
        provider_name: str = "mock",
        maximum_call_cost_usd: float = 0.25,
    ) -> None:
        self.provider_name = provider_name
        self.maximum_call_cost_usd = maximum_call_cost_usd
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def call(
        self,
        *,
        requested_model: str,
        prompt: str,
        schema: dict[str, Any],
        headers: dict[str, str],
    ) -> ProviderResult:
        self.calls.append(
            {
                "requested_model": requested_model,
                "prompt": prompt,
                "schema": schema,
                "headers": headers,
            }
        )
        if not self._results:
            raise AssertionError("mock provider has no scripted result remaining")
        return self._results.pop(0)


class CostLedger:
    def __init__(self, ceiling_usd: float) -> None:
        if ceiling_usd < 0:
            raise ValueError("cost ceiling must be non-negative")
        self.ceiling_usd = float(ceiling_usd)
        self.committed_usd = 0.0
        self.reserved_usd = 0.0

    def reserve(self, maximum_cost_usd: float) -> float:
        if maximum_cost_usd < 0:
            raise ValueError("reservation must be non-negative")
        projected = self.committed_usd + self.reserved_usd + maximum_cost_usd
        if projected > self.ceiling_usd + 1e-12:
            raise CostCeilingError(
                f"reservation {maximum_cost_usd:.6f} would exceed {self.ceiling_usd:.6f}"
            )
        self.reserved_usd += maximum_cost_usd
        return maximum_cost_usd

    def settle(self, reservation: float, actual_cost_usd: float) -> None:
        if actual_cost_usd < 0 or actual_cost_usd > reservation + 1e-12:
            raise CostCeilingError("actual cost is outside its pre-call reservation")
        self.reserved_usd -= reservation
        self.committed_usd += actual_cost_usd


class SecretRedactor:
    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        self.secrets = tuple(secret for secret in secrets if secret)

    def redact(self, value: Any, *, key: str = "") -> Any:
        lowered = key.lower()
        if any(fragment in lowered for fragment in SECRET_FIELD_FRAGMENTS):
            return "<REDACTED>"
        if isinstance(value, dict):
            return {name: self.redact(item, key=str(name)) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        if isinstance(value, str):
            result = value
            for secret in self.secrets:
                result = result.replace(secret, "<REDACTED>")
            return result
        return value


class HashChainLedger:
    def __init__(self, *, redactor: SecretRedactor, path: Path | None = None) -> None:
        self.redactor = redactor
        self.path = path
        self.entries: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        previous_hash = self.entries[-1]["entry_hash"] if self.entries else GENESIS_HASH
        body = {
            "previous_hash": previous_hash,
            "record": self.redactor.redact(record),
        }
        entry = {**body, "entry_hash": sha256_text(canonical_json(body))}
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(entry) + "\n")
        return entry

    def verify(self, entries: list[dict[str, Any]] | None = None) -> None:
        selected = self.entries if entries is None else entries
        previous_hash = GENESIS_HASH
        for index, entry in enumerate(selected):
            body = {"previous_hash": entry["previous_hash"], "record": entry["record"]}
            expected = sha256_text(canonical_json(body))
            if entry["previous_hash"] != previous_hash or entry["entry_hash"] != expected:
                raise ValueError(f"hash-chain verification failed at entry {index}")
            previous_hash = entry["entry_hash"]

    @classmethod
    def read_jsonl(cls, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    strategy: StrategySpec
    source: Literal["model", "fallback"]
    fallback_reason: str | None
    attempts: int


@dataclass(frozen=True, slots=True)
class ConfirmationReceipt:
    key: tuple[int, int]
    score: float


@dataclass(frozen=True, slots=True)
class BurnedConfirmation:
    key: tuple[int, int]
    promotion_eligible: Literal[False] = False
    intent_to_treat_failure: Literal[True] = True


class ConfirmationService:
    """A write-once audit service whose types are excluded from model history."""

    def __init__(self, scores: dict[tuple[int, int], float]) -> None:
        self._scores = dict(scores)
        self._queried: set[tuple[int, int]] = set()
        self._burned: set[tuple[int, int]] = set()

    def query(self, key: tuple[int, int]) -> ConfirmationReceipt | BurnedConfirmation:
        if key in self._queried or key in self._burned:
            self._burned.add(key)
            return BurnedConfirmation(key)
        if key not in self._scores:
            self._burned.add(key)
            return BurnedConfirmation(key)
        self._queried.add(key)
        return ConfirmationReceipt(key, self._scores[key])


def proposal_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "fast_window": {"type": "integer"},
            "slow_window": {"type": "integer"},
            "mix": {"type": "number"},
            "top_k": {"type": "integer"},
            "rebalance_days": {"type": "integer"},
            "vol_window": {"type": "integer"},
            "volatility_scale": {"type": "boolean"},
        },
        "required": [
            "fast_window",
            "slow_window",
            "mix",
            "top_k",
            "rebalance_days",
            "vol_window",
            "volatility_scale",
        ],
        "additionalProperties": False,
    }


def build_prompt(frozen_instruction: str, history: tuple[HistoryItem, ...]) -> str:
    if not all(type(item) is HistoryItem for item in history):
        raise InformationFlowError("only HistoryItem values may enter a provider prompt")
    payload = []
    for item in history:
        row: dict[str, Any] = {"strategy": item.strategy.to_dict()}
        if item.feedback_sharpe is not None:
            row["feedback_sharpe"] = round(float(item.feedback_sharpe), 6)
        payload.append(row)
    return frozen_instruction + "\nHISTORY:" + canonical_json(payload)


def _parse_strategy(content: str | None, seen: set[StrategySpec]) -> StrategySpec:
    if content is None:
        raise ValueError("missing model content")
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != set(proposal_schema()["required"]):
        raise ValueError("response does not match the frozen schema")
    spec = StrategySpec.from_dict(value)
    if spec in seen:
        raise ValueError("duplicate strategy")
    return spec


def _fallback_spec(key: EpisodeKey, base_seed: int, seen: set[StrategySpec]) -> StrategySpec:
    material = canonical_json(
        {
            "base_seed": base_seed,
            "market_seed": key.market_seed,
            "replicate": key.replicate,
            "opaque_arm_id": key.opaque_arm_id,
            "round": key.round_index,
            "slot": key.slot,
        }
    )
    rng = random.Random(int(sha256_text(material)[:16], 16))
    for _ in range(10_000):
        fast = rng.choice(FAST_WINDOWS)
        valid_slow = [value for value in SLOW_WINDOWS if value > fast]
        if not valid_slow:
            continue
        spec = StrategySpec(
            fast_window=fast,
            slow_window=rng.choice(valid_slow),
            mix=round(rng.uniform(-1.0, 1.0), 3),
            top_k=rng.choice(TOP_K),
            rebalance_days=rng.choice(REBALANCE_DAYS),
            vol_window=rng.choice(VOL_WINDOWS),
            volatility_scale=bool(rng.randrange(2)),
        )
        if spec not in seen:
            return spec
    raise RuntimeError("could not construct a unique deterministic fallback")


def execute_candidate(
    *,
    provider: Provider,
    requested_model: str,
    frozen_canonical_model: str,
    frozen_instruction: str,
    history: tuple[HistoryItem, ...],
    key: EpisodeKey,
    base_seed: int,
    cost_ledger: CostLedger,
    audit_ledger: HashChainLedger,
    headers: dict[str, str],
    executable_hash: str,
    price_schedule: str,
) -> CandidateOutcome:
    prompt = build_prompt(frozen_instruction, history)
    schema = proposal_schema()
    seen = {item.strategy for item in history}
    reason = "unknown"
    attempts = 0
    retry_reason: str | None = None
    for attempt in range(2):
        reservation = cost_ledger.reserve(provider.maximum_call_cost_usd)
        result = provider.call(
            requested_model=requested_model,
            prompt=prompt,
            schema=schema,
            headers=headers,
        )
        attempts += 1
        cost_ledger.settle(reservation, result.actual_cost_usd)
        parse_result = "not_attempted"
        parsed: StrategySpec | None = None
        if result.status == "ok":
            try:
                parsed = _parse_strategy(result.content, seen)
                parse_result = "valid"
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                reason = (
                    "duplicate_strategy"
                    if str(error) == "duplicate strategy"
                    else f"invalid_content:{type(error).__name__}"
                )
                parse_result = reason
        elif result.status == "transport_empty":
            if result.content not in (None, ""):
                reason = "transport_failure_returned_content"
            else:
                reason = "transport_empty"
        else:
            reason = "provider_invalid"
        audit_ledger.append(
            {
                "provider": provider.provider_name,
                "requested_model": requested_model,
                "canonical_model": result.canonical_model,
                "version_headers": dict(result.version_headers),
                "prompt_hash": sha256_text(prompt),
                "schema_hash": sha256_text(canonical_json(schema)),
                "opaque_arm_id": key.opaque_arm_id,
                "market_seed": key.market_seed,
                "replicate": key.replicate,
                "round": key.round_index,
                "slot": key.slot,
                "attempt": attempt,
                "request": {"prompt": prompt, "schema": schema, "headers": headers},
                "response": {
                    "status": result.status,
                    "content": result.content,
                    "metadata": dict(result.response_metadata),
                },
                "sampling_settings": {"temperature": 0.0, "max_output_tokens": 1_200},
                "latency_ms": result.latency_ms,
                "tokens": {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                },
                "price_schedule": price_schedule,
                "realized_cost_usd": result.actual_cost_usd,
                "retry_reason": retry_reason,
                "parse_result": parse_result,
                "duplicate_status": "duplicate" if "duplicate" in parse_result else "clear",
                "fallback_spec": None,
                "executable_hash": executable_hash,
                "evaluator_score": None,
            }
        )
        empty_transport = result.status == "transport_empty" and result.content in (None, "")
        if result.canonical_model != frozen_canonical_model and not empty_transport:
            raise ProvenanceError(
                f"expected canonical model {frozen_canonical_model!r}, "
                f"got {result.canonical_model!r}"
            )
        if parsed is not None:
            return CandidateOutcome(parsed, "model", None, attempts)
        may_retry = (
            attempt == 0
            and empty_transport
        )
        if not may_retry:
            break
        retry_reason = "transport_empty"
    fallback = _fallback_spec(key, base_seed, seen)
    audit_ledger.append(
        {
            "event": "intent_to_treat_fallback",
            "opaque_arm_id": key.opaque_arm_id,
            "market_seed": key.market_seed,
            "replicate": key.replicate,
            "round": key.round_index,
            "slot": key.slot,
            "fallback_reason": reason,
            "fallback_spec": fallback.to_dict(),
            "executable_hash": executable_hash,
        }
    )
    return CandidateOutcome(fallback, "fallback", reason, attempts)


def strategy_content(spec: StrategySpec) -> str:
    """Canonical mock response helper."""

    return canonical_json(asdict(spec))
