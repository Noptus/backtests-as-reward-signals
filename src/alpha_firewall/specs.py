from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

FAST_WINDOWS = (5, 10, 21, 42, 63)
SLOW_WINDOWS = (21, 42, 63, 126, 252)
TOP_K = (2, 3, 5, 7)
REBALANCE_DAYS = (1, 5, 10, 21)
VOL_WINDOWS = (10, 21, 42, 63)


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """A deliberately bounded, auditable cross-sectional strategy language."""

    fast_window: int
    slow_window: int
    mix: float
    top_k: int
    rebalance_days: int
    vol_window: int
    volatility_scale: bool

    def __post_init__(self) -> None:
        # ``vol_window`` has no effect for equal-weighted books. Canonicalizing
        # it prevents semantically duplicate trials from inflating the audit.
        if not self.volatility_scale:
            object.__setattr__(self, "vol_window", VOL_WINDOWS[0])
        allowed = {
            "fast_window": (self.fast_window, FAST_WINDOWS),
            "slow_window": (self.slow_window, SLOW_WINDOWS),
            "top_k": (self.top_k, TOP_K),
            "rebalance_days": (self.rebalance_days, REBALANCE_DAYS),
            "vol_window": (self.vol_window, VOL_WINDOWS),
        }
        for name, (value, values) in allowed.items():
            if value not in values:
                raise ValueError(f"{name}={value!r} is outside the frozen DSL")
        if not -1.0 <= self.mix <= 1.0:
            raise ValueError("mix must lie in [-1, 1]")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be shorter than slow_window")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StrategySpec:
        return cls(**value)

    @property
    def identifier(self) -> str:
        scale = "iv" if self.volatility_scale else "eq"
        mix = str(self.mix).replace("-", "m").replace(".", "p")
        return (
            f"f{self.fast_window}_s{self.slow_window}_m{mix}_k{self.top_k}_"
            f"r{self.rebalance_days}_v{self.vol_window}_{scale}"
        )
