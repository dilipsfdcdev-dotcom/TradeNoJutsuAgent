"""Veto Register — bridge between Claude (slow loop) and execution (fast loop).

Claude writes asynchronously. Fast loop reads in <0.1ms.
Backed by Redis for persistence, falls back to in-memory dict.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
import json
import structlog

logger = structlog.get_logger()


@dataclass
class VetoState:
    blocked: bool = False
    reason: str = ""
    risk_modifier: float = 1.0      # 1.0 = normal, 0.5 = half, 0 = blocked
    expires_at: Optional[datetime] = None
    source: str = ""
    set_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked, "reason": self.reason,
            "risk_modifier": self.risk_modifier,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "source": self.source, "set_at": self.set_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VetoState":
        exp = data.get("expires_at")
        sat = data.get("set_at")
        return cls(
            blocked=data.get("blocked", False),
            reason=data.get("reason", ""),
            risk_modifier=data.get("risk_modifier", 1.0),
            expires_at=datetime.fromisoformat(exp) if exp else None,
            source=data.get("source", ""),
            set_at=datetime.fromisoformat(sat) if sat else datetime.now(timezone.utc),
        )


class VetoRegister:
    """Thread-safe veto register. Slow loop writes, fast loop reads."""

    _REDIS_PREFIX = "veto:"

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._local: dict[str, VetoState] = {}  # fallback if Redis down

    def check(self, symbol: str) -> VetoState:
        """Called by fast loop. Returns combined veto state. <0.1ms."""
        now = datetime.now(timezone.utc)
        global_veto = self._get(symbol="GLOBAL")
        symbol_veto = self._get(symbol=symbol)

        # Auto-expire stale vetoes
        for key, v in [("GLOBAL", global_veto), (symbol, symbol_veto)]:
            if v and v.expires_at and now > v.expires_at:
                self._clear(key)
                if key == "GLOBAL":
                    global_veto = VetoState()
                else:
                    symbol_veto = VetoState()

        # Global block overrides everything
        if global_veto and global_veto.blocked:
            return global_veto

        # Symbol-specific block
        if symbol_veto and symbol_veto.blocked:
            return symbol_veto

        # Risk modifier (even if not blocked)
        result = VetoState(blocked=False)
        if symbol_veto and symbol_veto.risk_modifier != 1.0:
            result.risk_modifier = symbol_veto.risk_modifier
            result.reason = symbol_veto.reason

        return result

    def set_veto(self, symbol: str, state: VetoState) -> None:
        """Called by slow loop (Claude veto scanner)."""
        self._set(symbol, state)
        logger.info("veto_set", symbol=symbol, blocked=state.blocked,
                     reason=state.reason, modifier=state.risk_modifier)

    def clear_veto(self, symbol: str) -> None:
        self._clear(symbol)
        logger.info("veto_cleared", symbol=symbol)

    def get_all(self) -> dict[str, VetoState]:
        """Dashboard reads all current vetoes."""
        return dict(self._local)

    def _get(self, symbol: str) -> VetoState | None:
        # Try Redis first
        if self._redis:
            try:
                raw = self._redis.get(f"{self._REDIS_PREFIX}{symbol}")
                if raw:
                    data = json.loads(raw)
                    state = VetoState.from_dict(data)
                    # Sync to local cache
                    self._local[symbol] = state
                    return state
            except Exception:
                # Redis down -- fall through to local
                pass
        return self._local.get(symbol)

    def _set(self, symbol: str, state: VetoState) -> None:
        self._local[symbol] = state
        if self._redis:
            try:
                payload = json.dumps(state.to_dict())
                # TTL: if expires_at is set use it, otherwise 24h default
                if state.expires_at:
                    ttl_secs = max(int((state.expires_at - datetime.now(timezone.utc)).total_seconds()), 60)
                else:
                    ttl_secs = 86400
                self._redis.setex(f"{self._REDIS_PREFIX}{symbol}", ttl_secs, payload)
            except Exception as exc:
                logger.warning("redis_veto_set_failed", symbol=symbol, error=str(exc))

    def _clear(self, symbol: str) -> None:
        self._local.pop(symbol, None)
        if self._redis:
            try:
                self._redis.delete(f"{self._REDIS_PREFIX}{symbol}")
            except Exception:
                pass


# Module-level singleton
veto_register = VetoRegister()
