from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from threading import RLock
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class TargetHandoff:
    label: str | None
    normalized_label: str | None
    confidence: float
    transcript: str
    source_audio: str | None = None
    reason: str | None = None
    raw_output: str | None = None
    received_at: float = field(default_factory=perf_counter)


class TargetHandoffBus:
    def __init__(self) -> None:
        self._lock = RLock()
        self._latest: TargetHandoff | None = None
        self._version = 0
        self._consumed_version = 0

    def publish(self, target: TargetHandoff) -> None:
        with self._lock:
            self._latest = target
            self._version += 1

    def consume_latest(self) -> TargetHandoff | None:
        with self._lock:
            if self._latest is None or self._version == self._consumed_version:
                return None

            self._consumed_version = self._version
            return self._latest

    def peek_latest(self) -> TargetHandoff | None:
        with self._lock:
            return self._latest


def _clean_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def coerce_target_handoff(
    transcript: str,
    result: dict[str, Any],
    source_audio: str | None = None,
) -> TargetHandoff:
    label = _clean_label(result.get("object"))
    normalized_label = _clean_label(result.get("normalized_object")) or label
    reason = _clean_label(result.get("reason"))
    raw_output = _clean_label(result.get("raw_output"))

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return TargetHandoff(
        label=label,
        normalized_label=normalized_label,
        confidence=confidence,
        transcript=transcript,
        source_audio=source_audio,
        reason=reason,
        raw_output=raw_output,
    )
