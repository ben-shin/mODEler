from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


ProgressCallback = Callable[["ProgressEvent"], None]


@dataclass
class ProgressEvent:
    stage: str
    message: str
    current: int
    total: int | None
    elapsed_seconds: float
    eta_seconds: float | None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def fraction(self) -> float | None:
        if self.total is None or self.total <= 0:
            return None

        return min(max(self.current / self.total, 0.0), 1.0)

    @property
    def percent(self) -> float | None:
        if self.fraction is None:
            return None

        return 100.0 * self.fraction

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_console_line(self) -> str:
        if self.total is None:
            progress = f"{self.current}"
        else:
            progress = f"{self.current}/{self.total}"

        if self.percent is None:
            percent = ""
        else:
            percent = f" ({self.percent:5.1f}%)"

        eta = (
            "ETA --:--"
            if self.eta_seconds is None
            else f"ETA {format_duration(self.eta_seconds)}"
        )

        return (
            f"[{self.stage}] {progress}{percent} "
            f"elapsed {format_duration(self.elapsed_seconds)} "
            f"{eta} - {self.message}"
        )


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"

    seconds = max(float(seconds), 0.0)

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


class ProgressTracker:
    def __init__(
        self,
        *,
        stage: str,
        total: int | None,
        callback: ProgressCallback | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.total = total
        self.callback = callback
        self.payload = dict(payload or {})
        self.start_time = time.perf_counter()

    def make_event(
        self,
        *,
        current: int,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> ProgressEvent:
        elapsed = time.perf_counter() - self.start_time

        eta = estimate_eta(
            current=current,
            total=self.total,
            elapsed_seconds=elapsed,
        )

        merged_payload = dict(self.payload)

        if payload:
            merged_payload.update(payload)

        return ProgressEvent(
            stage=self.stage,
            message=message,
            current=current,
            total=self.total,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            payload=merged_payload,
        )

    def emit(
        self,
        *,
        current: int,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> ProgressEvent:
        event = self.make_event(
            current=current,
            message=message,
            payload=payload,
        )

        if self.callback is not None:
            self.callback(event)

        return event


def estimate_eta(
    *,
    current: int,
    total: int | None,
    elapsed_seconds: float,
) -> float | None:
    if total is None or total <= 0:
        return None

    if current <= 0:
        return None

    if current >= total:
        return 0.0

    rate = elapsed_seconds / current
    remaining = total - current

    return rate * remaining


def print_progress_event(event: ProgressEvent) -> None:
    print(event.to_console_line())
