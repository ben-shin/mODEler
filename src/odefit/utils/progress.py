from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, TextIO, TypeVar


T = TypeVar("T")


@dataclass
class ProgressStatus:
    """
    Current progress status.
    """

    completed: int
    total: int
    elapsed_seconds: float
    estimated_remaining_seconds: float | None
    rate_per_second: float | None

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0

        return min(1.0, max(0.0, self.completed / self.total))

    @property
    def percent(self) -> float:
        return 100.0 * self.fraction


def format_duration(seconds: float | None) -> str:
    """
    Format seconds as HH:MM:SS.

    None is shown as unknown.
    """

    if seconds is None:
        return "--:--:--"

    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


class ProgressReporter:
    """
    Lightweight dependency-free progress reporter with ETA.

    Designed for CLI long-running tasks such as multistart fitting.

    It prints a single updating line to stderr by default.
    """

    def __init__(
        self,
        total: int,
        label: str = "Progress",
        enabled: bool = True,
        stream: TextIO | None = None,
        min_interval_seconds: float = 0.25,
    ) -> None:
        if total < 0:
            raise ValueError("total must be non-negative")

        self.total = int(total)
        self.label = label
        self.enabled = enabled
        self.stream = sys.stderr if stream is None else stream
        self.min_interval_seconds = float(min_interval_seconds)

        self.completed = 0
        self.start_time = time.monotonic()
        self.last_render_time = 0.0
        self.closed = False

    def get_status(self) -> ProgressStatus:
        """
        Get current progress status.
        """

        elapsed = time.monotonic() - self.start_time

        if self.completed > 0 and elapsed > 0:
            rate = self.completed / elapsed
        else:
            rate = None

        if rate is not None and rate > 0 and self.total > 0:
            remaining = max(0, self.total - self.completed) / rate
        else:
            remaining = None

        return ProgressStatus(
            completed=self.completed,
            total=self.total,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=remaining,
            rate_per_second=rate,
        )

    def render(self, force: bool = False) -> None:
        """
        Render progress line.
        """

        if not self.enabled or self.closed:
            return

        now = time.monotonic()

        if (
            not force
            and now - self.last_render_time < self.min_interval_seconds
            and self.completed < self.total
        ):
            return

        self.last_render_time = now
        status = self.get_status()

        rate_text = "--/s"

        if status.rate_per_second is not None:
            rate_text = f"{status.rate_per_second:.2f}/s"

        message = (
            f"\r{self.label}: "
            f"{status.completed}/{status.total} "
            f"({status.percent:5.1f}%) "
            f"elapsed {format_duration(status.elapsed_seconds)} "
            f"ETA {format_duration(status.estimated_remaining_seconds)} "
            f"rate {rate_text}"
        )

        self.stream.write(message)
        self.stream.flush()

    def update(self, step: int = 1) -> None:
        """
        Advance progress.
        """

        if step < 0:
            raise ValueError("step must be non-negative")

        self.completed = min(self.total, self.completed + step)
        self.render(force=self.completed >= self.total)

    def close(self) -> None:
        """
        Finish progress display.
        """

        if self.closed:
            return

        self.render(force=True)

        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()

        self.closed = True

    def __enter__(self) -> ProgressReporter:
        self.render(force=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def progress_iter(
    iterable: Iterable[T],
    total: int,
    label: str = "Progress",
    enabled: bool = True,
) -> Iterator[T]:
    """
    Iterate with progress reporting.
    """

    with ProgressReporter(
        total=total,
        label=label,
        enabled=enabled,
    ) as reporter:
        for item in iterable:
            yield item
            reporter.update()
