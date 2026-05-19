from io import StringIO

import pytest

from odefit.utils.progress import (
    ProgressReporter,
    format_duration,
    progress_iter,
)


def test_format_duration():
    assert format_duration(None) == "--:--:--"
    assert format_duration(0) == "00:00:00"
    assert format_duration(65) == "00:01:05"
    assert format_duration(3661) == "01:01:01"


def test_progress_reporter_updates():
    stream = StringIO()

    reporter = ProgressReporter(
        total=3,
        label="Test",
        enabled=True,
        stream=stream,
        min_interval_seconds=0.0,
    )

    reporter.update()
    reporter.update()
    reporter.update()
    reporter.close()

    output = stream.getvalue()

    assert "Test" in output
    assert "3/3" in output
    assert "100.0%" in output


def test_progress_reporter_rejects_negative_total():
    with pytest.raises(ValueError):
        ProgressReporter(total=-1)


def test_progress_reporter_rejects_negative_step():
    reporter = ProgressReporter(total=3, enabled=False)

    with pytest.raises(ValueError):
        reporter.update(step=-1)


def test_progress_iter():
    stream_items = list(
        progress_iter(
            iterable=[1, 2, 3],
            total=3,
            enabled=False,
        )
    )

    assert stream_items == [1, 2, 3]
