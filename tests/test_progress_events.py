from odefit.progress import (
    ProgressEvent,
    ProgressTracker,
    estimate_eta,
    format_duration,
)


def test_format_duration():
    assert format_duration(0) == "00:00"
    assert format_duration(65) == "01:05"
    assert format_duration(3661) == "01:01:01"


def test_estimate_eta():
    assert estimate_eta(current=0, total=10, elapsed_seconds=5.0) is None
    assert estimate_eta(current=10, total=10, elapsed_seconds=5.0) == 0.0
    assert estimate_eta(current=5, total=10, elapsed_seconds=20.0) == 20.0


def test_progress_event_to_dict_and_console_line():
    event = ProgressEvent(
        stage="test",
        message="Working",
        current=2,
        total=4,
        elapsed_seconds=10.0,
        eta_seconds=10.0,
        payload={"model": "A"},
    )

    payload = event.to_dict()

    assert payload["stage"] == "test"
    assert payload["payload"]["model"] == "A"
    assert event.percent == 50.0
    assert "2/4" in event.to_console_line()


def test_progress_tracker_emits_callback():
    events = []

    tracker = ProgressTracker(
        stage="test_stage",
        total=2,
        callback=events.append,
    )

    tracker.emit(current=0, message="start")
    tracker.emit(current=1, message="middle")
    tracker.emit(current=2, message="done")

    assert len(events) == 3
    assert events[0].stage == "test_stage"
    assert events[-1].current == 2
    assert events[-1].eta_seconds == 0.0
