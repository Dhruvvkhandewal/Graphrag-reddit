"""Unit tests for the deterministic temporal parser."""
import datetime as dt

from retrieval.temporal import parse_temporal

NOW = dt.datetime(2026, 6, 22, tzinfo=dt.timezone.utc)


def test_past_n_months():
    # Mirrors the assignment example, which includes the cue word "changed".
    ranges, is_cmp = parse_temporal(
        "how has sentiment changed over the past 6 months", NOW
    )
    assert len(ranges) == 1
    r = ranges[0]
    assert r.start == dt.datetime(2025, 12, 1, tzinfo=dt.timezone.utc)
    assert r.end == NOW
    assert is_cmp  # "changed" flags evolution/comparison intent


def test_bare_window_is_not_comparison():
    ranges, is_cmp = parse_temporal("top posts in the past 6 months", NOW)
    assert len(ranges) == 1
    assert is_cmp is False


def test_explicit_quarters_comparison():
    ranges, is_cmp = parse_temporal(
        "concerns in Q1 2026 that weren't discussed in Q4 2025", NOW
    )
    labels = {r.label for r in ranges}
    assert labels == {"2026-Q1", "2025-Q4"}
    assert is_cmp
    q1 = next(r for r in ranges if r.label == "2026-Q1")
    assert q1.start == dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    assert q1.end == dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc)


def test_last_quarter_relative():
    ranges, _ = parse_temporal("what changed last quarter", NOW)
    assert ranges[0].label == "2026-Q1"  # prev quarter of Q2-2026


def test_whole_year():
    ranges, _ = parse_temporal("discussions in 2025", NOW)
    assert ranges[0].label == "2025"
    assert ranges[0].start.year == 2025
    assert ranges[0].end.year == 2026


def test_month_year():
    ranges, _ = parse_temporal("posts from March 2026", NOW)
    assert ranges[0].label == "2026-03"


def test_no_temporal_expression():
    ranges, is_cmp = parse_temporal("best open source LLM for coding", NOW)
    assert ranges == []
    assert is_cmp is False
