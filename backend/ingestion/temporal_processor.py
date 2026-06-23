"""Temporal bucketing.

Assigns each document a calendar **quarter** window (e.g. "2026-Q1") and a
**month** tag (e.g. "2026-01"), plus the raw epoch. These coarse buckets are
what power side-by-side period comparisons; the raw `created_utc` powers exact
range filters. We also drop anything older than the configured horizon so the
graph stays focused on the windows we can reason about.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable, Iterator

from config import get_settings
from utils.logging import get_logger
from utils.models import RedditDocument

logger = get_logger("ingestion.temporal")
UTC = dt.timezone.utc


def quarter_label(d: dt.datetime) -> str:
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


def month_label(d: dt.datetime) -> str:
    return f"{d.year}-{d.month:02d}"


class TemporalProcessor:
    def __init__(self, now: dt.datetime | None = None) -> None:
        self._settings = get_settings()
        self._now = now or dt.datetime.now(tz=UTC)
        months_back = self._settings.time_window_months_back
        cutoff = self._now
        for _ in range(months_back):
            first = cutoff.replace(day=1)
            cutoff = first - dt.timedelta(days=1)
        self._cutoff_utc = cutoff.replace(day=1, hour=0, minute=0, second=0).timestamp()

    def process(self, docs: Iterable[RedditDocument]) -> Iterator[RedditDocument]:
        kept = dropped = 0
        for doc in docs:
            if doc.created_utc < self._cutoff_utc:
                dropped += 1
                continue
            d = doc.created_dt
            doc.time_window = quarter_label(d)
            doc.month = month_label(d)
            doc.year = d.year
            kept += 1
            yield doc
        logger.info("Temporal bucketing: kept=%d dropped(<horizon)=%d", kept, dropped)
