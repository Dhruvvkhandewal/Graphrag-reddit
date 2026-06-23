"""Temporal expression parser.

Turns natural-language time references inside a query into concrete,
half-open UTC intervals (`TimeRange`). This runs *deterministically* (no LLM)
so it is fast, free, and unit-testable — the LLM query-understanding layer
only adds intent/topic hints on top.

Supported expressions (case-insensitive):
    * "last/past N day(s)|week(s)|month(s)|year(s)"
    * "last/past quarter", "this quarter"
    * "Q1 2026", "Q4 2025"               (explicit quarters)
    * "in 2025", "2025"                  (whole year)
    * "<Month> <Year>", "<Month> YYYY"   (whole month)
    * comparison: two references in one query, or "vs"/"compared to"/
      "changed"/"evolved"/"shifted" -> is_comparison=True

Anything it cannot parse simply yields no ranges (retrieval falls back to the
full corpus), which is the safe default.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import List, Optional, Tuple

from retrieval.models import TimeRange

UTC = dt.timezone.utc

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}
_MONTHS.update({m[:3].lower(): i for m, i in list(_MONTHS.items())})

_COMPARISON_HINTS = re.compile(
    r"\b(vs\.?|versus|compared?\s+to|changed?|evolv\w*|shift\w*|"
    r"trend\w*|over\s+time|emerg\w*|appear\w*|weren'?t|did\s+not)\b",
    re.IGNORECASE,
)


def _add_months(d: dt.datetime, months: int) -> dt.datetime:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    return d.replace(year=year, month=month, day=1)


def _quarter_range(year: int, q: int) -> TimeRange:
    start = dt.datetime(year, (q - 1) * 3 + 1, 1, tzinfo=UTC)
    end = _add_months(start, 3)
    return TimeRange(label=f"{year}-Q{q}", start=start, end=end)


def _quarter_of(d: dt.datetime) -> int:
    return (d.month - 1) // 3 + 1


def parse_temporal(
    query: str, now: Optional[dt.datetime] = None
) -> Tuple[List[TimeRange], bool]:
    """Extract time ranges and whether the query is a period comparison.

    Returns:
        (ranges, is_comparison). `ranges` is ordered as found in the text.
    """
    now = now or dt.datetime.now(tz=UTC)
    q = query.lower()
    ranges: List[TimeRange] = []

    # --- relative: "last/past N <unit>" ---
    for m in re.finditer(
        r"\b(?:last|past|previous)\s+(\d+)\s+(day|week|month|year)s?\b", q
    ):
        amount = int(m.group(1))
        unit = m.group(2)
        ranges.append(_relative_range(now, amount, unit))

    # --- relative bare: "last week/month/year/quarter", "past 6 months" handled above ---
    if re.search(r"\b(?:last|past|previous)\s+quarter\b", q):
        py, pq = _previous_quarter(now)
        ranges.append(_quarter_range(py, pq))
    if re.search(r"\bthis\s+quarter\b", q):
        ranges.append(_quarter_range(now.year, _quarter_of(now)))
    for unit_word, unit in (("week", "week"), ("month", "month"), ("year", "year")):
        if re.search(rf"\b(?:last|past|previous)\s+{unit_word}\b", q):
            ranges.append(_relative_range(now, 1, unit))

    # --- explicit quarter: "Q1 2026" ---
    for m in re.finditer(r"\bq([1-4])\s*[\s/-]?\s*(\d{4})\b", q):
        ranges.append(_quarter_range(int(m.group(2)), int(m.group(1))))

    # --- explicit "<Month> <Year>" ---
    for m in re.finditer(r"\b([a-z]+)\s+(\d{4})\b", q):
        mon = _MONTHS.get(m.group(1))
        if mon:
            start = dt.datetime(int(m.group(2)), mon, 1, tzinfo=UTC)
            end = _add_months(start, 1)
            ranges.append(
                TimeRange(label=f"{m.group(2)}-{mon:02d}", start=start, end=end)
            )

    # --- whole year: "in 2025" / standalone 2025 (only if no quarter/month captured it) ---
    if not ranges:
        for m in re.finditer(r"\b(?:in\s+)?(20\d{2})\b", q):
            year = int(m.group(1))
            start = dt.datetime(year, 1, 1, tzinfo=UTC)
            end = dt.datetime(year + 1, 1, 1, tzinfo=UTC)
            ranges.append(TimeRange(label=str(year), start=start, end=end))

    # De-duplicate by label while preserving order.
    seen, deduped = set(), []
    for r in ranges:
        if r.label not in seen:
            seen.add(r.label)
            deduped.append(r)

    is_comparison = len(deduped) >= 2 or bool(_COMPARISON_HINTS.search(query))
    return deduped, is_comparison


def _relative_range(now: dt.datetime, amount: int, unit: str) -> TimeRange:
    if unit == "day":
        start = now - dt.timedelta(days=amount)
    elif unit == "week":
        start = now - dt.timedelta(weeks=amount)
    elif unit == "month":
        start = _add_months(now.replace(day=1), -amount)
    elif unit == "year":
        start = now.replace(year=now.year - amount)
    else:  # pragma: no cover - guarded by regex
        raise ValueError(unit)
    return TimeRange(label=f"last_{amount}_{unit}", start=start, end=now)


def _previous_quarter(now: dt.datetime) -> Tuple[int, int]:
    q = _quarter_of(now)
    if q == 1:
        return now.year - 1, 4
    return now.year, q - 1
