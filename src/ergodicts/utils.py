"""Fiscal calendar utilities for ergodicts."""

from __future__ import annotations

import calendar

import pandas as pd


def date_to_quarter_string(date: str, *, fy_end_month: int = 7) -> str:
    """Return the fiscal quarter string in ``YYYYQX`` format.

    Parameters
    ----------
    date : str
        Any date string parseable by ``pd.to_datetime``.
    fy_end_month : int
        Calendar month in which the fiscal year ends (1-12).
        Defaults to 7 (QE-JUL).

    Examples
    --------
    ```python
    date_to_quarter_string("2025-08-15")           # '2026Q1'
    date_to_quarter_string("2026-01-10")           # '2026Q2'
    date_to_quarter_string("2026-03-15", fy_end_month=12)  # '2026Q1'
    ```
    """
    _validate_fy_end_month(fy_end_month)

    dt = pd.to_datetime(date)
    month, year = dt.month, dt.year

    fy_start_month = (fy_end_month % 12) + 1
    months_into_fy = (month - fy_start_month) % 12
    quarter = months_into_fy // 3 + 1
    fiscal_year = year + 1 if month > fy_end_month else year

    return f"{fiscal_year}Q{quarter}"


def quarter_string_to_date(quarter: str, *, fy_end_month: int = 7) -> pd.Timestamp:
    """Convert a fiscal quarter string to the quarter-end date.

    Accepts formats ``FYXXXX.QY`` or ``YYYYQX``.

    Parameters
    ----------
    quarter : str
        Fiscal quarter identifier, e.g. ``"FY2026.Q1"`` or ``"2026Q1"``.
    fy_end_month : int
        Calendar month in which the fiscal year ends (1-12).
        Defaults to 7 (QE-JUL).

    Examples
    --------
    ```python
    quarter_string_to_date("FY2026.Q1")  # Timestamp('2025-10-31')
    quarter_string_to_date("2026Q2")     # Timestamp('2026-01-31')
    ```
    """
    _validate_fy_end_month(fy_end_month)
    fiscal_year, quarter_num = _parse_quarter_string(quarter)

    fy_start_month = (fy_end_month % 12) + 1
    end_month = ((fy_start_month - 1 + quarter_num * 3 - 1) % 12) + 1
    end_year = fiscal_year - 1 if end_month > fy_end_month else fiscal_year
    last_day = calendar.monthrange(end_year, end_month)[1]

    return pd.Timestamp(year=end_year, month=end_month, day=last_day)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_fy_end_month(fy_end_month: int) -> None:
    if not 1 <= fy_end_month <= 12:
        raise ValueError(f"fy_end_month must be 1-12, got {fy_end_month}")


def _parse_quarter_string(quarter: str) -> tuple[int, int]:
    """Parse ``FYXXXX.QY`` or ``YYYYQX`` → (fiscal_year, quarter_num)."""
    q = quarter.strip().upper()
    try:
        if "." in q:
            # FYXXXX.QY
            fy_part, q_part = q.split(".")
            fiscal_year = int(fy_part.replace("FY", ""))
            quarter_num = int(q_part.replace("Q", ""))
        elif "Q" in q:
            # YYYYQX
            parts = q.split("Q")
            fiscal_year = int(parts[0])
            quarter_num = int(parts[1])
        else:
            raise ValueError()
    except (ValueError, IndexError):
        raise ValueError(f"Cannot parse quarter string: {quarter!r}") from None

    if not 1 <= quarter_num <= 4:
        raise ValueError(f"Quarter must be 1-4, got {quarter_num}")

    return fiscal_year, quarter_num
