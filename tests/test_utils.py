"""Tests for ergodicts.utils — fiscal calendar helpers."""

import pandas as pd
import pytest

from ergodicts.utils import date_to_quarter_string, quarter_string_to_date


# ---------------------------------------------------------------------------
# date_to_quarter_string — QE-JUL (default)
# ---------------------------------------------------------------------------

class TestDateToQuarterStringQEJUL:
    """QE-JUL: FY starts Aug, Q1=Aug-Oct, Q2=Nov-Jan, Q3=Feb-Apr, Q4=May-Jul."""

    @pytest.mark.parametrize("date, expected", [
        ("2025-08-01", "2026Q1"),
        ("2025-09-15", "2026Q1"),
        ("2025-10-31", "2026Q1"),
    ])
    def test_q1(self, date, expected):
        assert date_to_quarter_string(date) == expected

    @pytest.mark.parametrize("date, expected", [
        ("2025-11-01", "2026Q2"),
        ("2025-12-15", "2026Q2"),
        ("2026-01-31", "2026Q2"),
    ])
    def test_q2(self, date, expected):
        assert date_to_quarter_string(date) == expected

    @pytest.mark.parametrize("date, expected", [
        ("2026-02-01", "2026Q3"),
        ("2026-03-15", "2026Q3"),
        ("2026-04-30", "2026Q3"),
    ])
    def test_q3(self, date, expected):
        assert date_to_quarter_string(date) == expected

    @pytest.mark.parametrize("date, expected", [
        ("2026-05-01", "2026Q4"),
        ("2026-06-15", "2026Q4"),
        ("2026-07-31", "2026Q4"),
    ])
    def test_q4(self, date, expected):
        assert date_to_quarter_string(date) == expected

    def test_boundary_jul_to_aug(self):
        """Jul 31 is Q4 of current FY; Aug 1 is Q1 of next FY."""
        assert date_to_quarter_string("2026-07-31") == "2026Q4"
        assert date_to_quarter_string("2026-08-01") == "2027Q1"


# ---------------------------------------------------------------------------
# quarter_string_to_date — QE-JUL (default)
# ---------------------------------------------------------------------------

class TestQuarterStringToDateQEJUL:

    @pytest.mark.parametrize("quarter, expected", [
        ("FY2026.Q1", "2025-10-31"),
        ("FY2026.Q2", "2026-01-31"),
        ("FY2026.Q3", "2026-04-30"),
        ("FY2026.Q4", "2026-07-31"),
    ])
    def test_fy_format(self, quarter, expected):
        assert quarter_string_to_date(quarter) == pd.Timestamp(expected)

    @pytest.mark.parametrize("quarter, expected", [
        ("2026Q1", "2025-10-31"),
        ("2026Q2", "2026-01-31"),
        ("2026Q3", "2026-04-30"),
        ("2026Q4", "2026-07-31"),
    ])
    def test_compact_format(self, quarter, expected):
        assert quarter_string_to_date(quarter) == pd.Timestamp(expected)

    def test_both_formats_agree(self):
        assert quarter_string_to_date("FY2026.Q1") == quarter_string_to_date("2026Q1")


# ---------------------------------------------------------------------------
# Roundtrip: date → quarter string → quarter-end date
# ---------------------------------------------------------------------------

class TestRoundtrip:

    @pytest.mark.parametrize("date, fy_end_month", [
        ("2025-08-15", 7),
        ("2026-01-10", 7),
        ("2026-07-31", 7),
        ("2026-03-15", 12),
        ("2026-12-31", 12),
        ("2025-02-15", 1),
    ])
    def test_date_falls_within_its_quarter(self, date, fy_end_month):
        """The original date must be <= the quarter-end date we compute."""
        qs = date_to_quarter_string(date, fy_end_month=fy_end_month)
        end = quarter_string_to_date(qs, fy_end_month=fy_end_month)
        assert pd.Timestamp(date) <= end


# ---------------------------------------------------------------------------
# Other fiscal year-end months
# ---------------------------------------------------------------------------

class TestOtherFYEndMonths:

    def test_qe_dec_calendar_year(self):
        assert date_to_quarter_string("2026-01-15", fy_end_month=12) == "2026Q1"
        assert date_to_quarter_string("2026-06-15", fy_end_month=12) == "2026Q2"
        assert date_to_quarter_string("2026-09-15", fy_end_month=12) == "2026Q3"
        assert date_to_quarter_string("2026-12-31", fy_end_month=12) == "2026Q4"

    def test_qe_dec_quarter_end_dates(self):
        assert quarter_string_to_date("2026Q1", fy_end_month=12) == pd.Timestamp("2026-03-31")
        assert quarter_string_to_date("2026Q4", fy_end_month=12) == pd.Timestamp("2026-12-31")

    def test_qe_jan(self):
        assert date_to_quarter_string("2025-02-15", fy_end_month=1) == "2026Q1"
        assert date_to_quarter_string("2026-01-31", fy_end_month=1) == "2026Q4"

    def test_qe_jun(self):
        assert date_to_quarter_string("2025-07-15", fy_end_month=6) == "2026Q1"
        assert date_to_quarter_string("2026-06-30", fy_end_month=6) == "2026Q4"


# ---------------------------------------------------------------------------
# Validation / edge cases
# ---------------------------------------------------------------------------

class TestValidation:

    def test_invalid_fy_end_month(self):
        with pytest.raises(ValueError, match="fy_end_month must be 1-12"):
            date_to_quarter_string("2026-01-01", fy_end_month=0)

    def test_invalid_quarter_string(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            quarter_string_to_date("not-a-quarter")

    def test_invalid_quarter_number(self):
        with pytest.raises(ValueError, match="Quarter must be 1-4"):
            quarter_string_to_date("2026Q5")

    def test_leap_year_feb(self):
        """Q3 end in a leap year should land on Feb 29 when applicable."""
        # QE-OCT: fy_start=Nov, Q1=Nov-Jan, Q2=Feb-Apr → Q2 end = Apr 30
        # QE-NOV: fy_start=Dec, Q1=Dec-Feb → Q1 end = Feb 28/29
        end = quarter_string_to_date("2024Q1", fy_end_month=11)
        assert end == pd.Timestamp("2024-02-29")

        end = quarter_string_to_date("2025Q1", fy_end_month=11)
        assert end == pd.Timestamp("2025-02-28")
