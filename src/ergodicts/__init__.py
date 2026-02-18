"""Ergodic TS — tools for time-series forecasting."""

__version__ = "0.1.0"

from ergodicts.snowflake_client import SnowflakeClient, TableExistsError
from ergodicts.utils import date_to_quarter_string, quarter_string_to_date
from ergodicts.reducer import (
    ModelKey,
    DependencyGraph,
    ReducerConfig,
    ReducerResult,
    ReducerPipeline,
    apply_reducer,
    check_harmonization,
)


def snowflake_client(**kwargs) -> SnowflakeClient:
    """Create a Snowflake client. Params default to env vars / .env file."""
    return SnowflakeClient(**kwargs)
