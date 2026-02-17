"""Ergodic TS — tools for time-series forecasting."""

__version__ = "0.1.0"

from ergodicts.snowflake_client import SnowflakeClient


def snowflake_client(**kwargs) -> SnowflakeClient:
    """Create a Snowflake client. Params default to env vars / .env file."""
    return SnowflakeClient(**kwargs)
