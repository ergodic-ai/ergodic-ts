"""Snowflake client for ergodicts — thin wrapper around snowflake-connector-python."""

from __future__ import annotations

import os
import time
import logging
from typing import Any

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

import numpy as np

logger = logging.getLogger(__name__)


class TableExistsError(Exception):
    """Raised when attempting to create a table that already exists."""


def _pandas_dtype_to_snowflake(dtype: np.dtype) -> str:
    """Map a pandas/numpy dtype to a Snowflake column type."""
    if pd.api.types.is_integer_dtype(dtype):
        return "NUMBER"
    if pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP_NTZ"
    return "VARCHAR"

_ENV_MAP = {
    "account": "SNOWFLAKE_ACCOUNT",
    "user": "SNOWFLAKE_USER",
    "password": "SNOWFLAKE_PASSWORD",
    "role": "SNOWFLAKE_ROLE",
    "warehouse": "SNOWFLAKE_WAREHOUSE",
    "database": "SNOWFLAKE_DATABASE",
    "schema": "SNOWFLAKE_SCHEMA",
}


def _truncate_query(query: str, max_len: int = 200) -> str:
    """Return a single-line, truncated version of a query for log messages."""
    one_line = " ".join(query.split())
    if len(one_line) > max_len:
        return one_line[:max_len] + "..."
    return one_line


class SnowflakeClient:
    """Manages a Snowflake connection and exposes query helpers.

    Connection parameters can be passed explicitly; anything omitted is read
    from environment variables (a .env file is loaded automatically).

    Usage::

        client = SnowflakeClient()                   # all from env
        client = SnowflakeClient(warehouse="WH_XL")  # override one param
        df = client.get_data("SELECT * FROM my_table")
    """

    def __init__(
        self,
        *,
        account: str | None = None,
        user: str | None = None,
        password: str | None = None,
        role: str | None = None,
        warehouse: str | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        load_dotenv()

        self._params: dict[str, str] = {}
        supplied = {
            "account": account,
            "user": user,
            "password": password,
            "role": role,
            "warehouse": warehouse,
            "database": database,
            "schema": schema,
        }

        missing: list[str] = []
        for key, env_var in _ENV_MAP.items():
            value = supplied[key] or os.getenv(env_var)
            if not value:
                missing.append(env_var)
            else:
                self._params[key] = value

        if missing:
            raise EnvironmentError(
                f"Missing Snowflake config (pass explicitly or set in env/.env): "
                f"{', '.join(missing)}"
            )

        logger.debug(
            "Resolved config: account=%s user=%s role=%s warehouse=%s database=%s schema=%s",
            self._params["account"],
            self._params["user"],
            self._params.get("role"),
            self._params.get("warehouse"),
            self._params.get("database"),
            self._params.get("schema"),
        )

        self._conn: snowflake.connector.SnowflakeConnection | None = None
        self._connect()

    # -- connection lifecycle --------------------------------------------------

    def _connect(self) -> None:
        logger.info(
            "Connecting to Snowflake account=%s user=%s",
            self._params["account"],
            self._params["user"],
        )
        t0 = time.perf_counter()
        self._conn = snowflake.connector.connect(**self._params)
        elapsed = time.perf_counter() - t0
        logger.info("Connected in %.2fs", elapsed)

    def refresh(self) -> None:
        """Drop the current connection and establish a new one."""
        logger.info("Refreshing connection...")
        self.close()
        self._connect()

    def close(self) -> None:
        """Close the underlying connection if open."""
        if self._conn is not None and not self._conn.is_closed():
            logger.info("Closing connection.")
            self._conn.close()
            self._conn = None
            logger.debug("Connection closed.")

    @property
    def connection(self) -> snowflake.connector.SnowflakeConnection:
        """Return the live connection, reconnecting if it was dropped."""
        if self._conn is None or self._conn.is_closed():
            logger.warning("Connection lost — reconnecting automatically.")
            self._connect()
        return self._conn  # type: ignore[return-value]

    # -- query helpers ---------------------------------------------------------

    def get_data(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute *query* and return the result set as a DataFrame."""
        logger.info("get_data: %s", _truncate_query(query))
        if params:
            logger.debug("  params: %s", params)

        t0 = time.perf_counter()
        cur = self.connection.cursor()
        try:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            elapsed = time.perf_counter() - t0
            logger.info(
                "get_data: %d rows, %d cols returned in %.2fs",
                len(rows), len(columns), elapsed,
            )
            return pd.DataFrame(rows, columns=columns)
        except Exception:
            logger.exception("get_data failed: %s", _truncate_query(query))
            raise
        finally:
            cur.close()

    def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Execute a statement that returns no data (DDL / DML)."""
        logger.info("execute: %s", _truncate_query(query))
        if params:
            logger.debug("  params: %s", params)

        t0 = time.perf_counter()
        cur = self.connection.cursor()
        try:
            cur.execute(query, params)
            elapsed = time.perf_counter() - t0
            logger.info("execute: completed in %.2fs", elapsed)
        except Exception:
            logger.exception("execute failed: %s", _truncate_query(query))
            raise
        finally:
            cur.close()

    def create_table_from_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        *,
        overwrite: bool = False,
    ) -> None:
        """Write a DataFrame to a Snowflake table.

        Raises ``TableExistsError`` if the table already exists and
        *overwrite* is ``False`` (the default).
        """
        logger.info(
            "create_table_from_dataframe: table=%s rows=%d cols=%d overwrite=%s",
            table_name, len(df), len(df.columns), overwrite,
        )

        if not overwrite:
            exists = self.get_data(
                "SELECT 1 FROM information_schema.tables WHERE table_name = %s",
                {"1": table_name.upper()},
            )
            if not exists.empty:
                logger.warning("Table %s already exists and overwrite=False.", table_name)
                raise TableExistsError(
                    f"Table {table_name!r} already exists. "
                    f"Pass overwrite=True to replace it."
                )

        from snowflake.connector.pandas_tools import write_pandas

        if overwrite:
            logger.info("Dropping existing table %s (overwrite=True).", table_name)
            self.execute(f'DROP TABLE IF EXISTS "{table_name}"')

        # Build a CREATE TABLE from the DataFrame dtypes
        col_defs = ", ".join(
            f'"{col}" {_pandas_dtype_to_snowflake(df[col].dtype)}' for col in df.columns
        )
        self.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

        if not df.empty:
            logger.info("Writing %d rows to %s...", len(df), table_name)
            t0 = time.perf_counter()
            write_pandas(self.connection, df, table_name)
            elapsed = time.perf_counter() - t0
            logger.info("write_pandas completed in %.2fs", elapsed)
        else:
            logger.info("DataFrame is empty — table created with no rows.")

        logger.info("Table %s created successfully.", table_name)

    # -- dunder helpers --------------------------------------------------------

    def __enter__(self) -> SnowflakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"SnowflakeClient(account={self._params['account']!r}, "
            f"user={self._params['user']!r})"
        )
