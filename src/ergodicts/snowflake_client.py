"""Snowflake client for ergodicts — thin wrapper around snowflake-connector-python."""

from __future__ import annotations

import os
import logging
from typing import Any

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ENV_MAP = {
    "account": "SNOWFLAKE_ACCOUNT",
    "user": "SNOWFLAKE_USER",
    "password": "SNOWFLAKE_PASSWORD",
    "role": "SNOWFLAKE_ROLE",
    "warehouse": "SNOWFLAKE_WAREHOUSE",
    "database": "SNOWFLAKE_DATABASE",
    "schema": "SNOWFLAKE_SCHEMA",
}


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

        self._conn: snowflake.connector.SnowflakeConnection | None = None
        self._connect()

    # -- connection lifecycle --------------------------------------------------

    def _connect(self) -> None:
        logger.info("Connecting to Snowflake account=%s", self._params["account"])
        self._conn = snowflake.connector.connect(**self._params)

    def refresh(self) -> None:
        """Drop the current connection and establish a new one."""
        self.close()
        self._connect()
        logger.info("Connection refreshed.")

    def close(self) -> None:
        """Close the underlying connection if open."""
        if self._conn is not None and not self._conn.is_closed():
            self._conn.close()
            self._conn = None

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
        cur = self.connection.cursor()
        try:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=columns)
        finally:
            cur.close()

    def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Execute a statement that returns no data (DDL / DML)."""
        cur = self.connection.cursor()
        try:
            cur.execute(query, params)
        finally:
            cur.close()

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
