# Query Snowflake

This example connects to Snowflake and runs a query that aggregates deal amounts by quarter and vendor.

## Setup

```python
import ergodicts

client = ergodicts.snowflake_client()
```

The client reads credentials from your `.env` file automatically.

## Run a query

```python
df = client.get_data("""
    SELECT QUARTER, VMS_TOP_NAME, SUM(AMT)
    FROM ERG_DEALS
    WHERE PF_NAME = 'C9300'
    GROUP BY QUARTER, VMS_TOP_NAME
    ORDER BY QUARTER ASC
""")

print(df)
```

`get_data` returns a pandas DataFrame with column names from the query result.

## Run it

```bash
uv run examples/query_snowflake.py
```

!!! note
    Make sure your `.env` file is configured with valid Snowflake credentials before running. See [Getting Started](../getting-started.md) for setup instructions.
