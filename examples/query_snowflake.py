"""Example: query Snowflake and return a time-series DataFrame.

Prerequisites:
    1. Copy .env.example to .env and fill in your Snowflake credentials.
    2. uv sync

Run:
    uv run examples/query_snowflake.py
"""

import ergodicts

client = ergodicts.snowflake_client()

df = client.get_data("""
    SELECT QUARTER, VMS_TOP_NAME, SUM(AMT)
    FROM ERG_DEALS
    WHERE PF_NAME = 'C9300'
    GROUP BY QUARTER, VMS_TOP_NAME
    ORDER BY QUARTER ASC
""")

print(df)
