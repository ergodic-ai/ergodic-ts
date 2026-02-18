"""Example: create a Snowflake table from a DataFrame.

Creates ERG_TEST_TABLE with random sample data.  Uses overwrite=True so it
can be re-run safely.

Prerequisites:
    1. Copy .env.example to .env and fill in your Snowflake credentials.
    2. uv sync

Run:
    uv run examples/create_table.py
"""

import numpy as np
import pandas as pd

import ergodicts

# -- generate sample data ------------------------------------------------------

rng = np.random.default_rng(42)

quarters = [f"Q{q}-{y}" for y in range(2022, 2026) for q in range(1, 5)]
df = pd.DataFrame(
    {
        "QUARTER": quarters,
        "REVENUE": rng.integers(100_000, 1_000_000, size=len(quarters)),
        "UNITS": rng.integers(50, 500, size=len(quarters)),
        "GROWTH_PCT": np.round(rng.uniform(-0.1, 0.3, size=len(quarters)), 4),
    }
)

print("Sample data:")
print(df)
print()

# -- write to Snowflake --------------------------------------------------------

client = ergodicts.snowflake_client()
client.create_table_from_dataframe(df, "ERG_TEST_TABLE", overwrite=True)
print("ERG_TEST_TABLE created successfully.")

# -- query the table -----------------------------------------------------------

df = client.get_data("SELECT * FROM ERG_TEST_TABLE")
print("ERG_TEST_TABLE data:")
print(df)
