# Create Table

This example generates random quarterly data and writes it to a Snowflake table.

## Generate sample data

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

quarters = [f"Q{q}-{y}" for y in range(2022, 2026) for q in range(1, 5)]
df = pd.DataFrame({
    "QUARTER": quarters,
    "REVENUE": rng.integers(100_000, 1_000_000, size=len(quarters)),
    "UNITS": rng.integers(50, 500, size=len(quarters)),
    "GROWTH_PCT": np.round(rng.uniform(-0.1, 0.3, size=len(quarters)), 4),
})
```

## Write to Snowflake

```python
import ergodicts

client = ergodicts.snowflake_client()
client.create_table_from_dataframe(df, "ERG_TEST_TABLE", overwrite=True)
```

## Overwrite behavior

By default, `create_table_from_dataframe` raises `TableExistsError` if the table already exists:

```python
# This will raise TableExistsError if ERG_TEST_TABLE exists
client.create_table_from_dataframe(df, "ERG_TEST_TABLE")

# Pass overwrite=True to drop and recreate
client.create_table_from_dataframe(df, "ERG_TEST_TABLE", overwrite=True)
```

## Column type mapping

Column types are inferred automatically from pandas dtypes:

| pandas dtype | Snowflake type |
|-------------|---------------|
| `int64` | `NUMBER` |
| `float64` | `FLOAT` |
| `bool` | `BOOLEAN` |
| `datetime64` | `TIMESTAMP_NTZ` |
| `object` (str) | `VARCHAR` |

## Run it

```bash
uv run examples/create_table.py
```
