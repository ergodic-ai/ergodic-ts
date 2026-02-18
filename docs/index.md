# Ergodic TS

Tools for time-series forecasting.

## Modules

| Module | Description |
|--------|-------------|
| [Snowflake Client](api/snowflake_client.md) | Query Snowflake, write DataFrames to tables |
| [Reducer](api/reducer.md) | Aggregate hierarchical time-series across arbitrary dimension combinations |
| [Utils](api/utils.md) | Fiscal calendar helpers (quarter strings, date conversions) |

## Quick start

```python
import ergodicts

# Connect to Snowflake (reads credentials from .env)
client = ergodicts.snowflake_client()

# Query data
df = client.get_data("SELECT * FROM my_table")

# Reduce dimensions
from ergodicts import ReducerConfig, apply_reducer

config = ReducerConfig(
    parent_dimensions=["CITY"],
    child_dimensions=["CITY", "SKU"],
)
result = apply_reducer(df, config)
```

See the [Getting Started](getting-started.md) guide for full setup instructions.
