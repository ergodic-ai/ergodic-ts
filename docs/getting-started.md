# Getting Started

## Installation

```bash
pip install ergodicts
```

Or for development:

```bash
git clone <repo-url>
cd ergodicts
uv sync
```

## Snowflake setup

1. Copy the example environment file:

    ```bash
    cp .env.example .env
    ```

2. Fill in your Snowflake credentials:

    ```env
    SNOWFLAKE_ACCOUNT=
    SNOWFLAKE_USER=
    SNOWFLAKE_PASSWORD=
    SNOWFLAKE_ROLE=
    SNOWFLAKE_WAREHOUSE=
    SNOWFLAKE_DATABASE=
    SNOWFLAKE_SCHEMA=
    ```

3. Create a client:

    ```python
    import ergodicts

    client = ergodicts.snowflake_client()
    ```

    You can also pass credentials explicitly:

    ```python
    client = ergodicts.snowflake_client(
        account="my-account",
        user="my-user",
        password="my-password",
        role="MY_ROLE",
        warehouse="MY_WH",
        database="MY_DB",
        schema="MY_SCHEMA",
    )
    ```

## Querying data

```python
df = client.get_data("SELECT * FROM my_table WHERE year = 2025")
```

## Writing data

```python
import pandas as pd

df = pd.DataFrame({"QUARTER": ["Q1", "Q2"], "REVENUE": [100, 200]})
client.create_table_from_dataframe(df, "MY_TABLE")

# Overwrite an existing table
client.create_table_from_dataframe(df, "MY_TABLE", overwrite=True)
```

## Fiscal calendar utilities

```python
from ergodicts import date_to_quarter_string, quarter_string_to_date

# QE-JUL (default)
date_to_quarter_string("2025-08-15")        # "2026Q1"
quarter_string_to_date("FY2026.Q1")         # Timestamp("2025-10-31")

# Other fiscal year ends
date_to_quarter_string("2026-03-15", fy_end_month=12)  # "2026Q1"
```

## Reducing dimensions

```python
from ergodicts import ReducerConfig, ReducerPipeline

pipeline = ReducerPipeline([
    ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SKU"]),
    ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SL1"]),
])
result = pipeline.run(df)

# Check consistency
errors = result.check_harmonization()
```

## Running tests

```bash
uv run pytest tests/ -v
```

## Building docs locally

```bash
uv run --group docs mkdocs serve
```
