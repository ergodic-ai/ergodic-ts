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

### Optional extras

```bash
# Forecasting (JAX, NumPyro, matplotlib)
pip install ergodicts[forecast]

# Dashboard (FastAPI, uvicorn)
pip install ergodicts[dashboard]

# Visualization (graphviz)
pip install ergodicts[viz]
```

## Quick start: Forecasting

```python
import numpy as np
from ergodicts import (
    CausalDAG, DependencyGraph, HierarchicalForecaster,
    ModelKey, NodeConfig,
)
from ergodicts.components import FourierSeasonality, LocalLinearTrend

# Define a simple hierarchy: Total → A + B
total = ModelKey(("Level",), ("Total",))
a = ModelKey(("Level",), ("A",))
b = ModelKey(("Level",), ("B",))

hierarchy = DependencyGraph()
hierarchy.add(total, a)
hierarchy.add(total, b)

# Prepare data
T = 60
y_data = {
    a: np.linspace(100, 150, T) + 10 * np.sin(2 * np.pi * np.arange(T) / 12),
    b: np.linspace(80, 120, T) + 5 * np.sin(2 * np.pi * np.arange(T) / 12),
}
y_data[total] = y_data[a] + y_data[b]

# Configure and fit
configs = {k: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality())) for k in [total, a, b]}
model = HierarchicalForecaster(hierarchy, CausalDAG(), configs)
model.fit(y_data, num_warmup=200, num_samples=500, rng_seed=42)

# Forecast
forecasts = model.forecast(horizon=12)
for key, samples in forecasts.items():
    print(f"{key}: median_last={np.median(samples[:, -1]):.1f}")
```

## Quick start: Backtesting

```python
from ergodicts import Backtester

bt = Backtester(hierarchy, CausalDAG(), configs)
result = bt.run(y_data, mode="expanding", test_size=6, n_splits=3)
print(result.summary_df)
```

## Quick start: Dashboard

```bash
# After running a backtest with run_path="runs/my_run"
uv run --extra forecast python -m ergodicts.server --runs-dir runs
```

Open `http://localhost:8765` to browse runs interactively.

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

## Next steps

- [End-to-End tutorial](tutorials/end-to-end.md) — complete pipeline walkthrough
- [Backtesting tutorial](tutorials/backtesting.md) — evaluate and compare models
- [Concepts: Components](concepts/components.md) — understand the math
