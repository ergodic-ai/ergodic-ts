# Ergodicts

Hierarchical Bayesian time-series forecasting with composable structural components.

## Features

- **Hierarchical reconciliation** — bottom-up, soft, or independent forecasts coherent across aggregation levels
- **Composable components** — mix and match trend (LLT, damped LLT, OU), seasonality (Fourier, monthly, multiplicative), and regression per node
- **Causal DAG** — link external predictors to any node with configurable lags
- **Backtesting** — single-split and expanding-window evaluation with 13 built-in metrics
- **Dashboard** — FastAPI + Plotly interactive UI for exploring backtest results
- **Reducer** — aggregate raw data across arbitrary dimension combinations
- **Full uncertainty** — every forecast is a posterior predictive distribution

## Installation

```bash
pip install ergodicts

# With forecasting dependencies (JAX, NumPyro)
pip install ergodicts[forecast]

# With dashboard (FastAPI, uvicorn)
pip install ergodicts[dashboard]
```

## Quick example

```python
import numpy as np
from ergodicts import (
    CausalDAG, DependencyGraph, HierarchicalForecaster,
    ModelKey, NodeConfig,
)
from ergodicts.components import FourierSeasonality, LocalLinearTrend

# Define hierarchy: Total → A + B
total = ModelKey(("Level",), ("Total",))
a = ModelKey(("Level",), ("A",))
b = ModelKey(("Level",), ("B",))

hierarchy = DependencyGraph()
hierarchy.add(total, a)
hierarchy.add(total, b)

# Prepare data (60 months)
T = 60
y_data = {
    a: np.linspace(100, 150, T) + 10 * np.sin(2 * np.pi * np.arange(T) / 12),
    b: np.linspace(80, 120, T) + 5 * np.sin(2 * np.pi * np.arange(T) / 12),
}
y_data[total] = y_data[a] + y_data[b]

# Fit and forecast
configs = {k: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality())) for k in [total, a, b]}
model = HierarchicalForecaster(hierarchy, CausalDAG(), configs)
model.fit(y_data, num_warmup=200, num_samples=500)
forecasts = model.forecast(horizon=12)  # {key: (500, 12) array}
```

## Documentation

Full documentation is available at the project docs site. Build locally:

```bash
uv run --group docs mkdocs serve
```

## Development

```bash
git clone <repo-url>
cd ergodicts
uv sync
uv run pytest tests/ -v
```

## License

See LICENSE file.
