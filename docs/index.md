# Ergodicts

**Hierarchical Bayesian time-series forecasting with composable structural components.**

Ergodicts builds interpretable forecasting models that respect hierarchical
constraints — the parts always add up to the whole.  Under the hood it uses
NumPyro for Bayesian inference and JAX for hardware-accelerated computation.

## Key features

- **Hierarchical reconciliation** — bottom-up, soft, or independent forecasts
  that are coherent across aggregation levels
- **Composable components** — mix and match trend (LLT, damped LLT, OU),
  seasonality (Fourier, monthly, multiplicative), and regression components
  per node
- **Causal DAG** — link external predictors (GDP, weather, etc.) to any node
  with configurable lags
- **Backtesting** — single-split and expanding-window evaluation with 13
  built-in metrics
- **Dashboard** — FastAPI + Plotly interactive UI for exploring backtest runs
- **Reducer** — aggregate raw data across arbitrary dimension combinations
  with automatic harmonization checks
- **Full uncertainty** — every forecast is a posterior predictive distribution,
  not a point estimate

## Architecture

```
Raw Data  →  Reducer  →  Hierarchy + CausalDAG  →  Forecaster  →  Forecasts
                              ↑                         ↑
                         NodeConfig              NumPyro (NUTS/SVI)
                       (components,
                        aggregator)

Forecasts  →  Backtester  →  BacktestSummary  →  Dashboard Server
```

## Modules

| Module | Description |
|--------|-------------|
| [Forecaster](api/forecaster.md) | Hierarchical Bayesian STS model with NUTS/SVI inference |
| [Components](api/components.md) | Trend, seasonality, regression components and aggregators |
| [Causal DAG](api/causal_dag.md) | Directed graph of predictive relationships |
| [Backtester](api/backtester.md) | Time-series cross-validation with 13 metrics |
| [Dashboard Server](api/server.md) | FastAPI backend + interactive frontend |
| [Reducer](api/reducer.md) | Hierarchical aggregation across dimension combinations |
| [Snowflake Client](api/snowflake_client.md) | Query Snowflake, write DataFrames to tables |
| [Utils](api/utils.md) | Fiscal calendar helpers |

## Quick start

```python
from ergodicts import (
    CausalDAG, DependencyGraph, ExternalNode,
    HierarchicalForecaster, ModelKey, NodeConfig,
)
from ergodicts.components import FourierSeasonality, LocalLinearTrend

# 1. Define hierarchy
total = ModelKey(("Level",), ("Total",))
child_a = ModelKey(("Level",), ("A",))
child_b = ModelKey(("Level",), ("B",))

hierarchy = DependencyGraph()
hierarchy.add(total, child_a)
hierarchy.add(total, child_b)

# 2. Configure components
configs = {k: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality())) for k in [total, child_a, child_b]}

# 3. Fit and forecast
model = HierarchicalForecaster(hierarchy, CausalDAG(), configs)
model.fit(y_data)
forecasts = model.forecast(horizon=12)
```

See the [Getting Started](getting-started.md) guide for installation and
the [End-to-End tutorial](tutorials/end-to-end.md) for a complete walkthrough.

## Installation

```bash
pip install ergodicts

# With forecasting dependencies
pip install ergodicts[forecast]

# With dashboard
pip install ergodicts[dashboard]
```
