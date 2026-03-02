# End-to-End Forecasting Pipeline

This tutorial walks through a complete forecasting pipeline: define a
hierarchy, add external predictors, configure components, fit the model,
and generate forecasts.

## 1. Define the hierarchy

We'll model a company with 5 business units (BUs) that sum to a total:

```python
import numpy as np
from ergodicts import DependencyGraph, ModelKey

# Create ModelKeys
bu_names = ["ABU", "ATSBU", "CABU", "CNGBU", "CPBU"]
bu_keys = {name: ModelKey(dimensions=("BU",), values=(name,)) for name in bu_names}
total_key = ModelKey(dimensions=("BU",), values=("Total",))

# Build the hierarchy: Total → each BU
hierarchy = DependencyGraph()
for bk in bu_keys.values():
    hierarchy.add(total_key, bk)

print(hierarchy)
# DependencyGraph(parents=1, children=5, edges=5)
```

## 2. Prepare time-series data

Each node needs a 1-D NumPy array of length $T$:

```python
# Synthetic data: 60 months per BU
T = 60
rng = np.random.default_rng(42)

y_data: dict[ModelKey, np.ndarray] = {}
for name, key in bu_keys.items():
    trend = np.linspace(100, 150, T)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(T) / 12)
    noise = rng.normal(0, 5, T)
    y_data[key] = trend + seasonal + noise

# Total = sum of BUs (must satisfy the aggregation constraint)
y_data[total_key] = sum(y_data[k] for k in bu_keys.values())
```

## 3. Define external predictors

External predictors are series that are **not** part of the hierarchy but
may improve forecasts:

```python
from ergodicts import ExternalNode, CausalDAG

# Define external nodes
gdp = ExternalNode("GDP", dynamics="ar1")
rates = ExternalNode("InterestRate", dynamics="stationary")

# Create predictor arrays
x_data = {
    gdp: np.cumsum(rng.normal(0.1, 1, T)),     # trending
    rates: rng.normal(5, 0.5, T),               # stationary
}

# Build the CausalDAG: each external → each BU (lag=1)
dag = CausalDAG()
for ext in [gdp, rates]:
    for bk in bu_keys.values():
        dag.add_edge(ext, bk, lag=1)

print(dag)
# CausalDAG(nodes=7, edges=10)
```

## 4. Configure components per node

```python
from ergodicts import NodeConfig
from ergodicts.components import LocalLinearTrend, FourierSeasonality

node_configs = {}
for key in list(bu_keys.values()) + [total_key]:
    node_configs[key] = NodeConfig(
        components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=2)),
    )
```

## 5. Fit the model

```python
from ergodicts import HierarchicalForecaster

model = HierarchicalForecaster(
    hierarchy=hierarchy,
    causal_dag=dag,
    node_configs=node_configs,
    reconciliation="bottom_up",
)

model.fit(
    y_data=y_data,
    x_data=x_data,
    num_warmup=200,
    num_samples=500,
    rng_seed=42,
)
```

!!! tip
    For faster iteration, use `inference="svi"` instead of the default
    NUTS sampler.

## 6. Generate forecasts

```python
forecasts = model.forecast(horizon=12)

for key, samples in forecasts.items():
    median = np.median(samples, axis=0)
    print(f"{key}: shape={samples.shape}, last={median[-1]:.1f}")
```

Each entry is a `(num_samples, horizon)` array — a full posterior
predictive distribution.

## 7. Decompose forecasts

```python
forecasts, decomp = model.forecast_decomposed(horizon=12)

# Inspect trend contribution for one BU
key = bu_keys["ABU"]
trend = decomp.contributions[key]["trend_local_linear_trend"]
print(f"Trend shape: {trend.shape}")  # (500, 12)
print(f"Aggregator: {decomp.aggregator_type[key]}")  # "additive"
```

## 8. Plot results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, key in zip(axes.flat, [total_key] + list(bu_keys.values())):
    # Historical
    ax.plot(y_data[key], "k-", linewidth=0.8, label="Observed")
    # Forecast
    samples = forecasts[key]
    x_fc = np.arange(T, T + 12)
    ax.plot(x_fc, np.median(samples, axis=0), "b-", label="Median")
    ax.fill_between(
        x_fc,
        np.percentile(samples, 5, axis=0),
        np.percentile(samples, 95, axis=0),
        alpha=0.3,
    )
    ax.set_title(str(key))
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
```

## Next steps

- [Backtesting tutorial](backtesting.md) — evaluate forecast accuracy on
  historical data
- [Custom components tutorial](custom-components.md) — build your own trend
  or seasonality component
