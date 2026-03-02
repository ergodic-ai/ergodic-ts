# Backtesting

This tutorial shows how to evaluate a hierarchical forecaster on historical
data using the [`Backtester`][ergodicts.backtester.Backtester].

## 1. Setup (reusing the end-to-end example)

```python
import numpy as np
from ergodicts import (
    Backtester, CausalDAG, DependencyGraph, ExternalNode,
    ModelKey, NodeConfig,
)
from ergodicts.components import FourierSeasonality, LocalLinearTrend

# Build hierarchy, y_data, x_data, dag, node_configs
# (see End-to-End tutorial for full setup)

bu_names = ["ABU", "ATSBU", "CABU", "CNGBU", "CPBU"]
bu_keys = {n: ModelKey(("BU",), (n,)) for n in bu_names}
total_key = ModelKey(("BU",), ("Total",))

hierarchy = DependencyGraph()
for bk in bu_keys.values():
    hierarchy.add(total_key, bk)

T = 60
rng = np.random.default_rng(42)
y_data = {}
for name, key in bu_keys.items():
    y_data[key] = np.linspace(100, 150, T) + 10 * np.sin(2 * np.pi * np.arange(T) / 12) + rng.normal(0, 5, T)
y_data[total_key] = sum(y_data[k] for k in bu_keys.values())

dag = CausalDAG()
node_configs = {k: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=2))) for k in list(bu_keys.values()) + [total_key]}
```

## 2. Single-split backtest

The simplest evaluation: train on all but the last `test_size` observations,
forecast, and score.

```python
bt = Backtester(
    hierarchy=hierarchy,
    causal_dag=dag,
    node_configs=node_configs,
    reconciliation="bottom_up",
)

result = bt.run(
    y_data,
    mode="single",
    test_size=12,
    num_warmup=200,
    num_samples=500,
    rng_seed=42,
)

print(result.summary_df)
```

## 3. Expanding-window backtest

For a more robust evaluation, use multiple folds with an expanding training
window:

```python
result = bt.run(
    y_data,
    mode="expanding",
    test_size=6,
    n_splits=3,
    num_warmup=200,
    num_samples=500,
    rng_seed=42,
)

print(f"{len(result.folds)} folds")
print(result.summary_df)
```

The summary DataFrame averages metrics across folds.

## 4. Inspect per-fold results

```python
for i, fold in enumerate(result.folds):
    print(f"\nFold {i+1}: train=[0:{fold.cutoff}], test=[{fold.cutoff}:{fold.cutoff+fold.horizon}]")
    for node, metrics in fold.metrics.items():
        print(f"  {node}: MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.1f}%")
```

## 5. Available metrics

The backtester computes these metrics automatically:

| Metric | Description |
|--------|-------------|
| `mae` | Mean Absolute Error |
| `rmse` | Root Mean Squared Error |
| `mape` | Mean Absolute Percentage Error |
| `smape` | Symmetric MAPE (0--200 scale) |
| `rolling_mape_3` | MAPE on 3-step rolling sums |
| `rolling_mape_6` | MAPE on 6-step rolling sums |
| `mape_full` | MAPE on full-horizon sum |
| `accuracy` | Mean directional accuracy (0--1) |
| `coverage` | Empirical coverage of 90% prediction interval |
| `crps` | Continuous Ranked Probability Score |

## 6. Save and load

```python
# Save to disk
result.save("runs/my_backtest")

# Load later
from ergodicts.backtester import BacktestSummary
loaded = BacktestSummary.load("runs/my_backtest")
print(loaded.summary_df)
```

The save format is:

```
runs/my_backtest/
    meta.json          # configuration + per-fold metrics
    summary.csv        # summary DataFrame
    y_data.npz         # original time-series (for plotting)
    folds/
        fold_000.npz   # forecast + actual arrays
        params_000.json # parameter summaries
        diag_000.json   # convergence diagnostics
        decomp_000.npz  # component decomposition
```

## 7. Plot actuals vs forecasts

```python
import matplotlib.pyplot as plt

fig = result.plot(y_data)
plt.show()
```

## 8. Launch the dashboard

For an interactive exploration of backtest results:

```bash
uv run --extra forecast python -m ergodicts.server --runs-dir runs
```

Open `http://localhost:8765` to browse runs, view charts, compare metrics,
and inspect parameter posteriors.

## 9. Reproduce a run

Every saved run includes all configuration needed to reproduce it:

```python
loaded = BacktestSummary.load("runs/my_backtest")
repro = loaded.reproduce_config()

bt2 = Backtester(
    hierarchy=repro["hierarchy"],
    causal_dag=repro["causal_dag"],
    node_configs=repro["node_configs"],
    reconciliation=repro["reconciliation"],
)
```
