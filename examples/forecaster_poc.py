"""Hierarchical Bayesian forecaster — proof of concept.

Demonstrates end-to-end: load data → build hierarchy + CausalDAG →
fit a structural time-series model → forecast 12 months → plot.

Run::

    uv run --extra forecast python examples/forecaster_poc.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ergodicts import (
    CausalDAG,
    DependencyGraph,
    ExternalNode,
    HierarchicalForecaster,
    ModelKey,
    NodeConfig,
)
from ergodicts.components import FourierSeasonality, LocalLinearTrend

# ============================================================
# 1.  Load data
# ============================================================
DATA_PATH = "data/original_floor_plan_bu_dataset_20260128_221014.parquet"
df = pd.read_parquet(DATA_PATH)

# Trim last 2 months (very sparse) — keep through Nov 2025
df = df.loc[:"2025-11-30"]
print(f"Loaded {len(df)} months  ({df.index[0].date()} → {df.index[-1].date()})")

# ============================================================
# 2.  Define Business-Unit ModelKeys
# ============================================================
BU_NAMES = ["ABU", "ATSBU", "CABU", "CNGBU", "CPBU"]
bu_keys = {
    name: ModelKey(dimensions=("BU",), values=(name,))
    for name in BU_NAMES
}
total_key = ModelKey(dimensions=("BU",), values=("Total",))

# ============================================================
# 3.  Build hierarchy:  Total → 5 BUs
# ============================================================
hierarchy = DependencyGraph()
for bk in bu_keys.values():
    hierarchy.add(total_key, bk)

print(f"Hierarchy: {hierarchy}")

# ============================================================
# 4.  Extract internal time-series
# ============================================================
y_data: dict[ModelKey, np.ndarray] = {}
for name, key in bu_keys.items():
    col = f"Cisco | net_bookings | {name}"
    y_data[key] = df[col].values.astype(np.float64)

# Total = sum of the 5 BUs
y_data[total_key] = sum(y_data[k] for k in bu_keys.values())

print("Internal series:")
for k, v in y_data.items():
    print(f"  {k!s:>10s}  T={len(v)}  mean={np.mean(v):,.0f}")

# ============================================================
# 5.  Define external predictors
# ============================================================
ext_indpro = ExternalNode("INDPRO", dynamics="ar1")
ext_fuel = ExternalNode("ONS_Fuel_Energy", dynamics="ar1")
ext_mfg = ExternalNode("ONS_Mfg_Capital", dynamics="ar1")

x_data: dict[ExternalNode, np.ndarray] = {
    ext_indpro: df["FRED | INDPRO | Industrial Production: Total Index | SA"].values.astype(np.float64),
    ext_fuel: df["ONS UK | Fuel and Energy | index"].values.astype(np.float64),
    ext_mfg: df["ONS UK | Manufactured Capital Goods | index"].values.astype(np.float64),
}

print("\nExternal predictors:")
for k, v in x_data.items():
    nan_pct = np.isnan(v).mean() * 100
    print(f"  {k!s:>20s}  T={len(v)}  NaN={nan_pct:.1f}%")

# ============================================================
# 6.  Build CausalDAG:  externals → each BU (lag=1)
# ============================================================
dag = CausalDAG()
for ext in [ext_indpro, ext_fuel, ext_mfg]:
    for bk in bu_keys.values():
        dag.add_edge(ext, bk, lag=1)

print(f"\nCausalDAG: {dag}")

# ============================================================
# 7.  Configure nodes
# ============================================================
node_configs: dict[ModelKey, NodeConfig] = {
    total_key: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=2))),
}
for bk in bu_keys.values():
    node_configs[bk] = NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=2)))

# ============================================================
# 8.  Fit
# ============================================================
print("\n--- Fitting model ---")
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

# ============================================================
# 9.  Forecast 12 months
# ============================================================
print("\n--- Generating forecasts ---")
forecasts = model.forecast(horizon=12)

for k, v in forecasts.items():
    median = np.median(v, axis=0)
    print(f"  {k!s:>10s}  shape={v.shape}  median_last={median[-1]:,.0f}")

# ============================================================
# 10.  Plot
# ============================================================
dates = df.index
forecast_dates = pd.date_range(
    dates[-1] + pd.DateOffset(months=1), periods=12, freq="ME",
)

fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
all_keys = [total_key] + list(bu_keys.values())

for ax, key in zip(axes.flat, all_keys):
    # Historical
    ax.plot(dates, y_data[key], "k-", linewidth=0.8, label="Observed")

    # Forecast: posterior predictive quantiles
    samples = forecasts[key]
    median = np.median(samples, axis=0)
    lo = np.percentile(samples, 5, axis=0)
    hi = np.percentile(samples, 95, axis=0)

    ax.plot(forecast_dates, median, "b-", linewidth=1.5, label="Forecast (median)")
    ax.fill_between(
        forecast_dates, lo, hi, alpha=0.3, color="blue", label="90% CI",
    )
    ax.set_title(str(key), fontsize=11)
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("Hierarchical Bayesian Forecast — POC", fontsize=14)
out_path = "examples/forecaster_poc_results.png"
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.show()
