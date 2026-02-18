"""Backtester proof of concept — two-level hierarchy.

Hierarchy::

    Cisco (Total)
      ├── ABU
      ├── ATSBU
      ├── CABU
      ├── CNGBU
      └── CPBU

Runs an expanding-window backtest with 3 folds (test_size=6 months)
and prints the summary metrics table.

Run::

    uv run --extra forecast python examples/backtester_poc.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodicts import (
    Backtester,
    CausalDAG,
    DependencyGraph,
    ExternalNode,
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
# 2.  Define ModelKeys — Cisco Total → 5 BUs
# ============================================================
BU_NAMES = ["ABU", "ATSBU", "CABU", "CNGBU", "CPBU"]
bu_keys = {
    name: ModelKey(dimensions=("BU",), values=(name,))
    for name in BU_NAMES
}
cisco_key = ModelKey(dimensions=("Company",), values=("Cisco",))

# ============================================================
# 3.  Build hierarchy:  Cisco → 5 BUs
# ============================================================
hierarchy = DependencyGraph()
for bk in bu_keys.values():
    hierarchy.add(cisco_key, bk)

print(f"Hierarchy: {hierarchy}")

# ============================================================
# 4.  Extract internal time-series
# ============================================================
y_data: dict[ModelKey, np.ndarray] = {}
for name, key in bu_keys.items():
    col = f"Cisco | net_bookings | {name}"
    y_data[key] = df[col].values.astype(np.float64)

# Cisco total = sum of the 5 BUs
y_data[cisco_key] = sum(y_data[k] for k in bu_keys.values())

print("\nInternal series:")
for k, v in y_data.items():
    print(f"  {str(k):>20s}  T={len(v)}  mean={np.mean(v):,.0f}")

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
    print(f"  {str(k):>20s}  T={len(v)}  NaN={nan_pct:.1f}%")

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
    cisco_key: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=2))),
}
for bk in bu_keys.values():
    node_configs[bk] = NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=2)))

# ============================================================
# 8.  Backtest — expanding window, 3 folds, 6-month horizon
# ============================================================
print("\n--- Running backtest (expanding, 3 folds, h=6) ---")
bt = Backtester(
    hierarchy=hierarchy,
    causal_dag=dag,
    node_configs=node_configs,
    reconciliation="bottom_up",
)

result = bt.run(
    y_data,
    x_data,
    mode="expanding",
    test_size=6,
    n_splits=3,
    num_warmup=200,
    num_samples=500,
    rng_seed=42,
)

# ============================================================
# 9.  Print results
# ============================================================
print(f"\n{result}")
print("\n--- Summary metrics (averaged across folds) ---")
print(result.summary_df.to_string(float_format="{:.2f}".format))

# Per-fold detail
for i, fold in enumerate(result.folds):
    print(f"\n--- Fold {i + 1}  (cutoff={fold.cutoff}, horizon={fold.horizon}) ---")
    for node, m in fold.metrics.items():
        line = "  ".join(f"{k}={v:.2f}" for k, v in m.items())
        print(f"  {str(node):>20s}  {line}")

# ============================================================
# 10.  Plot actuals vs predicted
# ============================================================
import matplotlib.pyplot as plt

out_path = "examples/backtester_poc_results.png"
fig = result.plot(y_data, dates=df.index, save_path=out_path)
print(f"\nPlot saved to {out_path}")
plt.show()
