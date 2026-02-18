"""Backtest — Stratocaster HE category by VMS segment with macro predictors.

Hierarchy::

    HE (Category total)
      ├── Education- Public/Private
      ├── Energy/Utilities
      ├── Financial Services
      ├── Government
      ├── Health Care
      ├── ...
      └── Wholesale/Distribution

External predictors (from ERG_MACRO in Snowflake)::

    Aggregate-level (→ HE):
        Fed Funds Rate          ──lag=1──► HE
        Semiconductor IP        ──lag=1──► HE
        Hi-Tech Capacity        ──lag=1──► HE

    Segment-level:
        S&P 500                 ──lag=1──► Financial Services
        Unemployment Rate       ──lag=1──► Government
        Manufacturing IP        ──lag=1──► Manufacturing
        PCE                     ──lag=1──► Health Care
        C&I Loans               ──lag=1──► Professional Services
        Consumer Credit         ──lag=1──► Retail
        Tech Equipment PMI      ──lag=1──► Technical Services
        Nondefense CapEx Orders ──lag=1──► Energy/Utilities

Pulls VMS data and macro series from Snowflake, builds a hierarchy
of clean VMS segments summing to HE, wires macro predictors into
both the aggregate and individual segments, and runs an
expanding-window backtest.

Run::

    uv run --extra forecast python examples/backtest_vms.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodicts import (
    Backtester,
    CausalDAG,
    DampedLocalLinearTrend,
    DependencyGraph,
    ExternalNode,
    LocalLinearTrend,
    ModelKey,
    MultiplicativeMonthlySeasonality,
    MultiplicativeSeasonality,
    NodeConfig,
    SnowflakeClient,
)

# ============================================================
# 1.  Pull VMS data from Snowflake
# ============================================================
print("--- Connecting to Snowflake ---")
client = SnowflakeClient()
df = client.get_data(
    """SELECT
            DATE,
            'HE' as CATEGORY,
            VMS_TOP_NAME,
            SUM(AMT) AMT
        FROM ERG_DEALS
        WHERE PLATFORM like '%Stratocaster%'
        AND ERG_DEALS.END_CUSTOMER_GU_NAME NOT IN (SELECT END_CUSTOMER_GU_NAME FROM ERG_DISTIS)
        GROUP BY CATEGORY, DATE, VMS_TOP_NAME
        ORDER BY DATE, VMS_TOP_NAME"""
)

df["DATE"] = pd.to_datetime(df["DATE"])
df["AMT"] = pd.to_numeric(df["AMT"])

# Pivot to wide format: rows=dates, columns=VMS segments
wide = df.pivot(index="DATE", columns="VMS_TOP_NAME", values="AMT").sort_index()
print(f"Loaded {len(wide)} months  ({wide.index[0].date()} → {wide.index[-1].date()})")
print(f"VMS segments: {len(wide.columns)}")

# ============================================================
# 2.  Filter to clean, meaningful VMS segments
# ============================================================
EXCLUDE = {"Client Unknown", "Consigned Stock", "Discard"}
vms_names = [c for c in wide.columns if c not in EXCLUDE]

# Drop any segment with NaN in the remaining set
vms_names = [c for c in vms_names if wide[c].notna().all()]
print(f"\nUsing {len(vms_names)} VMS segments (excluded: {EXCLUDE})")

# Trim last month if it looks incomplete — keep through Dec 2025
wide = wide.loc[:"2025-12-31"]
print(
    f"Trimmed to {len(wide)} months  ({wide.index[0].date()} → {wide.index[-1].date()})"
)

# ============================================================
# 3.  Load macro predictors from ERG_MACRO (Snowflake)
# ============================================================
print("\n--- Loading macro predictors from ERG_MACRO ---")

# Aggregate-level predictors (→ HE total)
AGG_MACRO_COLS = {
    "FRED_|_FEDFUNDS_|_FEDERAL_FUNDS_EFFECTIVE_RATE_|_NSA": "Fed Funds Rate",
    "FRED_|_IPG3344S_|_INDUSTRIAL_PRODUCTION__MANUFACTURING__DURABLE_GOODS__SEMICONDUCTOR_AND_OTHER_ELECTRONIC_COMPONENT__NAICS___3344__|_SA": "Semiconductor IP",
    "FRED_|_CAPUTLHITEK2S_|_CAPACITY_UTILIZATION__MANUFACTURING__DURABLE_GOODS__COMPUTERS__COMMUNICATIONS_EQUIPMENT__AND_SEMICONDUCTORS__NAICS___3341_3342_3344__|_SA": "Hi-Tech Capacity",
}

# Segment-level predictors (→ individual VMS segments)
SEG_MACRO_COLS = {
    "FRED_|_SP500_|_S_P_500_|_NSA": "S&P 500",
    "FRED_|_UNRATE_|_UNEMPLOYMENT_RATE_|_SA": "Unemployment Rate",
    "FRED_|_IPMAN_|_INDUSTRIAL_PRODUCTION__MANUFACTURING__NAICS__|_SA": "Manufacturing IP",
    "FRED_|_PCE_|_PERSONAL_CONSUMPTION_EXPENDITURES_|_SAAR": "PCE",
    "FRED_|_BUSLOANS_|_COMMERCIAL_AND_INDUSTRIAL_LOANS__ALL_COMMERCIAL_BANKS_|_SA": "C&I Loans",
    "FRED_|_REVOLSL_|_REVOLVING_CONSUMER_CREDIT_OWNED_AND_SECURITIZED_|_SA": "Consumer Credit",
    "IHS_|_PMI_|_US_|_TECHNOLOGY_EQUIPMENT_|_OUTPUT_NSA": "Tech Equipment PMI",
    "FRED_|_NEWORDER_|_MANUFACTURERS__NEW_ORDERS__NONDEFENSE_CAPITAL_GOODS_EXCLUDING_AIRCRAFT_|_SA": "Nondefense CapEx Orders",
}

ALL_MACRO_COLS = {**AGG_MACRO_COLS, **SEG_MACRO_COLS}

# Build a quoted column list for the SQL query
quoted_cols = ", ".join(f'"{col}"' for col in ALL_MACRO_COLS)
macro_df = client.get_data(f"SELECT DATE, {quoted_cols} FROM ERG_MACRO ORDER BY DATE")
client.close()

macro_df["DATE"] = pd.to_datetime(macro_df["DATE"])
macro_df = macro_df.set_index("DATE")

# Align macro data to the same date range as VMS
macro_aligned = macro_df.reindex(wide.index)

print("\n  Aggregate-level predictors:")
for col, short_name in AGG_MACRO_COLS.items():
    s = macro_aligned[col]
    non_null = s.notna().sum()
    last = s.dropna().index[-1].date() if non_null > 0 else "N/A"
    print(f"    {short_name:25s}  T={non_null}  last={last}")

print("\n  Segment-level predictors:")
for col, short_name in SEG_MACRO_COLS.items():
    s = macro_aligned[col]
    non_null = s.notna().sum()
    last = s.dropna().index[-1].date() if non_null > 0 else "N/A"
    print(f"    {short_name:25s}  T={non_null}  last={last}")

# ============================================================
# 4.  Define ModelKeys — HE → VMS segments
# ============================================================
he_key = ModelKey(dimensions=("CATEGORY",), values=("HE",))
vms_keys = {
    name: ModelKey(dimensions=("CATEGORY", "VMS"), values=("HE", name))
    for name in vms_names
}

# ============================================================
# 5.  Build hierarchy:  HE → all VMS segments
# ============================================================
hierarchy = DependencyGraph()
for vk in vms_keys.values():
    hierarchy.add(he_key, vk)

print(f"\nHierarchy: {hierarchy}")

# ============================================================
# 6.  Extract time-series
# ============================================================
y_data: dict[ModelKey, np.ndarray] = {}
for name, key in vms_keys.items():
    y_data[key] = wide[name].values.astype(np.float64)

# HE total = sum of all VMS segments
y_data[he_key] = sum(y_data[k] for k in vms_keys.values())

print("\nInternal series:")
for k, v in y_data.items():
    print(f"  {str(k):>45s}  T={len(v)}  mean={np.mean(v):>12,.0f}")

# ============================================================
# 7.  Define external nodes and build CausalDAG
# ============================================================
external_nodes: dict[str, ExternalNode] = {}
x_data: dict[ExternalNode, np.ndarray] = {}

for col, short_name in ALL_MACRO_COLS.items():
    node = ExternalNode(name=short_name, dynamics="rw", integrated=False)
    external_nodes[short_name] = node
    series = macro_aligned[col].values.astype(np.float64)
    # Forward-fill any NaN gaps
    s = pd.Series(series).ffill().bfill().values
    x_data[node] = s

dag = CausalDAG()

# --- Aggregate-level edges: macro → HE total ---
for col, short_name in AGG_MACRO_COLS.items():
    dag.add_edge(external_nodes[short_name], he_key, lag=1)

# --- Segment-level edges: macro → individual VMS segments ---
#
# Each macro predictor is wired to the segment(s) it most plausibly
# influences.  All edges use lag=1 (one month lead).
SEGMENT_EDGES: dict[str, list[str]] = {
    "S&P 500":                ["Financial Services"],
    "Unemployment Rate":      ["Government"],
    "Manufacturing IP":       ["Manufacturing"],
    "PCE":                    ["Health Care"],
    "C&I Loans":              ["Professional Services"],
    "Consumer Credit":        ["Retail"],
    "Tech Equipment PMI":     ["Technical Services"],
    "Nondefense CapEx Orders": ["Energy/Utilities"],
}

for macro_name, seg_names in SEGMENT_EDGES.items():
    for seg_name in seg_names:
        if seg_name in vms_keys:
            dag.add_edge(external_nodes[macro_name], vms_keys[seg_name], lag=1)

print(f"\nCausal DAG: {dag}")
print("\n  Aggregate edges:")
for col, short_name in AGG_MACRO_COLS.items():
    for e in dag.children_of(external_nodes[short_name]):
        print(f"    {e}")
print("\n  Segment edges:")
for macro_name in SEGMENT_EDGES:
    for e in dag.children_of(external_nodes[macro_name]):
        print(f"    {e}")

# ============================================================
# 8.  Configure nodes — multiplicative seasonality everywhere
# ============================================================
#
# All nodes use (trend + exogenous) * seasonal_factor via the
# MultiplicativeSeasonality aggregator.  Seasonality weights are
# softmax-normalised per-period dummies (sum = 1, mean = 1 over
# a full 12-month cycle).  Natural for revenue data where
# seasonality acts as a percentage of the base level.
#
#   - HE aggregate:  DampedLocalLinearTrend (conservative at the top)
#   - Large segments: LocalLinearTrend (flexible trend)
#   - Smaller segments: DampedLocalLinearTrend (regularised trend)

# Identify large segments (top-5 by average revenue)
segment_means = {name: np.mean(y_data[key]) for name, key in vms_keys.items()}
large_segments = set(sorted(segment_means, key=segment_means.get, reverse=True)[:5])
print(f"\nLarge segments (local linear trend): {sorted(large_segments)}")

node_configs: dict[ModelKey, NodeConfig] = {
    # HE aggregate — damped trend * multiplicative monthly seasonality
    he_key: NodeConfig(
        mode="active",
        components=(
            DampedLocalLinearTrend(),
            MultiplicativeMonthlySeasonality(period=12),
        ),
        aggregator=MultiplicativeSeasonality(),
    ),
}

for name, vk in vms_keys.items():
    if name in large_segments:
        # Large segments: local linear trend * multiplicative monthly seasonality
        node_configs[vk] = NodeConfig(
            mode="active",
            components=(
                LocalLinearTrend(),
                MultiplicativeMonthlySeasonality(period=12),
            ),
            aggregator=MultiplicativeSeasonality(),
        )
    else:
        # Smaller segments: damped trend * multiplicative monthly seasonality
        node_configs[vk] = NodeConfig(
            mode="active",
            components=(
                DampedLocalLinearTrend(),
                MultiplicativeMonthlySeasonality(period=12),
            ),
            aggregator=MultiplicativeSeasonality(),
        )

print(f"Configured {len(node_configs)} nodes with multiplicative seasonality")

# ============================================================
# 9.  Backtest — expanding window, 3 folds, 6-month horizon
# ============================================================
print("\n--- Running backtest (expanding, 3 folds, h=6) ---")
bt = Backtester(
    hierarchy=hierarchy,
    causal_dag=dag,
    node_configs=node_configs,
    reconciliation="soft",
)

result = bt.run(
    y_data,
    x_data,
    mode="expanding",
    test_size=6,
    n_splits=3,
    num_warmup=300,
    num_samples=500,
    rng_seed=42,
    run_name="vms_macro_segments",
    run_path="runs/vms_macro_segments",
)

# ============================================================
# 10.  Print results
# ============================================================
print(f"\n{result}")
print("\n--- Summary metrics (averaged across folds) ---")
pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 140)
print(result.summary_df.to_string(float_format="{:.2f}".format))

# ============================================================
# 11.  Plot actuals vs predicted
# ============================================================
import matplotlib.pyplot as plt

out_path = "examples/backtest_vms_results.png"
fig = result.plot(y_data, dates=wide.index, save_path=out_path)
print(f"\nPlot saved to {out_path}")
# plt.show()
