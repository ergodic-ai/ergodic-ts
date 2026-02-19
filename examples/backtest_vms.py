"""Backtest — HE category by VMS segment × Platform with macro predictors.

Hierarchy (3 levels, built via ReducerPipeline)::

    HE (Category total)
      ├── Education- Public/Private
      │     ├── Education- Public/Private × Stratocaster-1
      │     └── Education- Public/Private × Stratocaster-2
      ├── Financial Services
      │     ├── Financial Services × Stratocaster-1
      │     └── ...
      └── ...

External predictors (from ERG_MACRO in Snowflake)::

    Aggregate-level (→ HE):
        Fed Funds Rate          ──lag=1──► HE
        Semiconductor IP        ──lag=1──► HE
        Hi-Tech Capacity        ──lag=1──► HE

    Segment-level (→ all leaf platforms under each VMS segment):
        S&P 500                 ──lag=1──► Financial Services × *
        Unemployment Rate       ──lag=1──► Government × *
        Manufacturing IP        ──lag=1──► Manufacturing × *
        PCE                     ──lag=1──► Health Care × *
        C&I Loans               ──lag=1──► Professional Services × *
        Consumer Credit         ──lag=1──► Retail × *
        Tech Equipment PMI      ──lag=1──► Technical Services × *
        Nondefense CapEx Orders ──lag=1──► Energy/Utilities × *

Run::

    uv run python examples/backtest_vms.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodicts import (
    Backtester,
    CausalDAG,
    DampedLocalLinearTrend,
    ExternalNode,
    LocalLinearTrend,
    ModelKey,
    MultiplicativeMonthlySeasonality,
    MultiplicativeSeasonality,
    NodeConfig,
    PipelineResult,
    ReducerConfig,
    ReducerPipeline,
    SnowflakeClient,
)

# ============================================================
# 1.  Pull VMS × Platform data from Snowflake
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
        WHERE ERG_DEALS.END_CUSTOMER_GU_NAME NOT IN (SELECT END_CUSTOMER_GU_NAME FROM ERG_DISTIS)
        and PLATFORM_NAME like '%Stratocaster%'
        GROUP BY DATE, CATEGORY, VMS_TOP_NAME
        ORDER BY DATE, VMS_TOP_NAME"""
)

df["DATE"] = pd.to_datetime(df["DATE"])
df["AMT"] = pd.to_numeric(df["AMT"])

print(f"Loaded {len(df)} rows")
print(f"Date range: {df['DATE'].min().date()} → {df['DATE'].max().date()}")
print(f"VMS segments: {df['VMS_TOP_NAME'].nunique()}")

df = df[df["DATE"] <= "2025-12-31"]

# ============================================================
# 2.  Filter to clean, meaningful VMS segments
# ============================================================
EXCLUDE_VMS = {"Client Unknown", "Consigned Stock", "Discard"}
df = df[~df["VMS_TOP_NAME"].isin(EXCLUDE_VMS)]

# Drop VMS segments that have any missing months
dates = sorted(df["DATE"].unique())
T_total = len(dates)
vms_counts = df.groupby("VMS_TOP_NAME")["DATE"].nunique()
complete_vms = set(vms_counts[vms_counts == T_total].index)
df = df[df["VMS_TOP_NAME"].isin(complete_vms)]

vms_names = sorted(df["VMS_TOP_NAME"].unique())
print(f"\nUsing {len(vms_names)} VMS segments (leaf nodes)")

# ============================================================
# 3.  Build hierarchy using ReducerPipeline
# ============================================================
print("\n--- Building hierarchy with ReducerPipeline ---")

pipeline = ReducerPipeline(
    [
        ReducerConfig(
            parent_dimensions=["CATEGORY"],
            child_dimensions=["CATEGORY", "VMS_TOP_NAME"],
            date_column="DATE",
            value_column="AMT",
        ),
    ]
)

pipeline_result = pipeline.run(df)
hierarchy = pipeline_result.dependencies

# Check harmonization
errors = pipeline_result.check_harmonization(atol=0.01)
if errors:
    print(f"  WARNING: {len(errors)} harmonization errors found!")
    for e in errors[:5]:
        print(f"    {e}")
else:
    print("  Harmonization check passed ✓")

print(f"  Hierarchy: {hierarchy}")

# ============================================================
# 4.  Extract y_data from reducer datasets
# ============================================================
print("\n--- Extracting time-series ---")


def _extract_y_data(pipeline_result: PipelineResult) -> dict[ModelKey, np.ndarray]:
    """Convert reducer datasets to dict[ModelKey, ndarray]."""
    y_data: dict[ModelKey, np.ndarray] = {}
    for level_name, dataset in pipeline_result.datasets.items():
        for mk, group in dataset.groupby("__model_key"):
            series = group.sort_values("date")["value"].values.astype(np.float64)
            y_data[mk] = series
    return y_data


y_data = _extract_y_data(pipeline_result)

# Identify hierarchy levels from the keys
root_keys = [k for k in y_data if len(k.dimensions) == 1]  # CATEGORY level
leaf_keys = [k for k in y_data if len(k.dimensions) == 2]  # CATEGORY×VMS level

print(f"  Root nodes: {len(root_keys)}")
print(f"  Leaf nodes (VMS): {len(leaf_keys)}")

print("\nInternal series (sample):")
for k in sorted(y_data.keys(), key=str)[:5]:
    v = y_data[k]
    print(f"  {str(k):>55s}  T={len(v)}  mean={np.mean(v):>12,.0f}")
print(f"  ... ({len(y_data)} total)")

# ============================================================
# 5.  Load macro predictors from ERG_MACRO (Snowflake)
# ============================================================
print("\n--- Loading macro predictors from ERG_MACRO ---")

# Aggregate-level predictors (→ HE total)
AGG_MACRO_COLS = {
    "FRED_|_FEDFUNDS_|_FEDERAL_FUNDS_EFFECTIVE_RATE_|_NSA": "Fed Funds Rate",
    "FRED_|_IPG3344S_|_INDUSTRIAL_PRODUCTION__MANUFACTURING__DURABLE_GOODS__SEMICONDUCTOR_AND_OTHER_ELECTRONIC_COMPONENT__NAICS___3344__|_SA": "Semiconductor IP",
    "FRED_|_CAPUTLHITEK2S_|_CAPACITY_UTILIZATION__MANUFACTURING__DURABLE_GOODS__COMPUTERS__COMMUNICATIONS_EQUIPMENT__AND_SEMICONDUCTORS__NAICS___3341_3342_3344__|_SA": "Hi-Tech Capacity",
}

# Segment-level predictors (→ all leaf platforms under each VMS segment)
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

quoted_cols = ", ".join(f'"{col}"' for col in ALL_MACRO_COLS)
macro_df = client.get_data(f"SELECT DATE, {quoted_cols} FROM ERG_MACRO ORDER BY DATE")
client.close()

macro_df["DATE"] = pd.to_datetime(macro_df["DATE"])
macro_df = macro_df.set_index("DATE")

# Align macro data to the deal dates
date_index = pd.DatetimeIndex(sorted(dates))
macro_aligned = macro_df.reindex(date_index)

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
# 6.  Define external nodes and build CausalDAG
# ============================================================
external_nodes: dict[str, ExternalNode] = {}
x_data: dict[ExternalNode, np.ndarray] = {}

for col, short_name in ALL_MACRO_COLS.items():
    node = ExternalNode(name=short_name, dynamics="rw", integrated=False)
    external_nodes[short_name] = node
    series = macro_aligned[col].values.astype(np.float64)
    s = pd.Series(series).ffill().bfill().values
    x_data[node] = s

dag = CausalDAG()

# --- Aggregate-level edges: macro → HE total ---
he_key = root_keys[0]
for col, short_name in AGG_MACRO_COLS.items():
    dag.add_edge(external_nodes[short_name], he_key, lag=1)

# --- Segment-level edges: macro → ALL leaf nodes under matching VMS ---
# Each macro predictor is wired to every leaf platform under its VMS segment.
SEGMENT_EDGES: dict[str, list[str]] = {
    "S&P 500": ["Financial Services"],
    "Unemployment Rate": ["Government"],
    "Manufacturing IP": ["Manufacturing"],
    "PCE": ["Health Care"],
    "C&I Loans": ["Professional Services"],
    "Consumer Credit": ["Retail"],
    "Tech Equipment PMI": ["Technical Services"],
    "Nondefense CapEx Orders": ["Energy/Utilities"],
}

for macro_name, seg_names in SEGMENT_EDGES.items():
    for seg_name in seg_names:
        # Wire to all leaf nodes whose VMS dimension matches
        for leaf_key in leaf_keys:
            if leaf_key.values[1] == seg_name:
                dag.add_edge(external_nodes[macro_name], leaf_key, lag=1)

print(f"\nCausal DAG: {dag}")
print("\n  Aggregate edges:")
for col, short_name in AGG_MACRO_COLS.items():
    for e in dag.children_of(external_nodes[short_name]):
        print(f"    {e}")
n_seg_edges = sum(len(dag.children_of(external_nodes[m])) for m in SEGMENT_EDGES)
print(f"\n  Segment edges: {n_seg_edges} total")

# ============================================================
# 7.  Configure nodes — multiplicative seasonality everywhere
# ============================================================
#
# Leaf nodes (VMS segments) are modelled individually.
# Root (HE) is reconciled as sum-of-leaves (bottom-up).
#
# Trend choice by leaf size:
#   - Large leaves (top 25% by mean revenue): LocalLinearTrend
#   - Smaller leaves: DampedLocalLinearTrend (more regularised)

leaf_means = {k: np.mean(y_data[k]) for k in leaf_keys}
threshold = np.percentile(list(leaf_means.values()), 75)
large_leaves = {k for k, m in leaf_means.items() if m >= threshold}
print(f"\nLarge leaves (top 25%, local linear trend): {len(large_leaves)}")
print(f"Small leaves (damped trend): {len(leaf_keys) - len(large_leaves)}")

node_configs: dict[ModelKey, NodeConfig] = {}

for leaf_key in leaf_keys:
    if leaf_key in large_leaves:
        node_configs[leaf_key] = NodeConfig(
            mode="active",
            components=(
                LocalLinearTrend(),
                MultiplicativeMonthlySeasonality(period=12),
            ),
            aggregator=MultiplicativeSeasonality(),
        )
    else:
        node_configs[leaf_key] = NodeConfig(
            mode="active",
            components=(
                DampedLocalLinearTrend(),
                MultiplicativeMonthlySeasonality(period=12),
            ),
            aggregator=MultiplicativeSeasonality(),
        )

print(f"Configured {len(node_configs)} leaf nodes with multiplicative seasonality")

# ============================================================
# 8.  Backtest — single fold, 6-month horizon
# ============================================================
# Using single fold with more samples since we have many leaf nodes.
print("\n--- Running backtest (single fold, h=6) ---")
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
    test_size=24,
    n_splits=1,
    num_warmup=300,
    num_samples=500,
    rng_seed=42,
    run_name="vms_platform_macro_1",
    run_path="runs/vms_platform_macro",
    time_index=date_index.strftime("%Y-%m-%d").to_numpy(),
)

# ============================================================
# 9.  Print results
# ============================================================
print(f"\n{result}")
print("\n--- Summary metrics (averaged across folds) ---")
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 160)
print(result.summary_df.to_string(float_format="{:.2f}".format))

# ============================================================
# 10.  Plot actuals vs predicted (leaf + root nodes)
# ============================================================
import matplotlib.pyplot as plt

out_path = "examples/backtest_vms_results.png"
fig = result.plot(y_data, dates=date_index, save_path=out_path)
print(f"\nPlot saved to {out_path}")
# plt.show()
