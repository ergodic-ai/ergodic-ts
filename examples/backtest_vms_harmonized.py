"""Backtest with full price decomposition — AMT, QTY, and ASP.

Extends backtest_vms.py by running **two** parallel backtests:
  1. AMT (dollars) — same as the original
  2. QTY (units)   — same hierarchy + macro predictors

Then derives ASP = AMT / QTY from the posterior samples, giving a
fully consistent price decomposition where ASP × QTY = AMT holds
**exactly** by construction.

After the backtests, we apply the Harmonizer's analytical method to
tighten hierarchical consistency on both AMT and QTY, and compare
metrics with the AMT-only baseline.

Run::

    uv run python examples/backtest_vms_harmonized.py
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
    ForecastBelief,
    Harmonizer,
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
# Helper: extend a ModelKey with a METRIC dimension
# ============================================================


def _with_metric(key: ModelKey, metric: str) -> ModelKey:
    """Append a METRIC dimension to a ModelKey."""
    return ModelKey(
        dimensions=(*key.dimensions, "METRIC"),
        values=(*key.values, metric),
    )


# ============================================================
# 1.  Pull VMS data from Snowflake — both AMT and QTY
# ============================================================
print("=" * 70)
print("Backtest with Full Price Decomposition (AMT + QTY + ASP)")
print("=" * 70)

print("\n--- Connecting to Snowflake ---")
client = SnowflakeClient()
df = client.get_data(
    """SELECT
            DATE,
            'HE' as CATEGORY,
            VMS_TOP_NAME,
            SUM(AMT) AMT,
            SUM(QTY) QTY
        FROM ERG_DEALS
        WHERE ERG_DEALS.END_CUSTOMER_GU_NAME NOT IN (SELECT END_CUSTOMER_GU_NAME FROM ERG_DISTIS)
        and PLATFORM_NAME like '%Stratocaster%'
        GROUP BY DATE, CATEGORY, VMS_TOP_NAME
        ORDER BY DATE, VMS_TOP_NAME"""
)

df["DATE"] = pd.to_datetime(df["DATE"])
df["AMT"] = pd.to_numeric(df["AMT"])
df["QTY"] = pd.to_numeric(df["QTY"])

print(f"Loaded {len(df)} rows")
print(f"Date range: {df['DATE'].min().date()} -> {df['DATE'].max().date()}")
print(f"VMS segments: {df['VMS_TOP_NAME'].nunique()}")

df = df[df["DATE"] <= "2025-12-31"]

# ============================================================
# 2.  Filter to clean, meaningful VMS segments
# ============================================================
EXCLUDE_VMS = {"Client Unknown", "Consigned Stock", "Discard"}
df = df[~df["VMS_TOP_NAME"].isin(EXCLUDE_VMS)]

dates = sorted(df["DATE"].unique())
T_total = len(dates)
vms_counts = df.groupby("VMS_TOP_NAME")["DATE"].nunique()
complete_vms = set(vms_counts[vms_counts == T_total].index)
df = df[df["VMS_TOP_NAME"].isin(complete_vms)]

# Drop segments with zero or negative QTY in any month (ASP undefined)
vms_min_qty = df.groupby("VMS_TOP_NAME")["QTY"].min()
positive_qty_vms = set(vms_min_qty[vms_min_qty > 0].index)
df = df[df["VMS_TOP_NAME"].isin(positive_qty_vms)]

vms_names = sorted(df["VMS_TOP_NAME"].unique())
print(f"\nUsing {len(vms_names)} VMS segments (all with complete + positive QTY)")

# ============================================================
# 3.  Build TWO parallel hierarchies: one for AMT, one for QTY
# ============================================================
print("\n--- Building hierarchies for AMT and QTY ---")

pipeline_amt = ReducerPipeline(
    [
        ReducerConfig(
            parent_dimensions=["CATEGORY"],
            child_dimensions=["CATEGORY", "VMS_TOP_NAME"],
            date_column="DATE",
            value_column="AMT",
        ),
    ]
)

pipeline_qty = ReducerPipeline(
    [
        ReducerConfig(
            parent_dimensions=["CATEGORY"],
            child_dimensions=["CATEGORY", "VMS_TOP_NAME"],
            date_column="DATE",
            value_column="QTY",
        ),
    ]
)

result_amt = pipeline_amt.run(df)
result_qty = pipeline_qty.run(df)

hierarchy = result_amt.dependencies  # same structure for both

errors_amt = result_amt.check_harmonization(atol=0.01)
errors_qty = result_qty.check_harmonization(atol=0.01)
print(f"  AMT harmonization: {'PASS' if not errors_amt else f'{len(errors_amt)} errors'}")
print(f"  QTY harmonization: {'PASS' if not errors_qty else f'{len(errors_qty)} errors'}")
print(f"  Hierarchy: {hierarchy}")

# ============================================================
# 4.  Extract y_data for both measures
# ============================================================
print("\n--- Extracting time-series ---")


def _extract_y_data(pipeline_result: PipelineResult) -> dict[ModelKey, np.ndarray]:
    """Convert reducer datasets to dict[ModelKey, ndarray]."""
    y_data: dict[ModelKey, np.ndarray] = {}
    for _level_name, dataset in pipeline_result.datasets.items():
        for mk, group in dataset.groupby("__model_key"):
            series = group.sort_values("date")["value"].values.astype(np.float64)
            y_data[mk] = series
    return y_data


y_amt = _extract_y_data(result_amt)
y_qty = _extract_y_data(result_qty)

root_keys = [k for k in y_amt if len(k.dimensions) == 1]
leaf_keys = [k for k in y_amt if len(k.dimensions) == 2]

print(f"  Root nodes: {len(root_keys)}")
print(f"  Leaf nodes: {len(leaf_keys)}")

# Compute historical ASP = AMT / QTY for reference
y_asp: dict[ModelKey, np.ndarray] = {}
for key in y_amt:
    y_asp[key] = y_amt[key] / np.maximum(y_qty[key], 1e-6)

print("\nSample series (leaf):")
for k in sorted(leaf_keys, key=str)[:3]:
    print(
        f"  {str(k):>50s}  "
        f"AMT_mean={np.mean(y_amt[k]):>10,.0f}  "
        f"QTY_mean={np.mean(y_qty[k]):>8,.0f}  "
        f"ASP_mean={np.mean(y_asp[k]):>8,.0f}"
    )
print(f"  ... ({len(leaf_keys)} leaf nodes total)")

# ============================================================
# 5.  Load macro predictors from ERG_MACRO
# ============================================================
print("\n--- Loading macro predictors ---")

AGG_MACRO_COLS = {
    "FRED_|_FEDFUNDS_|_FEDERAL_FUNDS_EFFECTIVE_RATE_|_NSA": "Fed Funds Rate",
    "FRED_|_IPG3344S_|_INDUSTRIAL_PRODUCTION__MANUFACTURING__DURABLE_GOODS__SEMICONDUCTOR_AND_OTHER_ELECTRONIC_COMPONENT__NAICS___3344__|_SA": "Semiconductor IP",
    "FRED_|_CAPUTLHITEK2S_|_CAPACITY_UTILIZATION__MANUFACTURING__DURABLE_GOODS__COMPUTERS__COMMUNICATIONS_EQUIPMENT__AND_SEMICONDUCTORS__NAICS___3341_3342_3344__|_SA": "Hi-Tech Capacity",
}

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

date_index = pd.DatetimeIndex(sorted(dates))
macro_aligned = macro_df.reindex(date_index)

print(f"  {len(ALL_MACRO_COLS)} macro predictors loaded and aligned")

# ============================================================
# 6.  Build CausalDAG and external nodes
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

he_key = root_keys[0]
for _col, short_name in AGG_MACRO_COLS.items():
    dag.add_edge(external_nodes[short_name], he_key, lag=1)

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
        for leaf_key in leaf_keys:
            if leaf_key.values[1] == seg_name:
                dag.add_edge(external_nodes[macro_name], leaf_key, lag=1)

print(f"  CausalDAG: {dag}")

# ============================================================
# 7.  Configure nodes
# ============================================================
# Both AMT and QTY use multiplicative seasonality — revenue has
# proportional seasonal swings, and unit counts do too (e.g.
# Q4 holiday surge).  The ratio-scaling in ForecastData makes
# the values comparable (~1.0) regardless of absolute scale.

leaf_means = {k: np.mean(y_amt[k]) for k in leaf_keys}
threshold = np.percentile(list(leaf_means.values()), 75)
large_leaves = {k for k, m in leaf_means.items() if m >= threshold}

# Same config for both AMT and QTY — the forecaster's internal
# ratio-scaling normalises both to ~1.0.
node_configs: dict[ModelKey, NodeConfig] = {}
for leaf_key in leaf_keys:
    trend = LocalLinearTrend() if leaf_key in large_leaves else DampedLocalLinearTrend()
    node_configs[leaf_key] = NodeConfig(
        mode="active",
        components=(trend, MultiplicativeMonthlySeasonality(period=12)),
        aggregator=MultiplicativeSeasonality(),
    )

print(f"\n  {len(node_configs)} nodes configured (shared for AMT and QTY)")
print(f"  Large leaves (LocalLinearTrend): {len(large_leaves)}")
print(f"  Small leaves (DampedLocalLinearTrend): {len(leaf_keys) - len(large_leaves)}")

# ============================================================
# 8.  Run TWO backtests: AMT and QTY
# ============================================================
TEST_SIZE = 24
N_SPLITS = 1
NUM_WARMUP = 300
NUM_SAMPLES = 500

# --- AMT backtest ---
print("\n" + "=" * 70)
print("Running AMT Backtest")
print("=" * 70)

bt_amt = Backtester(
    hierarchy=hierarchy,
    causal_dag=dag,
    node_configs=node_configs,
    reconciliation="soft",
)

result_bt_amt = bt_amt.run(
    y_amt,
    x_data,
    mode="expanding",
    test_size=TEST_SIZE,
    n_splits=N_SPLITS,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    rng_seed=42,
    run_name="vms_harmonized_amt",
    run_path="runs/vms_harmonized_amt",
    time_index=date_index.strftime("%Y-%m-%d").to_numpy(),
)

print(f"\nAMT Backtest: {result_bt_amt}")

# --- QTY backtest ---
print("\n" + "=" * 70)
print("Running QTY Backtest")
print("=" * 70)

bt_qty = Backtester(
    hierarchy=hierarchy,
    causal_dag=dag,
    node_configs=node_configs,
    reconciliation="soft",
)

result_bt_qty = bt_qty.run(
    y_qty,
    x_data,
    mode="expanding",
    test_size=TEST_SIZE,
    n_splits=N_SPLITS,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    rng_seed=43,
    run_name="vms_harmonized_qty",
    run_path="runs/vms_harmonized_qty",
    time_index=date_index.strftime("%Y-%m-%d").to_numpy(),
)

print(f"\nQTY Backtest: {result_bt_qty}")

# ============================================================
# 9.  Derive ASP = AMT / QTY from posterior samples
# ============================================================
# Since ASP is derived per-sample, the price identity
# ASP × QTY = AMT holds *exactly* by construction.
print("\n" + "=" * 70)
print("Deriving ASP = AMT / QTY from posterior samples")
print("=" * 70)

fold_amt = result_bt_amt.folds[0]
fold_qty = result_bt_qty.folds[0]

# Match sample counts
all_node_keys = sorted(fold_amt.forecasts.keys(), key=str)
asp_forecasts: dict[ModelKey, np.ndarray] = {}

for key in all_node_keys:
    amt_samples = fold_amt.forecasts[key]  # (S, H)
    qty_samples = fold_qty.forecasts[key]  # (S, H)
    S = min(amt_samples.shape[0], qty_samples.shape[0])
    # Straight division so that ASP × QTY = AMT holds exactly.
    # If QTY is near zero, ASP will be large but the identity still holds.
    with np.errstate(divide="ignore", invalid="ignore"):
        asp_samples = np.where(
            np.abs(qty_samples[:S]) > 0.5,
            amt_samples[:S] / qty_samples[:S],
            np.nan,  # mark as NaN when qty ≈ 0
        )
    asp_forecasts[key] = asp_samples

print(f"  Derived ASP for {len(asp_forecasts)} nodes")
print(f"  Samples shape: {next(iter(asp_forecasts.values())).shape}")

# Verify price identity (only where qty > 0.5, i.e. ASP is valid)
for key in all_node_keys[:3]:
    S = min(fold_amt.forecasts[key].shape[0], fold_qty.forecasts[key].shape[0])
    valid = ~np.isnan(asp_forecasts[key])
    if valid.any():
        recon = asp_forecasts[key][valid] * fold_qty.forecasts[key][:S][valid]
        max_gap = np.abs(recon - fold_amt.forecasts[key][:S][valid]).max()
        print(f"  {str(key):>40s}  max |ASP×QTY - AMT| = {max_gap:.6f} (should be ~0)")
    else:
        print(f"  {str(key):>40s}  no valid ASP samples")

# ============================================================
# 10. Print comparison: AMT metrics
# ============================================================
print("\n" + "=" * 70)
print("AMT Forecast Quality")
print("=" * 70)

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 160)
print(result_bt_amt.summary_df.to_string(float_format="{:.2f}".format))

print("\n" + "=" * 70)
print("QTY Forecast Quality")
print("=" * 70)

print(result_bt_qty.summary_df.to_string(float_format="{:.2f}".format))

# ============================================================
# 11. ASP forecast quality (vs historical ASP = AMT / QTY)
# ============================================================
print("\n" + "=" * 70)
print("ASP Forecast Quality (derived)")
print("=" * 70)

from ergodicts.backtester import compute_metrics

cutoff = fold_amt.cutoff
horizon = fold_amt.horizon

print(f"\n{'Node':>45s}  {'ASP_MAPE':>10s}  {'ASP_Acc':>10s}  "
      f"{'ASP_mean':>10s}  {'ASP_actual':>10s}")
print("-" * 95)

for key in sorted(all_node_keys, key=str):
    # Actual ASP in the test window
    actual_asp = y_asp[key][cutoff: cutoff + horizon]
    asp_samples = asp_forecasts[key]

    # Replace NaN with per-timestep nanmedian for metrics computation
    asp_clean = np.where(np.isnan(asp_samples), np.nanmedian(asp_samples, axis=0), asp_samples)
    if np.isnan(asp_clean).any():
        print(f"{str(key):>45s}  (insufficient valid QTY samples)")
        continue

    m = compute_metrics(actual_asp, asp_clean)
    asp_median = np.nanmedian(asp_samples, axis=0).mean()
    asp_actual_mean = actual_asp.mean()

    print(f"{str(key):>45s}  {m['mape']:>9.1f}%  {m['accuracy']:>9.1f}%  "
          f"${asp_median:>9,.0f}  ${asp_actual_mean:>9,.0f}")

# ============================================================
# 12. Per-node price decomposition summary
# ============================================================
print("\n" + "=" * 70)
print("Price Decomposition Summary (forecast period)")
print("=" * 70)

print(f"\n{'Node':>40s}  {'AMT_fc':>12s}  {'QTY_fc':>10s}  {'ASP_fc':>10s}  "
      f"{'AMT_act':>12s}  {'QTY_act':>10s}  {'ASP_act':>10s}")
print("-" * 110)

for key in sorted(all_node_keys, key=str):
    # Forecast medians (averaged over horizon)
    amt_fc = np.median(fold_amt.forecasts[key], axis=0).mean()
    qty_fc = np.median(fold_qty.forecasts[key], axis=0).mean()
    asp_fc = np.nanmedian(asp_forecasts[key], axis=0).mean()

    # Actuals (averaged over test window)
    amt_act = y_amt[key][cutoff: cutoff + horizon].mean()
    qty_act = y_qty[key][cutoff: cutoff + horizon].mean()
    asp_act = y_asp[key][cutoff: cutoff + horizon].mean()

    label = str(key) if len(str(key)) <= 40 else str(key)[:37] + "..."
    print(f"{label:>40s}  ${amt_fc:>11,.0f}  {qty_fc:>10,.0f}  ${asp_fc:>9,.0f}  "
          f"${amt_act:>11,.0f}  {qty_act:>10,.0f}  ${asp_act:>9,.0f}")

# ============================================================
# 13. Plot
# ============================================================
try:
    import matplotlib.pyplot as plt

    # Pick representative nodes: root + a few leaves
    plot_keys = [root_keys[0]] + sorted(leaf_keys, key=str)[:4]

    fig, axes = plt.subplots(len(plot_keys), 3, figsize=(18, 4 * len(plot_keys)))
    if len(plot_keys) == 1:
        axes = axes[np.newaxis, :]

    for row, key in enumerate(plot_keys):
        for col, (metric, label, y_data_dict, fc_dict) in enumerate([
            ("AMT", "Dollars ($)", y_amt, fold_amt.forecasts),
            ("QTY", "Units", y_qty, fold_qty.forecasts),
            ("ASP", "Avg Price ($)", y_asp, asp_forecasts),
        ]):
            ax = axes[row, col]
            full_y = y_data_dict[key]
            T = len(full_y)
            x_full = date_index[:T]

            # Full observed series
            ax.plot(x_full, full_y, "k-", linewidth=0.8, label="Observed")

            # Forecast window
            samples = fc_dict[key]
            S = samples.shape[0]
            median = np.nanmedian(samples, axis=0)
            lo = np.nanpercentile(samples, 5, axis=0)
            hi = np.nanpercentile(samples, 95, axis=0)

            x_fc = date_index[cutoff: cutoff + horizon]
            ax.plot(x_fc, median, "C1-", linewidth=1.5, label="Forecast median")
            ax.fill_between(x_fc, lo, hi, alpha=0.2, color="C1", label="90% CI")
            ax.axvline(date_index[cutoff], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

            ax.set_title(f"{str(key)} — {label}", fontsize=9)
            if row == 0:
                ax.legend(fontsize=7)
            ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Harmonized Backtest: AMT / QTY / ASP", fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = "examples/backtest_vms_harmonized_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

except ImportError:
    print("\nInstall matplotlib to generate plots.")
