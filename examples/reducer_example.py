"""Reducer tutorial — a walkthrough of every class and object in the reducer module.

Demonstrates:
  1. ModelKey — identifying models by dimension names and values
  2. ReducerConfig — defining an aggregation step
  3. apply_reducer / ReducerResult — running a single reduction
  4. DependencyGraph — querying parent/child relationships
  5. check_harmonization — verifying children sum to parent
  6. ReducerPipeline / PipelineResult — running multiple reducers together

Run:
    uv run examples/reducer_example.py
"""

import pandas as pd
import numpy as np

from ergodicts import (
    ModelKey,
    ReducerConfig,
    ReducerPipeline,
    apply_reducer,
    check_harmonization,
)

# ==========================================================================
# Sample data
# ==========================================================================
# Monthly sales across cities, SKUs, and a product class (SL1).

rng = np.random.default_rng(42)

dates = pd.date_range("2024-01-31", periods=12, freq="ME")
cities = ["NYC", "LA", "CHI"]
skus = ["A100", "A200", "B300"]
sl1s = ["ClassX", "ClassY"]

records = []
for date in dates:
    for city in cities:
        for sku in skus:
            for sl1 in sl1s:
                records.append({
                    "month_end_date": date,
                    "CITY": city,
                    "SKU": sku,
                    "SL1": sl1,
                    "QTY": rng.integers(10, 500),
                })

df = pd.DataFrame(records)
print(f"Raw data: {len(df)} rows, columns: {list(df.columns)}")
print()


# ==========================================================================
# 1. ModelKey
# ==========================================================================
# A ModelKey identifies a single time-series by its dimension names + values.
# It's frozen, hashable, and orderable.

print("=" * 60)
print("1. ModelKey")
print("=" * 60)

key = ModelKey(dimensions=("CITY", "SKU"), values=("NYC", "A100"))
print(f"  key.dimensions = {key.dimensions}")
print(f"  key.values     = {key.values}")
print(f"  key.label      = {key.label}")
print(f"  str(key)       = {str(key)}")

# Projection: drop dimensions to get the parent key
parent_key = key.project(("CITY",))
print(f"  projected to CITY: {parent_key.label}")

# Hashable — works as dict keys and in sets
key2 = ModelKey(dimensions=("CITY", "SKU"), values=("LA", "B300"))
print(f"  set of keys: {sorted({key, key2, key})}")  # deduplicates
print()


# ==========================================================================
# 2. ReducerConfig
# ==========================================================================
# Defines a single aggregation step: parent dims must be a strict subset
# of child dims.

print("=" * 60)
print("2. ReducerConfig")
print("=" * 60)

config = ReducerConfig(
    parent_dimensions=["CITY"],
    child_dimensions=["CITY", "SKU"],
)
print(f"  parent_dimensions = {config.parent_dimensions}")
print(f"  child_dimensions  = {config.child_dimensions}")
print(f"  date_col          = {config.date_col}")
print(f"  target_col        = {config.target_col}")

# You can use aliases for the column names
config_aliased = ReducerConfig(
    parent_dimensions=["CITY"],
    child_dimensions=["CITY", "SKU"],
    date_column="month_end_date",
    value_column="QTY",
)
print(f"  aliased date_col  = {config_aliased.date_col}")
print(f"  aliased target_col = {config_aliased.target_col}")
print()


# ==========================================================================
# 3. apply_reducer / ReducerResult
# ==========================================================================
# The core function: takes a DataFrame + config, returns a ReducerResult.

print("=" * 60)
print("3. apply_reducer / ReducerResult")
print("=" * 60)

result = apply_reducer(df, config)

# -- Parent data: aggregated at the CITY level
print(f"\n  Parent level: '{result.parent_level_name}' ({len(result.parent_data)} rows)")
print(f"  Columns: {list(result.parent_data.columns)}")
print(result.parent_data.head(6).to_string(index=False))

# -- Child data: aggregated at CITY x SKU, with __parent_key linking back
print(f"\n  Child level: '{result.child_level_name}' ({len(result.child_data)} rows)")
print(f"  Columns: {list(result.child_data.columns)}")
print(result.child_data.head(6).to_string(index=False))

# -- The config that produced this result is always available
print(f"\n  result.config.parent_dimensions = {result.config.parent_dimensions}")

# -- get_series: extract the time-series for a single ModelKey
#    Returns model_name + date + value columns.
nyc = ModelKey(dimensions=("CITY",), values=("NYC",))
nyc_a100 = ModelKey(dimensions=("CITY", "SKU"), values=("NYC", "A100"))

print(f"\n  get_series for parent {nyc.label}:")
print(result.get_series(nyc).to_string(index=False))

print(f"\n  get_series for child {nyc_a100.label}:")
print(result.get_series(nyc_a100).to_string(index=False))

# -- Wildcards: use "*" to match any value for a dimension
#    Get ALL SKU children for NYC in one call
pattern = ModelKey(dimensions=("CITY", "SKU"), values=("NYC", "*"))
print(f"\n  get_series with wildcard {pattern.label}:")
print(result.get_series(pattern).head(6).to_string(index=False))

# Get all children across all cities for SKU=A100
pattern2 = ModelKey(dimensions=("CITY", "SKU"), values=("*", "A100"))
print(f"\n  get_series with wildcard {pattern2.label}:")
print(result.get_series(pattern2).head(6).to_string(index=False))

# KeyError for unknown keys
try:
    result.get_series(ModelKey(dimensions=("CITY",), values=("MARS",)))
except KeyError as e:
    print(f"\n  Missing key raises KeyError: {e}")
print()


# ==========================================================================
# 4. DependencyGraph
# ==========================================================================
# Tracks parent → child relationships. Every ReducerResult has one.

print("=" * 60)
print("4. DependencyGraph")
print("=" * 60)

graph = result.dependencies
print(f"\n  {graph}")

# -- All parents and children
print(f"  Parents: {sorted(graph.all_parents, key=str)}")
print(f"  Children: {len(graph.all_children)} total")

# -- Query children of a specific parent
nyc = ModelKey(dimensions=("CITY",), values=("NYC",))
nyc_children = graph.children_of(nyc)
print(f"\n  Children of {nyc.label}:")
for child in sorted(nyc_children, key=str):
    print(f"    {child.label}")

# -- Query parents of a specific child
child_key = ModelKey(dimensions=("CITY", "SKU"), values=("NYC", "A100"))
parents = graph.parents_of(child_key)
print(f"\n  Parents of {child_key.label}: {[p.label for p in parents]}")

# -- Orphan detection
orphan = ModelKey(dimensions=("CITY", "SKU"), values=("MARS", "Z999"))
all_keys = graph.all_parents | graph.all_children | {orphan}
orphans = graph.orphans(all_keys)
print(f"\n  Orphans (not in any relationship): {[o.label for o in orphans]}")
print()


# ==========================================================================
# 5. check_harmonization
# ==========================================================================
# Verifies: for every parent + date, sum(children) == parent value.

print("=" * 60)
print("5. check_harmonization")
print("=" * 60)

errors = check_harmonization(result)
print(f"\n  Errors: {len(errors)} (should be 0)")

# Simulate a mismatch by corrupting a parent value
corrupted_result = apply_reducer(df, config)
corrupted_result.parent_data.iloc[0, corrupted_result.parent_data.columns.get_loc("value")] += 999.0

errors = check_harmonization(corrupted_result)
print(f"  After corruption: {len(errors)} error(s)")
if errors:
    err = errors[0]
    print(f"    parent      = {err.parent_key.label}")
    print(f"    date        = {err.date}")
    print(f"    parent_val  = {err.parent_value}")
    print(f"    children_sum = {err.children_sum}")
    print(f"    diff        = {err.diff:+.1f}")

# Tolerance control
errors_strict = check_harmonization(corrupted_result, atol=1e-8)
errors_loose = check_harmonization(corrupted_result, atol=1000.0)
print(f"\n  strict (atol=1e-8): {len(errors_strict)} error(s)")
print(f"  loose  (atol=1000): {len(errors_loose)} error(s)")
print()


# ==========================================================================
# 6. ReducerPipeline / PipelineResult
# ==========================================================================
# Run multiple reducers and merge everything together.

print("=" * 60)
print("6. ReducerPipeline / PipelineResult")
print("=" * 60)

pipeline = ReducerPipeline([
    ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SKU"]),
    ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SL1"]),
])
pipeline_result = pipeline.run(df)

# -- Datasets: one DataFrame per dimension level
print(f"\n  Datasets: {list(pipeline_result.datasets.keys())}")
for name, ds in pipeline_result.datasets.items():
    print(f"    {name}: {len(ds)} rows")

# -- The CITY level is shared by both reducers but is deduplicated
print(f"\n  CITY rows: {len(pipeline_result.datasets['CITY'])} (not doubled)")

# -- Merged dependency graph: each CITY parent has children from BOTH reducers
merged_graph = pipeline_result.dependencies
print(f"\n  Merged graph: {merged_graph}")
for parent in sorted(merged_graph.all_parents, key=str):
    children = merged_graph.children_of(parent)
    sku_count = sum(1 for c in children if "SKU" in c.dimensions)
    sl1_count = sum(1 for c in children if "SL1" in c.dimensions)
    print(f"    {parent.label}: {sku_count} SKU children + {sl1_count} SL1 children")

# -- Individual results are still accessible
print(f"\n  Individual results: {len(pipeline_result.results)}")
for i, r in enumerate(pipeline_result.results):
    print(f"    [{i}] {r.child_level_name} → {r.parent_level_name}")

# -- get_series: works across all datasets in the pipeline
nyc = ModelKey(dimensions=("CITY",), values=("NYC",))
nyc_classx = ModelKey(dimensions=("CITY", "SL1"), values=("NYC", "ClassX"))

print(f"\n  get_series for {nyc.label} (parent from shared CITY level):")
print(pipeline_result.get_series(nyc).to_string(index=False))

print(f"\n  get_series for {nyc_classx.label} (child from CITY@SL1 level):")
print(pipeline_result.get_series(nyc_classx).head(4).to_string(index=False))

# -- Pipeline-wide harmonization
all_errors = pipeline_result.check_harmonization()
print(f"\n  Pipeline harmonization errors: {len(all_errors)}")
print()


# ==========================================================================
# 7. Visualization (optional — requires graphviz)
# ==========================================================================

print("=" * 60)
print("7. Visualization")
print("=" * 60)

try:
    dot = merged_graph.show()
    dot.render("/tmp/ergodicts_reducer_graph", format="png", cleanup=True)
    print("\n  Graph saved to /tmp/ergodicts_reducer_graph.png")
    print("  In a Jupyter notebook, just call graph.show() to display inline.")
except ImportError:
    print("\n  graphviz not installed — skip with: pip install ergodicts[viz]")
print()
