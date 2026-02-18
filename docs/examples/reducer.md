# Reducer Tutorial

The reducer module aggregates time-series data across arbitrary dimension combinations. This tutorial walks through every class and object in the library.

## Sample data

We'll work with monthly sales data that has three dimension columns: `CITY`, `SKU`, and `SL1` (a product class).

```python
import pandas as pd
import numpy as np

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
# 216 rows: 12 months × 3 cities × 3 SKUs × 2 SL1s
```

Each row is a unique combination of date + city + sku + sl1 with a quantity value. The goal is to aggregate this data at different dimension levels for forecasting.

---

## ModelKey

A `ModelKey` identifies a single model (i.e. a single time-series) by its dimension names and values. It's frozen, hashable, and orderable — so it works as a dict key, in sets, and in DataFrame columns.

```python
from ergodicts import ModelKey

key = ModelKey(dimensions=("CITY", "SKU"), values=("NYC", "A100"))

key.dimensions  # ('CITY', 'SKU')
key.values      # ('NYC', 'A100')
key.label       # 'CITY=NYC, SKU=A100'
str(key)        # 'NYC@A100'
```

### Projecting to a parent level

If you have a child key at the `[CITY, SKU]` level, you can project it down to the `[CITY]` level — this is how the reducer links children to their parents:

```python
child_key = ModelKey(dimensions=("CITY", "SKU"), values=("NYC", "A100"))
parent_key = child_key.project(("CITY",))

parent_key.label  # 'CITY=NYC'
str(parent_key)   # 'NYC'
```

Projection keeps only the requested dimensions and drops the rest. It raises `KeyError` if you ask for a dimension that doesn't exist.

---

## ReducerConfig

A `ReducerConfig` defines a single aggregation step: which dimensions to keep (parent) and which to start from (child).

```python
from ergodicts import ReducerConfig

config = ReducerConfig(
    parent_dimensions=["CITY"],
    child_dimensions=["CITY", "SKU"],
)
```

The only rule is that **parent dimensions must be a strict subset of child dimensions**. This makes sense — you're aggregating away some dimensions to get to the parent level.

### Custom column names

By default the reducer expects `month_end_date` as the date column and `QTY` as the target column. Override with `date_col`/`target_col` or the aliases `date_column`/`value_column`:

```python
config = ReducerConfig(
    parent_dimensions=["CATEGORY"],
    child_dimensions=["CATEGORY", "PLATFORM"],
    date_column="date",
    value_column="AMT",
)
```

### Invalid configs are rejected immediately

```python
# This raises ValidationError — COUNTRY is not in child_dimensions
ReducerConfig(
    parent_dimensions=["COUNTRY"],
    child_dimensions=["CITY", "SKU"],
)
```

---

## apply_reducer

This is the core function. It takes a DataFrame and a config, and returns a `ReducerResult`.

```python
from ergodicts import ReducerConfig, apply_reducer

config = ReducerConfig(
    parent_dimensions=["CITY"],
    child_dimensions=["CITY", "SKU"],
)
result = apply_reducer(df, config)
```

It validates that all required columns exist in the DataFrame before doing any work.

---

## ReducerResult

The result of a single `apply_reducer` call. Contains everything you need:

### Parent data

The time-series aggregated at the parent level. Each row has a `date`, a `value` (the sum), and a `__model_key` identifying which parent model it belongs to.

```python
result.parent_data
#          date   value      __model_key
# 0  2024-01-31  1649.0  ModelKey(CITY=NYC)
# 1  2024-01-31  1322.0   ModelKey(CITY=LA)
# 2  2024-01-31  1841.0  ModelKey(CITY=CHI)
# ...
```

### Child data

The time-series at the finer child level. Has the same columns plus `__parent_key` linking each child row to its parent:

```python
result.child_data
#          date  value           __model_key       __parent_key
# 0  2024-01-31  442.0  ModelKey(CITY=NYC, SKU=A100)  ModelKey(CITY=NYC)
# 1  2024-01-31  555.0  ModelKey(CITY=NYC, SKU=A200)  ModelKey(CITY=NYC)
# ...
```

This linkage is what makes reconciliation possible — you can always trace a child back to its parent.

### Level names

```python
result.parent_level_name  # 'CITY'
result.child_level_name   # 'CITY@SKU'
```

### Config reference

The config that produced this result is always available:

```python
result.config  # the ReducerConfig you passed in
```

### Extracting a single time-series

Use `get_series` to pull out the `date` + `value` rows for a specific `ModelKey`. It searches both parent and child data:

```python
from ergodicts import ModelKey

nyc = ModelKey(dimensions=("CITY",), values=("NYC",))
result.get_series(nyc)
#         date   value
# 0 2024-01-31  1649.0
# 1 2024-02-29  1660.0
# ...

nyc_a100 = ModelKey(dimensions=("CITY", "SKU"), values=("NYC", "A100"))
result.get_series(nyc_a100)
#         date  value
# 0 2024-01-31  442.0
# 1 2024-02-29  312.0
# ...
```

The returned DataFrame has clean `date` and `value` columns only — the internal `__model_key` / `__parent_key` columns are stripped. Raises `KeyError` if the key is not found.

---

## DependencyGraph

The dependency graph tracks which child models roll up to which parent models. Every `ReducerResult` has one at `result.dependencies`.

### Querying the graph

```python
graph = result.dependencies

# All parent keys in the graph
graph.all_parents
# {ModelKey(CITY=NYC), ModelKey(CITY=LA), ModelKey(CITY=CHI)}

# All child keys
graph.all_children
# {ModelKey(CITY=NYC, SKU=A100), ModelKey(CITY=NYC, SKU=A200), ...}

# Children of a specific parent
nyc = ModelKey(dimensions=("CITY",), values=("NYC",))
graph.children_of(nyc)
# {ModelKey(CITY=NYC, SKU=A100), ModelKey(CITY=NYC, SKU=A200), ModelKey(CITY=NYC, SKU=B300)}

# Parents of a specific child
child = ModelKey(dimensions=("CITY", "SKU"), values=("NYC", "A100"))
graph.parents_of(child)
# {ModelKey(CITY=NYC)}
```

### Merging graphs

When you run multiple reducers that share a parent level, you can merge their graphs:

```python
# graph1: [CITY, SKU] → [CITY]
# graph2: [CITY, SL1] → [CITY]
merged = graph1.merge(graph2)

# Now NYC has children from both reducers
merged.children_of(nyc)
# {ModelKey(CITY=NYC, SKU=A100), ..., ModelKey(CITY=NYC, SL1=ClassX), ...}
```

`merge` returns a new graph — the originals are not modified.

### Finding orphans

If you have a set of all known model keys, `orphans` tells you which ones aren't connected to anything:

```python
all_keys = {some_key, another_key, ...}
graph.orphans(all_keys)  # keys not appearing as parent or child
```

### Visualizing the graph

If you have `graphviz` installed (`pip install ergodicts[viz]`), you can render the graph visually:

```python
dot = graph.show()  # returns a graphviz.Digraph
```

In a Jupyter notebook this displays inline. You can also save it:

```python
dot.render("my_graph", format="png")  # saves my_graph.png
```

The visualization clusters nodes by dimension level and color-codes them, with edges showing parent→child relationships.

---

## Harmonization check

The fundamental invariant of a reducer is: **for every parent and date, the sum of children must equal the parent value**. `check_harmonization` verifies this.

```python
from ergodicts import check_harmonization

errors = check_harmonization(result)
# [] — empty means everything is consistent
```

If something is off (e.g. data was modified after reduction), you get a list of `HarmonizationError` objects:

```python
# Each error tells you exactly what's wrong
error = errors[0]
error.parent_key    # which parent
error.date          # which date
error.parent_value  # the parent's value
error.children_sum  # what the children sum to
error.diff          # children_sum - parent_value
```

You can control the tolerance with `atol`:

```python
# Allow small floating-point differences
errors = check_harmonization(result, atol=0.01)
```

---

## ReducerPipeline

A pipeline runs multiple reducers on the same DataFrame and merges everything together. This is the typical entry point when you have several dimension combinations to aggregate.

```python
from ergodicts import ReducerPipeline, ReducerConfig

pipeline = ReducerPipeline([
    ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SKU"]),
    ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SL1"]),
])
result = pipeline.run(df)
```

### PipelineResult

The pipeline returns a `PipelineResult` with three things:

**Datasets** — a dict of DataFrames keyed by level name. When two reducers share a parent level (like `CITY` above), the parent data is deduplicated automatically:

```python
result.datasets.keys()
# dict_keys(['CITY', 'CITY@SKU', 'CITY@SL1'])

result.datasets["CITY"]      # 36 rows (12 months × 3 cities)
result.datasets["CITY@SKU"]  # 108 rows (12 months × 3 cities × 3 SKUs)
result.datasets["CITY@SL1"]  # 72 rows (12 months × 3 cities × 2 SL1s)
```

**Dependencies** — a single merged `DependencyGraph` spanning all reducers:

```python
graph = result.dependencies
# Each CITY parent now has 5 children (3 SKUs + 2 SL1s)
```

**Individual results** — the list of `ReducerResult` objects if you need to inspect each reducer separately:

```python
result.results[0].config.child_dimensions  # ['CITY', 'SKU']
result.results[1].config.child_dimensions  # ['CITY', 'SL1']
```

### Extracting a single time-series

`get_series` works on `PipelineResult` too — it searches across all datasets:

```python
nyc = ModelKey(dimensions=("CITY",), values=("NYC",))
result.get_series(nyc)  # finds NYC in the shared CITY dataset

# Also works for children from any reducer in the pipeline
nyc_classx = ModelKey(dimensions=("CITY", "SL1"), values=("NYC", "ClassX"))
result.get_series(nyc_classx)  # finds in the CITY@SL1 dataset
```

### Pipeline harmonization

Check all reducers at once:

```python
errors = result.check_harmonization()
# Checks [CITY, SKU] → [CITY] and [CITY, SL1] → [CITY] in one call
```

---

## Run the example

```bash
uv run examples/reducer_example.py
```
