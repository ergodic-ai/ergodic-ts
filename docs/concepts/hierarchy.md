# Hierarchical Forecasting

## What is hierarchical forecasting?

Many real-world time-series have a natural tree structure.  Revenue for a
company is the sum of revenue from its business units, which are themselves
sums of product lines, regions, etc.  **Hierarchical forecasting** exploits
this structure to produce forecasts that are both accurate at each level and
coherent across levels — i.e. the parts always add up to the whole.

```
          Total
        /   |   \
      BU_A BU_B BU_C      ← parent level
     / \    |    / \
    p1 p2  p3  p4  p5     ← leaf level
```

## Aggregation constraints

For any parent node $p$ with children $\{c_1, \dots, c_K\}$:

$$
y_{p,t} = \sum_{k=1}^{K} y_{c_k,t} \qquad \forall\; t
$$

This constraint must hold for **both** historical data (verified by
[`check_harmonization`][ergodicts.reducer.check_harmonization]) and
forecasts.

## Reconciliation strategies

Ergodicts supports three reconciliation strategies, configured via the
`reconciliation` parameter on
[`HierarchicalForecaster`][ergodicts.forecaster.HierarchicalForecaster]:

### Bottom-up (`"bottom_up"`)

Forecast only at the leaf level; parent forecasts are deterministic sums.

$$
\hat{y}_{p,t} = \sum_{k=1}^{K} \hat{y}_{c_k,t}
$$

**Pros:** Guaranteed coherence, simple, fast.
**Cons:** Parent-level data is only used as a soft observation target, not
to inform leaf dynamics directly.

### Soft reconciliation (`"soft"`)

Like bottom-up but adds a Gaussian potential at the parent level:

$$
y_{p,t} \sim \mathcal{N}\!\left(\sum_k \hat{y}_{c_k,t},\; \sigma_{\text{recon}}^2\right)
$$

This lets the parent data gently pull the leaf forecasts towards
consistency.  The strength is controlled by
[`NodeConfig.reconciliation_sigma`][ergodicts.causal_dag.NodeConfig].

### None (`"none"`)

Each node is forecast independently.  No aggregation constraints are
imposed.  Useful for flat (non-hierarchical) datasets.

## Multi-level pipelines

Use the [`ReducerPipeline`][ergodicts.reducer.ReducerPipeline] to build
hierarchies with multiple aggregation dimensions:

```python
from ergodicts import ReducerPipeline, ReducerConfig

pipeline = ReducerPipeline([
    ReducerConfig(parent_dimensions=["REGION"], child_dimensions=["REGION", "PRODUCT"]),
    ReducerConfig(parent_dimensions=["REGION"], child_dimensions=["REGION", "CHANNEL"]),
])
result = pipeline.run(df)
```

The merged
[`DependencyGraph`][ergodicts.reducer.DependencyGraph] connects each parent
to all its children across dimension combinations.

## From reducer to forecaster

The typical pipeline is:

1. **Reduce** — `ReducerPipeline.run(df)` → `PipelineResult` with
   `dependencies` (hierarchy) and `datasets` (time-series arrays)
2. **Build DAG** — create a [`CausalDAG`][ergodicts.causal_dag.CausalDAG]
   with external predictor edges
3. **Configure** — assign
   [`NodeConfig`][ergodicts.causal_dag.NodeConfig] per node (trend,
   seasonality, aggregator)
4. **Fit** — `HierarchicalForecaster.fit(y_data, x_data)`
5. **Forecast** — `model.forecast(horizon=12)`
