# Causal DAG

## Two-graph design

Ergodicts uses **two independent graphs** to describe the structure of a
forecasting problem:

| Graph | Module | Describes | Example |
|---|---|---|---|
| **Hierarchy** | [`DependencyGraph`][ergodicts.reducer.DependencyGraph] | Aggregation (what sums to what) | Total → BU_A + BU_B |
| **Causal DAG** | [`CausalDAG`][ergodicts.causal_dag.CausalDAG] | Prediction (what predicts what) | GDP → BU_A revenue |

Keeping them separate is deliberate: a parent in the hierarchy is not
necessarily a predictor of its children (and vice versa).

## External nodes

An [`ExternalNode`][ergodicts.causal_dag.ExternalNode] represents a
predictor series that is **not** part of the hierarchy — e.g. GDP,
weather, commodity prices.

```python
from ergodicts import ExternalNode

gdp = ExternalNode("GDP", dynamics="rw", integrated=True)
weather = ExternalNode("Weather_LA", dynamics="stationary")
```

The `dynamics` parameter controls how the series is extended into the
forecast horizon when future values are not provided:

| Dynamics | Behaviour |
|---|---|
| `"rw"` | Random walk from last observed value |
| `"ar1"` | First-order autoregressive |
| `"ar2"` | Second-order autoregressive |
| `"stationary"` | Reverts to historical mean |
| `"known_future"` | User provides future values |

## Edge specs

Each directed edge carries temporal metadata:

```python
from ergodicts import CausalDAG, ExternalNode, ModelKey

dag = CausalDAG()
dag.add_edge(gdp, bu_a, lag=1)                         # GDP at t-1 predicts BU_A at t
dag.add_edge(bu_a, bu_b, lag=0, contemporaneous=True)   # same-timestep effect
```

- **`lag`** — how many periods back the source value is used
- **`contemporaneous`** — if `True`, the source and target interact within
  the same time step (requires `lag=0`)

## Topological ordering

The DAG must be acyclic.  `dag.topological_order()` returns nodes in an
order where every predictor comes before its targets — this is the order
the forecaster processes causal effects.

```python
dag.topological_order()
# [ExternalNode('GDP'), ModelKey(BU=A), ModelKey(BU=B)]
```

`dag.validate()` raises `ValueError` if a cycle is detected.

## Visualization

```python
dot = dag.show()   # returns graphviz.Digraph
```

External nodes render as orange ellipses; internal `ModelKey` nodes render
as green rounded boxes.  Edge labels show lag information.

## How the forecaster uses the DAG

For each **leaf node** in the hierarchy, the forecaster:

1. Queries `dag.parents_of(node)` to find incoming edges
2. Builds a predictor matrix $X \in \mathbb{R}^{T \times J}$ from the
   external series, applying lags per `EdgeSpec`
3. Adds an [`ExternalRegression`][ergodicts.components.ExternalRegression]
   component: $r_t = \sum_j \beta_j\, x_{j,t}$

You do **not** need to manually add regression components to
`NodeConfig.components` — they are injected automatically when edges exist.
