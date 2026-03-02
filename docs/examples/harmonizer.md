# Harmonizer Examples

This page walks through two complete examples of forecast harmonization.
The first uses **hierarchical (additive) constraints** only — the simplest
and most common case.  The second adds **price and elasticity constraints**
for revenue decomposition.

Both examples use synthetic data so you can run them without any external
datasets.

---

## Example 1 — Hierarchical harmonization

### The problem

Imagine you forecast three regions independently: East, West, and a
National total.  Each model was fit separately, so the national forecast
doesn't exactly equal East + West.  We want to adjust all three forecasts
so they become **coherent** — the total always equals the sum of the
parts — while staying as close to the original beliefs as possible.

```
        National
        /      \
     East     West
```

### Step 1: Define the hierarchy

Every node in the hierarchy is identified by a
[`ModelKey`][ergodicts.reducer.ModelKey] — a frozen, hashable pair of
dimension names and values.  The hierarchy itself is a
[`DependencyGraph`][ergodicts.reducer.DependencyGraph] that maps parents
to children.

```python
import numpy as np
from ergodicts import ModelKey, DependencyGraph

# Create ModelKeys — think of these as unique IDs for each time-series.
# The first tuple is the *dimension names*, the second is the *values*.
national = ModelKey(("REGION",), ("NATIONAL",))
east     = ModelKey(("REGION",), ("EAST",))
west     = ModelKey(("REGION",), ("WEST",))

# Build the hierarchy: National is the parent of East and West.
hierarchy = DependencyGraph()
hierarchy.add(national, east)
hierarchy.add(national, west)
```

### Step 2: Create beliefs

A [`ForecastBelief`][ergodicts.harmonizer.ForecastBelief] wraps whatever
you know about a series.  It accepts two input formats:

1. **Samples** — a `(num_samples, T)` array of posterior draws (e.g. from
   your own NumPyro model)
2. **Parametric** — a distribution name (`"normal"`, `"studentt"`,
   `"lognormal"`, `"laplace"`) plus its parameters as `(T,)` arrays

Here we use parametric beliefs to keep things simple.  We're forecasting
**3 time steps** ahead, and we intentionally make the beliefs
**inconsistent**: National expects ~200, but East (~120) + West (~90) = 210.

```python
from ergodicts import ForecastBelief

# National forecast: mean [200, 210, 220], std 10 at each step.
# This is what our "national model" predicts.
beliefs = {
    national: ForecastBelief(
        distribution="normal",
        params={
            "loc":   np.array([200.0, 210.0, 220.0]),
            "scale": np.array([10.0,  10.0,  10.0]),
        },
    ),
    # East forecast: mean [120, 125, 130], std 5.
    # Our "east model" is more precise (lower std), so it should move less.
    east: ForecastBelief(
        distribution="normal",
        params={
            "loc":   np.array([120.0, 125.0, 130.0]),
            "scale": np.array([5.0,   5.0,   5.0]),
        },
    ),
    # West forecast: mean [90, 95, 100], std 8.
    west: ForecastBelief(
        distribution="normal",
        params={
            "loc":   np.array([90.0, 95.0, 100.0]),
            "scale": np.array([8.0,  8.0,  8.0]),
        },
    ),
}
```

Notice the inconsistency: at t=0, National=200 but East+West=210.
The harmonizer will split this 10-unit gap across all three series,
weighted by their uncertainty (std).

### Step 3: Harmonize (analytical)

Since we only have additive (linear) constraints and Gaussian beliefs,
we can use the **analytical** method.  This applies a MinT/WLS projection
that is exact, instantaneous, and requires no tuning:

$$
\hat{\mathbf{y}} = \mathbf{y} - \mathbf{W} \mathbf{A}^\top
(\mathbf{A} \mathbf{W} \mathbf{A}^\top)^{-1}
\mathbf{A} \mathbf{y}
$$

```python
from ergodicts import Harmonizer

# The Harmonizer auto-builds an AdditiveConstraint for each parent
# in the DependencyGraph.  No need to specify constraints manually.
harmonizer = Harmonizer(hierarchy=hierarchy)

result = harmonizer.harmonize(
    beliefs,
    method="analytical",  # closed-form — fast and exact
    num_samples=1000,     # number of posterior samples to draw & reconcile
    rng_seed=42,
)
```

### Step 4: Inspect the result

The [`HarmonizedResult`][ergodicts.harmonizer.HarmonizedResult] holds
reconciled posterior samples and provides convenience methods:

```python
# Mean forecast per series — shape (T,)
means = result.mean()
for key in [national, east, west]:
    print(f"{key.label:20s}  {means[key]}")

# Check: does National ≈ East + West?
gap = means[national] - (means[east] + means[west])
print(f"\nNational - (East + West) = {gap}")
# Should be essentially zero (< 0.01)
```

You can also get full quantiles and a tidy summary DataFrame:

```python
# Quantiles — shape (len(q), T)
q = result.quantiles([0.1, 0.5, 0.9])

# Tidy DataFrame with mean, std, q10, median, q90 per series per timestep
df = result.summary()
print(df)
```

### Step 5: Compare original vs reconciled

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, key in zip(axes, [national, east, west]):
    original = beliefs[key].params["loc"]
    reconciled = means[key]
    x = np.arange(len(original))
    ax.bar(x - 0.15, original, 0.3, label="Original", alpha=0.7)
    ax.bar(x + 0.15, reconciled, 0.3, label="Reconciled", alpha=0.7)
    ax.set_title(key.label)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Forecast")
    ax.legend()
plt.suptitle("Hierarchical Harmonization: Original vs Reconciled")
plt.tight_layout()
plt.show()
```

### The trust parameter

What if you trust your East model more than the others?  Set `trust > 1`
to make it resist adjustment:

```python
beliefs[east] = ForecastBelief(
    distribution="normal",
    params={
        "loc":   np.array([120.0, 125.0, 130.0]),
        "scale": np.array([5.0,   5.0,   5.0]),
    },
    trust=3.0,  # 3× tighter prior — East will barely move
)

result = harmonizer.harmonize(beliefs, method="analytical", num_samples=1000)
# Now most of the adjustment falls on National and West
```

---

## Example 2 — Price, quantity, and elasticity

### The problem

You're forecasting three related series for a product:

- **Revenue** (dollars) = ASP × Quantity
- **ASP** (average selling price)
- **Quantity** (units sold)

Each series was forecast independently, so `ASP × Qty ≠ Revenue`.
Additionally, you believe there's a price–demand relationship: if ASP
goes up, quantity should go down (and vice versa).  We want to impose both
the **price identity** and an **elasticity prior** during reconciliation.

### Step 1: Define the series

```python
import numpy as np
from ergodicts import ModelKey, ForecastBelief

# Three related series for the same product
asp = ModelKey(("METRIC",), ("ASP",))
qty = ModelKey(("METRIC",), ("QTY",))
rev = ModelKey(("METRIC",), ("REVENUE",))

# Forecast horizon: 6 months
T = 6

# ASP forecast: ~$50, slight upward trend
beliefs = {
    asp: ForecastBelief(
        distribution="normal",
        params={
            "loc":   np.array([50.0, 51.0, 52.0, 53.0, 54.0, 55.0]),
            "scale": np.array([2.0] * T),
        },
    ),
    # Quantity forecast: ~1000 units, slight downward trend
    qty: ForecastBelief(
        distribution="normal",
        params={
            "loc":   np.array([1000.0, 990.0, 980.0, 970.0, 960.0, 950.0]),
            "scale": np.array([50.0] * T),
        },
    ),
    # Revenue forecast: ~$52k (intentionally inconsistent with ASP × Qty)
    # ASP × Qty ≈ $50k at t=0, but Revenue belief says $52k → $2k gap
    rev: ForecastBelief(
        distribution="normal",
        params={
            "loc":   np.array([52000.0, 52000.0, 52000.0, 52000.0, 52000.0, 52000.0]),
            "scale": np.array([3000.0] * T),
        },
    ),
}

# Check the initial inconsistency:
print("Initial ASP × Qty:", beliefs[asp].params["loc"] * beliefs[qty].params["loc"])
print("Initial Revenue:   ", beliefs[rev].params["loc"])
```

### Step 2: Define constraints

We need two constraints:

1. **PriceConstraint** — enforces ASP × Qty ≈ Revenue
2. **ElasticityConstraint** — enforces the log-linear price–demand
   relationship with a prior on the elasticity coefficient

```python
from ergodicts import (
    Harmonizer,
    PriceConstraint,
    ElasticityConstraint,
)

# PriceConstraint: the accounting identity ASP × Qty = Revenue.
# This is a "hard" business rule — we want lambda_price to be high.
price_constraint = PriceConstraint(
    asp_key=asp,
    qty_key=qty,
    dollar_key=rev,
)

# ElasticityConstraint: log(Qty) ~ ε × log(ASP) + intercept.
#
# The elasticity_prior is (mean, std):
#   mean = -1.0 → unit elastic (10% price increase → 10% qty decrease)
#   std  =  0.5 → moderate uncertainty about the exact elasticity
#
# In MCMC mode, the elasticity is actually *sampled* as a latent variable,
# so the posterior will tell you what elasticity is consistent with the data.
elasticity_constraint = ElasticityConstraint(
    asp_key=asp,
    qty_key=qty,
    elasticity_prior=(-1.0, 0.5),
)
```

### Step 3: Build the harmonizer

Since these constraints are **nonlinear** (multiplication, logarithms),
we must use the MCMC method.  The analytical method would automatically
fall back to MCMC with a warning.

```python
harmonizer = Harmonizer(
    constraints=[price_constraint, elasticity_constraint],
    # No hierarchy= needed here — these aren't hierarchical series
)
```

### Step 4: Harmonize with MCMC

The `lambda_*` parameters control how strongly each constraint is
enforced.  Think of them as penalty weights:

- **Higher lambda** → constraint is satisfied more tightly, but beliefs
  are distorted more
- **Lower lambda** → beliefs are preserved better, but constraint
  residuals are larger

```python
result = harmonizer.harmonize(
    beliefs,
    method="mcmc",
    num_samples=500,
    lambda_price=10.0,      # strong: revenue = asp × qty is an identity
    lambda_elasticity=0.5,  # moderate: elasticity is a soft prior
    mcmc_kwargs={
        "num_warmup": 300,
        "num_chains": 1,
    },
    rng_seed=42,
)
```

### Step 5: Check the results

```python
means = result.mean()

print("Reconciled ASP:", means[asp])
print("Reconciled Qty:", means[qty])
print("Reconciled Rev:", means[rev])
print()
print("ASP × Qty:     ", means[asp] * means[qty])
print("Revenue:       ", means[rev])
print("Gap:           ", means[asp] * means[qty] - means[rev])
```

The gap should be much smaller than the original $5k inconsistency.

### Step 6: Inspect constraint violations

The `constraint_violations` dict shows per-constraint, per-timestep
residuals after harmonization:

```python
for name, violation in result.constraint_violations.items():
    print(f"{name}: mean |residual| = {violation.mean():.2f}")
```

### Step 7: Visualise

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
labels = {asp: "ASP ($)", qty: "Quantity", rev: "Revenue ($)"}

for ax, key in zip(axes, [asp, qty, rev]):
    original = beliefs[key].params["loc"]
    reconciled = means[key]
    samples = result.samples[key]
    x = np.arange(T)

    # Posterior uncertainty band
    q10 = np.quantile(samples, 0.1, axis=0)
    q90 = np.quantile(samples, 0.9, axis=0)
    ax.fill_between(x, q10, q90, alpha=0.2, color="C1", label="80% CI")

    ax.plot(x, original, "k--", label="Original belief")
    ax.plot(x, reconciled, "C1-", linewidth=2, label="Reconciled")
    ax.set_title(labels[key])
    ax.set_xlabel("Month")
    ax.legend(fontsize=8)

plt.suptitle("Price + Elasticity Harmonization")
plt.tight_layout()
plt.show()

# Verify the price identity visually
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(means[asp] * means[qty], "C0-o", label="ASP × Qty (reconciled)")
ax.plot(means[rev], "C1-s", label="Revenue (reconciled)")
ax.plot(beliefs[rev].params["loc"], "k--", alpha=0.5, label="Revenue (original)")
ax.legend()
ax.set_xlabel("Month")
ax.set_ylabel("Dollars")
ax.set_title("Price Identity: ASP × Qty vs Revenue")
plt.tight_layout()
plt.show()
```

---

## Combining both: hierarchy + price

In practice, you often have **both** a hierarchy and price constraints.
For example, revenue at the national level is the sum of regional
revenues, and each region has its own ASP × Qty = Revenue identity.

```python
from ergodicts import (
    Harmonizer, ForecastBelief, PriceConstraint,
    ModelKey, DependencyGraph,
)
import numpy as np

# Regional revenue hierarchy
rev_total = ModelKey(("LEVEL",), ("TOTAL_REV",))
rev_east  = ModelKey(("LEVEL",), ("EAST_REV",))
rev_west  = ModelKey(("LEVEL",), ("WEST_REV",))

hierarchy = DependencyGraph()
hierarchy.add(rev_total, rev_east)
hierarchy.add(rev_total, rev_west)

# Per-region ASP and Qty
asp_east = ModelKey(("LEVEL",), ("EAST_ASP",))
qty_east = ModelKey(("LEVEL",), ("EAST_QTY",))
asp_west = ModelKey(("LEVEL",), ("WEST_ASP",))
qty_west = ModelKey(("LEVEL",), ("WEST_QTY",))

T = 4
beliefs = {
    rev_total: ForecastBelief(distribution="normal",
        params={"loc": np.array([100000.0]*T), "scale": np.array([5000.0]*T)}),
    rev_east:  ForecastBelief(distribution="normal",
        params={"loc": np.array([60000.0]*T), "scale": np.array([3000.0]*T)}),
    rev_west:  ForecastBelief(distribution="normal",
        params={"loc": np.array([45000.0]*T), "scale": np.array([3000.0]*T)}),
    asp_east:  ForecastBelief(distribution="normal",
        params={"loc": np.array([50.0]*T), "scale": np.array([2.0]*T)}),
    qty_east:  ForecastBelief(distribution="normal",
        params={"loc": np.array([1100.0]*T), "scale": np.array([50.0]*T)}),
    asp_west:  ForecastBelief(distribution="normal",
        params={"loc": np.array([45.0]*T), "scale": np.array([2.0]*T)}),
    qty_west:  ForecastBelief(distribution="normal",
        params={"loc": np.array([900.0]*T), "scale": np.array([50.0]*T)}),
}

harmonizer = Harmonizer(
    hierarchy=hierarchy,  # auto-generates: rev_total = rev_east + rev_west
    constraints=[
        PriceConstraint(asp_key=asp_east, qty_key=qty_east, dollar_key=rev_east),
        PriceConstraint(asp_key=asp_west, qty_key=qty_west, dollar_key=rev_west),
    ],
)

result = harmonizer.harmonize(
    beliefs,
    method="mcmc",
    lambda_hierarchy=2.0,
    lambda_price=1.5,
    num_samples=500,
    mcmc_kwargs={"num_warmup": 300},
    rng_seed=42,
)

means = result.mean()
print(f"Total Rev:         {means[rev_total][0]:.0f}")
print(f"East + West Rev:   {means[rev_east][0] + means[rev_west][0]:.0f}")
print(f"East ASP × Qty:    {means[asp_east][0] * means[qty_east][0]:.0f}")
print(f"East Rev:          {means[rev_east][0]:.0f}")
```

---

## Run the example

```bash
uv run python examples/harmonizer_example.py
```
