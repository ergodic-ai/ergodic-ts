# Forecast Harmonization

## Why harmonize?

Even with hierarchical reconciliation during fitting, there are situations
where forecasts need post-hoc adjustment:

- **External inputs** — a vendor provides a regional forecast that must be
  blended with your own bottom-up model
- **Price identities** — revenue = ASP × quantity must hold, but ASP, qty,
  and revenue may come from independent models
- **Cross-model consistency** — combining forecasts from multiple systems
  that were not jointly estimated

The [`Harmonizer`][ergodicts.harmonizer.Harmonizer] takes a set of
**beliefs** about individual series and adjusts them so they satisfy a
configurable set of constraints.

## Beliefs

A [`ForecastBelief`][ergodicts.harmonizer.ForecastBelief] wraps whatever
you know about a single time series — full posterior samples, a parametric
distribution (normal, Student-t, log-normal, Laplace), or simple
$\mu / \sigma$ summaries:

```python
from ergodicts import ForecastBelief
import numpy as np

# From your own model's posterior
own = ForecastBelief(samples=posterior_samples)  # (num_samples, T)

# From a vendor (mu/std only)
vendor = ForecastBelief(
    distribution="normal",
    params={"loc": np.array([100, 110, 120]),
            "scale": np.array([5, 6, 7])},
    trust=0.5,   # lower trust → wider prior → less influence
)
```

The `trust` parameter controls how tightly the harmonizer should respect
the original belief.  Higher values shrink the prior standard deviation,
giving that series more weight in the reconciliation.

## Constraints

### Additive (hierarchy)

For any parent $p$ with children $\{c_1, \dots, c_K\}$:

$$
y_{p,t} = \sum_{k=1}^{K} y_{c_k,t}
$$

[`AdditiveConstraint`][ergodicts.harmonizer.AdditiveConstraint] is
generated automatically from a
[`DependencyGraph`][ergodicts.reducer.DependencyGraph]:

```python
harmonizer = Harmonizer(hierarchy=dep_graph)
```

### Price identity

$$
\text{ASP}_t \times \text{qty}_t = \text{dollars}_t
$$

[`PriceConstraint`][ergodicts.harmonizer.PriceConstraint] penalizes
deviations from this identity via a quadratic potential:

```python
PriceConstraint(asp_key=asp, qty_key=qty, dollar_key=rev)
```

### Elasticity

A log-linear relationship between price and demand:

$$
\log(\text{qty}_t) = \alpha + \varepsilon \cdot \log(\text{ASP}_t) + \eta_t
$$

where $\varepsilon$ is the price elasticity of demand, sampled from a
prior $\varepsilon \sim \mathcal{N}(\mu_\varepsilon, \sigma_\varepsilon)$.

```python
ElasticityConstraint(
    asp_key=asp, qty_key=qty,
    elasticity_prior=(-0.8, 0.3),  # mildly inelastic
)
```

## Reconciliation methods

### MCMC (default)

The harmonizer builds a NumPyro model where each series is a latent
variable centered on its belief, and constraints are added as
`numpyro.factor` potentials.  NUTS explores the joint posterior:

$$
p(\mathbf{y} \mid \text{beliefs, constraints}) \propto
\prod_i \mathcal{N}(y_i \mid \mu_i, \sigma_i / \text{trust}_i)
\;\cdot\; \prod_j \exp\!\bigl(-\lambda_j \, R_j(\mathbf{y})^2\bigr)
$$

where $R_j$ is the residual for constraint $j$.

```python
result = harmonizer.harmonize(
    beliefs,
    method="mcmc",
    lambda_hierarchy=2.0,
    lambda_price=1.0,
    num_samples=1000,
)
```

The $\lambda$ parameters control constraint strength.  Higher values
enforce tighter adherence but can distort the original beliefs more.

### Analytical (MinT projection)

When all constraints are **linear** (i.e. only
[`AdditiveConstraint`][ergodicts.harmonizer.AdditiveConstraint]) and
beliefs are Gaussian, a closed-form solution exists via the MinT/WLS
projection:

$$
\hat{\mathbf{y}} = \mathbf{y} - \mathbf{W} \mathbf{A}^\top
(\mathbf{A} \mathbf{W} \mathbf{A}^\top)^{-1}
\mathbf{A} \mathbf{y}
$$

where $\mathbf{A}$ is the constraint matrix and $\mathbf{W}$ is the
diagonal covariance (scaled by trust).  This is applied per sample,
preserving the full posterior distribution.

```python
result = harmonizer.harmonize(beliefs, method="analytical")
```

If any nonlinear constraints are present, the analytical method
automatically falls back to MCMC with a warning.

## Putting it together

```python
from ergodicts import (
    Harmonizer, ForecastBelief, PriceConstraint,
    ElasticityConstraint, ModelKey,
)

# 1. Collect beliefs from your forecaster
beliefs = Harmonizer.from_forecaster(fitted_model, horizon=12)

# 2. Blend in an external forecast
vendor_key = ModelKey(("REGION",), ("WEST",))
beliefs[vendor_key] = ForecastBelief(
    distribution="normal",
    params={"loc": vendor_mu, "scale": vendor_std},
    trust=0.5,
)

# 3. Build harmonizer with hierarchy + price + elasticity
harmonizer = Harmonizer(
    hierarchy=dep_graph,
    constraints=[
        PriceConstraint(asp_key=asp, qty_key=qty, dollar_key=rev),
        ElasticityConstraint(asp_key=asp, qty_key=qty),
    ],
)

# 4. Reconcile
result = harmonizer.harmonize(
    beliefs,
    method="mcmc",
    lambda_hierarchy=2.0,
    lambda_price=1.0,
    lambda_elasticity=0.5,
)

# 5. Inspect
result.mean()       # {ModelKey: (T,) array}
result.quantiles([0.1, 0.5, 0.9])
result.summary()    # tidy DataFrame
result.constraint_violations  # check residuals
```

## When to use which method

| Scenario | Method | Why |
|----------|--------|-----|
| Additive hierarchy only | `analytical` | Exact, instant, no tuning |
| Price or elasticity constraints | `mcmc` | Nonlinear — no closed form |
| Large hierarchies (>100 nodes) | `analytical` | Scales better than MCMC |
| Blending with external forecasts | `mcmc` | Flexible trust weighting |
