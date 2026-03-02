# Bayesian Inference

## Overview

Ergodicts fits a **Bayesian structural time-series (STS)** model using
[NumPyro](https://num.pyro.ai/).  The generative model is:

$$
y_t \sim \mathcal{N}\!\bigl(\mu_t,\; \sigma_{\text{obs}}^2\bigr)
$$

where $\mu_t$ is the sum (or product) of component contributions —
trend, seasonality, regression — as described in
[Structural Components](components.md).

All model parameters have priors; inference produces a **posterior
distribution** over parameters, which propagates uncertainty into
forecasts automatically.

## Data preprocessing

Before fitting, [`prepare_data`][ergodicts.forecaster.prepare_data]
transforms raw arrays:

### Internal series — ratio scaling

Each internal series $y$ is divided by its median:

$$
\tilde{y}_t = \frac{y_t}{\text{median}(y)}
$$

This centres values around 1.0 while preserving positivity and
multiplicative structure.  Use
[`ForecastData.unstandardize`][ergodicts.forecaster.ForecastData.unstandardize]
to map back to the original scale.

### External predictors — z-score

External series are z-score standardised:

$$
\tilde{x}_t = \frac{x_t - \bar{x}}{s_x}
$$

This puts all predictors on a comparable scale so that regression
coefficient priors ($\beta \sim \mathcal{N}(0, 1)$) are sensible.

NaN values are forward-filled before standardisation.

## Inference methods

### NUTS (default)

The No-U-Turn Sampler — a variant of Hamiltonian Monte Carlo.  Produces
exact posterior samples but is computationally expensive.

```python
model.fit(y_data, x_data, inference="nuts", num_warmup=500, num_samples=1000)
```

- **num_warmup**: adaptation phase (learning step size and mass matrix)
- **num_samples**: posterior samples to draw
- **num_chains**: independent chains (>1 enables $\hat{R}$ diagnostics)

### SVI (fast approximate)

Stochastic Variational Inference fits a Gaussian approximation to the
posterior.  Much faster but may underestimate uncertainty.

```python
model.fit(y_data, x_data, inference="svi", svi_steps=5000, num_samples=500)
```

- **svi_steps**: optimisation iterations
- **svi_lr**: Adam learning rate
- **num_samples**: samples drawn from the fitted guide

## Vectorized implementation

For hierarchies with many nodes, ergodicts **batches** nodes by trend
component type and runs them through a single `jax.vmap`-ed `jax.lax.scan`.
This dramatically reduces XLA compilation time compared to a per-node loop.

For example, if 50 leaf nodes all use `LocalLinearTrend`, their state-space
scans are stacked into tensors of shape `(50, T, 2)` and executed as one
vectorised operation.

Batching is now **owned by the component**, not the forecaster.  Each
built-in `TrendComponent` subclass implements `prepare_batch_data`,
`_make_scan_step_fn`, and `batched_scan`.  Custom trend components
get an automatic fallback sequential scan.

## Reconciliation strategies

Reconciliation controls how leaf-node forecasts produce root-node
forecasts.  The forecaster uses a **strategy pattern** with three
options:

| Strategy | Description |
|----------|-------------|
| `"bottom_up"` | Root = deterministic sum of leaf forecasts (default) |
| `"soft"` | Same as bottom-up for now (placeholder for future Gaussian potential) |
| `"none"` | No root forecasts produced |

## Forecasting

After fitting, [`forecast`][ergodicts.forecaster.HierarchicalForecaster.forecast]
generates posterior predictive samples:

1. For each posterior sample $s = 1, \dots, S$:
   - Extract the final state from the fitted scan
   - Roll forward trend dynamics for $H$ steps using fresh innovations
   - Compute seasonality and regression contributions
   - Aggregate via the node's aggregator
   - Add observation noise: $\hat{y}^{(s)}_t = \mu^{(s)}_t + \sigma^{(s)}_{\text{obs}} \cdot \varepsilon_t$
2. Root forecasts (bottom-up) = sum of leaf forecasts

The result is a `(S, H)` array per node — a full posterior predictive
distribution.

## Diagnostics

After fitting with NUTS, check convergence:

- **$\hat{R}$ (R-hat)**: should be < 1.05 for all parameters
- **Effective sample size (n_eff)**: should be > 100
- **Divergences**: should be 0

The [backtester][ergodicts.backtester.Backtester] automatically extracts
these diagnostics and saves them with each fold.

```python
model.summary()  # prints MCMC diagnostics table
```
