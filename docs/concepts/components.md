# Structural Components

## Structural decomposition

Ergodicts models each time-series as a **sum** (or product) of interpretable
components:

$$
y_t = \underbrace{\ell_t}_{\text{trend}} + \underbrace{s_t}_{\text{seasonality}} + \underbrace{r_t}_{\text{regression}} + \varepsilon_t
$$

Each component is a separate Python class that registers NumPyro sample sites
and produces a contribution array.  The
[aggregator][ergodicts.components.Aggregator] controls how contributions
are combined.

## Trend components

Trends capture the slowly-evolving level of the series.  All trends are
**stateful** — they use `jax.lax.scan` to propagate a hidden state forward
in time.

### Local Linear Trend

The default.  Level and slope both follow random walks:

$$
\ell_t = \ell_{t-1} + b_{t-1} + \sigma_\ell\,\varepsilon^\ell_t, \qquad
b_t = b_{t-1} + \sigma_b\,\varepsilon^b_t
$$

Use when you expect the series to trend up or down without bound.

### Damped Local Linear Trend

Adds an autoregressive damping factor $\varphi \in (0,1)$ to the slope:

$$
b_t = \varphi\, b_{t-1} + \sigma_b\,\varepsilon^b_t
$$

The slope decays towards zero with half-life $\ln 2 / (-\ln\varphi)$.
Produces more conservative long-horizon forecasts than the undamped variant.

### Ornstein--Uhlenbeck Mean Reversion

For series that oscillate around a stable mean $\mu$:

$$
\ell_t = \ell_{t-1} + \theta\,(\mu - \ell_{t-1}) + \sigma\,\varepsilon_t
$$

Half-life of mean reversion is $\ln 2 / \theta$.

## Seasonality components

Seasonality components are **stateless** — they compute a deterministic
function of time given their sampled parameters.

### Fourier Seasonality

Models the seasonal pattern with $H$ harmonic pairs:

$$
s_t = \sum_{h=1}^{H} \bigl(a_h \sin(2\pi h\, t / P) + b_h \cos(2\pi h\, t / P)\bigr)
$$

- `n_harmonics=1`: smooth sinusoidal pattern
- `n_harmonics=2`: captures asymmetric peaks
- `n_harmonics=6` (with `period=12`): equivalent to 12 free parameters

### Multiplicative Fourier Seasonality

Like Fourier seasonality, but outputs are normalised via softmax so the
mean factor over one period equals 1.  Pair with
[`MultiplicativeSeasonality`][ergodicts.components.MultiplicativeSeasonality]
aggregator for models where seasonality scales proportionally with the level.

### Monthly Seasonality

12 free dummy effects centred to sum to zero.  More flexible than Fourier
(can represent any monthly pattern) but uses more parameters.

### Multiplicative Monthly Seasonality

Like monthly seasonality, but softmax-normalised for multiplicative
composition.

## Regression component

The [`ExternalRegression`][ergodicts.components.ExternalRegression] adds
linear effects from external predictors:

$$
r_t = \sum_j \beta_j\, x_{j,t}
$$

Predictors are linked via the [CausalDAG][ergodicts.causal_dag.CausalDAG].
This component is **automatically included** for any leaf node with incoming
DAG edges — you don't add it to `NodeConfig.components`.

## Aggregator selection guide

| Aggregator | Formula | When to use |
|---|---|---|
| [`AdditiveAggregator`][ergodicts.components.AdditiveAggregator] | $\mu = \text{trend} + \text{seas} + \text{reg}$ | Default.  Good for centred data. |
| [`MultiplicativeSeasonality`][ergodicts.components.MultiplicativeSeasonality] | $\mu = (\text{trend} + \text{reg}) \times (1 + s)$ | Revenue/sales where seasonal swings scale with level. |
| [`MultiplicativeAggregator`][ergodicts.components.MultiplicativeAggregator] | $\mu = \prod_c f_c$ | All components output factors around 1. |
| [`LogAdditiveAggregator`][ergodicts.components.LogAdditiveAggregator] | $\mu = \exp(\sum_c f_c)$ | Ensures positivity.  Natural for counts. |

## Configuring components per node

```python
from ergodicts import NodeConfig
from ergodicts.components import (
    DampedLocalLinearTrend,
    FourierSeasonality,
    MultiplicativeSeasonality,
)

# Additive trend + seasonality (default aggregator)
cfg_additive = NodeConfig(
    components=(DampedLocalLinearTrend(), FourierSeasonality(n_harmonics=2)),
)

# Multiplicative seasonality
cfg_mult = NodeConfig(
    components=(DampedLocalLinearTrend(), FourierSeasonality(n_harmonics=2)),
    aggregator=MultiplicativeSeasonality(),
)
```
