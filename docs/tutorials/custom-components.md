# Custom Components

This tutorial shows how to create your own trend or seasonality component
and use it in the forecasting pipeline.

## Architecture overview

Every component is a subclass of one of three abstract base classes:

- [`TrendComponent`][ergodicts.components.TrendComponent] — stateful
  (uses `jax.lax.scan`)
- [`SeasonalityComponent`][ergodicts.components.SeasonalityComponent] —
  stateless (deterministic function of time)
- [`RegressionComponent`][ergodicts.components.RegressionComponent] —
  stateless, data-driven

Register your component by passing a `name` keyword to the class
definition:

```python
class MyTrend(TrendComponent, name="my_trend"):
    ...
```

This auto-registers it in the `TrendComponent._registry` and makes it
available via [`ComponentLibrary`][ergodicts.components.ComponentLibrary].

## Example: Flat (constant) trend

A minimal trend that samples a single level and holds it constant:

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist

from ergodicts.components import TrendComponent


class FlatTrend(TrendComponent, name="flat_trend"):
    """Constant-level trend (no dynamics)."""

    @property
    def state_dim(self) -> int:
        return 1

    def sample_params(self, node_key: str) -> dict:
        level = numpyro.sample(f"flat_level_{node_key}", dist.Normal(0.0, 1.0))
        return {"level": level}

    def sample_innovations(self, node_key: str, T: int) -> dict:
        # No innovations needed for a constant trend
        return {"dummy": jnp.zeros(T)}

    def init_state(self, params: dict) -> jnp.ndarray:
        return jnp.array([params["level"]])

    def transition_fn(self, carry, innovations, params):
        # State doesn't change
        return carry, carry[0]

    def extract_posterior(self, node_key: str, samples: dict) -> dict:
        return {"level": samples[f"flat_level_{node_key}"]}

    def forecast_from_state(self, final_state, params, horizon, rng_key):
        # Constant level for all horizon steps
        return jnp.full(horizon, final_state[0])
```

## Using your custom component

```python
from ergodicts import NodeConfig

cfg = NodeConfig(components=(FlatTrend(),))
```

It works with the forecaster, backtester, and dashboard just like the
built-in components.

## Example: Weekly seasonality

A seasonality component for daily data with a 7-day cycle:

```python
from ergodicts.components import SeasonalityComponent


class WeeklySeasonality(SeasonalityComponent, name="weekly_seasonality"):
    """7-day dummy seasonality."""

    def sample_params(self, node_key: str) -> dict:
        raw = numpyro.sample(
            f"weekly_raw_{node_key}",
            dist.Normal(0, 1.0).expand([7]),
        )
        centred = raw - jnp.mean(raw)
        return {"effects": centred, "raw": raw}

    def contribute(self, params: dict, total_T: int) -> jnp.ndarray:
        idx = jnp.arange(total_T) % 7
        return params["effects"][idx]

    def extract_posterior(self, node_key: str, samples: dict) -> dict:
        raw = samples[f"weekly_raw_{node_key}"]
        centred = raw - jnp.mean(raw, axis=-1, keepdims=True)
        return {"effects": centred, "raw": raw}

    def forecast_contribute(self, params: dict, T_hist: int, horizon: int) -> jnp.ndarray:
        idx = jnp.arange(T_hist, T_hist + horizon) % 7
        return params["effects"][idx]
```

## Abstract methods reference

### TrendComponent

| Method | Purpose |
|--------|---------|
| `sample_params(node_key)` | Register NumPyro sample sites, return param dict |
| `sample_innovations(node_key, T)` | Sample innovation sequences of length T |
| `init_state(params)` | Return initial carry state `(state_dim,)` |
| `transition_fn(carry, innovations, params)` | Single-step transition → `(new_carry, level_t)` |
| `extract_posterior(node_key, samples)` | Extract params from the flat posterior dict |
| `forecast_from_state(final_state, params, horizon, rng_key)` | Roll forward → `(horizon,)` |

### SeasonalityComponent

| Method | Purpose |
|--------|---------|
| `sample_params(node_key)` | Register NumPyro sample sites |
| `contribute(params, total_T)` | Return `(total_T,)` seasonal contribution |
| `extract_posterior(node_key, samples)` | Extract params from posterior |
| `forecast_contribute(params, T_hist, horizon)` | Return `(horizon,)` contribution |

## Serialization

For your component to survive save/load cycles, ensure:

1. All constructor parameters are stored as instance attributes
2. The `to_dict()` method (inherited) captures them automatically
3. The `name` matches between serialization and deserialization

```python
class MyTrend(TrendComponent, name="my_trend"):
    def __init__(self, scale: float = 1.0):
        self.scale = scale  # stored as attribute → auto-serialized
```

## Tips

- Keep `sample_params` names unique per component by prefixing with the
  component name (e.g. `f"flat_level_{node_key}"`)
- Use `jax.lax.scan` in `forecast_from_state` for trends that need
  sequential state propagation
- Test your component with a single node before scaling to a hierarchy
