"""Composable dynamics components for structural time-series models.

Each component contributes a term to the predicted mean.  By default
the contributions are combined additively (``mu = trend + seasonality
+ regression``), but the aggregation strategy is itself pluggable via
:class:`Aggregator`.  Components and aggregators are configured
per-node via :class:`~ergodicts.causal_dag.NodeConfig`.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist

if TYPE_CHECKING:
    from ergodicts.causal_dag import NodeConfig
    from ergodicts.reducer import ModelKey


# ---------------------------------------------------------------------------
# Aggregator — how component contributions are combined
# ---------------------------------------------------------------------------


class Aggregator(ABC):
    """Defines how component contributions are combined into the predicted mean.

    Subclasses implement a single ``aggregate`` method that receives
    *all* contributions grouped by role and produces the final output
    in one shot.  This makes it trivial to express mixed formulas like
    ``(trend + regression) * seasonality``.

    Register a custom aggregator by subclassing with a ``name`` keyword::

        class MyAgg(Aggregator, name="my_agg"):
            ...
    """

    _registry: ClassVar[dict[str, type[Aggregator]]] = {}
    component_name: ClassVar[str]

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if name is not None:
            cls.component_name = name
            Aggregator._registry[name] = cls

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        sig = inspect.signature(self.__class__.__init__)
        params = {}
        for p_name, p in sig.parameters.items():
            if p_name == "self":
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if hasattr(self, p_name):
                params[p_name] = getattr(self, p_name)
        return {"type": self.component_name, "params": params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Aggregator:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        type_name = data["type"]
        if type_name not in cls._registry:
            raise ValueError(
                f"Unknown Aggregator type: {type_name!r}. "
                f"Available: {sorted(cls._registry)}"
            )
        klass = cls._registry[type_name]
        return klass(**data.get("params", {}))

    @abstractmethod
    def aggregate(
        self,
        contributions: dict[str, list[jnp.ndarray]],
        shape: tuple[int, ...],
    ) -> jnp.ndarray:
        """Combine all component contributions into the predicted mean.

        Parameters
        ----------
        contributions
            Maps role names (``"trend"``, ``"seasonality"``,
            ``"regression"``, ...) to lists of contribution arrays,
            each of shape ``*shape``.
        shape
            Output shape — used for identity elements when a role
            has no contributions.
        """


class AdditiveAggregator(Aggregator, name="additive"):
    """``mu = sum(all contributions across all roles)`` (default)."""

    def aggregate(
        self,
        contributions: dict[str, list[jnp.ndarray]],
        shape: tuple[int, ...],
    ) -> jnp.ndarray:
        result = jnp.zeros(shape)
        for arrays in contributions.values():
            for arr in arrays:
                result = result + arr
        return result


class MultiplicativeAggregator(Aggregator, name="multiplicative"):
    """``mu = product(all contributions across all roles)``.

    Components should output *factors* centered around 1.
    """

    def aggregate(
        self,
        contributions: dict[str, list[jnp.ndarray]],
        shape: tuple[int, ...],
    ) -> jnp.ndarray:
        result = jnp.ones(shape)
        for arrays in contributions.values():
            for arr in arrays:
                result = result * arr
        return result


class LogAdditiveAggregator(Aggregator, name="log_additive"):
    """``mu = exp(sum(all contributions across all roles))``.

    Components contribute in log-space; the result is exponentiated
    to ensure positivity.  Natural for revenue or count data.
    """

    def aggregate(
        self,
        contributions: dict[str, list[jnp.ndarray]],
        shape: tuple[int, ...],
    ) -> jnp.ndarray:
        result = jnp.zeros(shape)
        for arrays in contributions.values():
            for arr in arrays:
                result = result + arr
        return jnp.exp(result)


class MultiplicativeSeasonality(Aggregator, name="multiplicative_seasonality"):
    """``mu = (sum(trend) + sum(regression)) * product(1 + seasonality)``.

    Trend and regression contributions are additive among themselves.
    Seasonality components are interpreted as additive effects that
    become multiplicative factors via ``1 + s``.  Unknown roles
    default to additive inclusion (forward-compatible).
    """

    def aggregate(
        self,
        contributions: dict[str, list[jnp.ndarray]],
        shape: tuple[int, ...],
    ) -> jnp.ndarray:
        # Additive base: trend + regression + any unknown roles
        base = jnp.zeros(shape)
        for role, arrays in contributions.items():
            if role == "seasonality":
                continue
            for arr in arrays:
                base = base + arr

        # Multiplicative seasonality: product(1 + s_i)
        seasonal_factor = jnp.ones(shape)
        for arr in contributions.get("seasonality", []):
            seasonal_factor = seasonal_factor * (1.0 + arr)

        return base * seasonal_factor


def resolve_aggregator(cfg: NodeConfig) -> Aggregator:
    """Return the aggregator for *cfg*, defaulting to additive."""
    if cfg.aggregator is not None:
        return cfg.aggregator
    return AdditiveAggregator()


# ---------------------------------------------------------------------------
# Abstract base classes — dynamics components
# ---------------------------------------------------------------------------


class TrendComponent(ABC):
    """Stateful trend component (uses ``jax.lax.scan``).

    Register a custom trend by subclassing with a ``name`` keyword::

        class MyTrend(TrendComponent, name="my_trend"):
            ...
    """

    _registry: ClassVar[dict[str, type[TrendComponent]]] = {}
    component_name: ClassVar[str]

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if name is not None:
            cls.component_name = name
            TrendComponent._registry[name] = cls

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        sig = inspect.signature(self.__class__.__init__)
        params = {}
        for p_name, p in sig.parameters.items():
            if p_name == "self":
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if hasattr(self, p_name):
                params[p_name] = getattr(self, p_name)
        return {"type": self.component_name, "params": params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrendComponent:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        type_name = data["type"]
        if type_name not in cls._registry:
            raise ValueError(
                f"Unknown TrendComponent type: {type_name!r}. "
                f"Available: {sorted(cls._registry)}"
            )
        klass = cls._registry[type_name]
        return klass(**data.get("params", {}))

    @abstractmethod
    def sample_params(self, node_key: str) -> dict[str, Any]:
        """Register NumPyro sample sites. Return param dict."""

    @abstractmethod
    def sample_innovations(self, node_key: str, T: int) -> dict[str, jnp.ndarray]:
        """Sample innovation sequences of length *T*."""

    @abstractmethod
    def init_state(self, params: dict[str, Any]) -> jnp.ndarray:
        """Return initial carry state ``(state_dim,)``."""

    @abstractmethod
    def transition_fn(
        self,
        carry: jnp.ndarray,
        innovations: dict[str, jnp.ndarray],
        params: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Single-step transition → ``(new_carry, level_t)``."""

    @abstractmethod
    def extract_posterior(
        self, node_key: str, samples: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        """Extract this component's params from the flat posterior dict."""

    @abstractmethod
    def forecast_from_state(
        self,
        final_state: jnp.ndarray,
        params: dict[str, Any],
        horizon: int,
        rng_key: jax.Array,
    ) -> jnp.ndarray:
        """Roll forward from *final_state* → ``(horizon,)`` levels."""

    @property
    def state_dim(self) -> int:
        """Dimensionality of the carry state."""
        return 2


class SeasonalityComponent(ABC):
    """Stateless seasonality component.

    Register a custom seasonality by subclassing with a ``name`` keyword::

        class MySeason(SeasonalityComponent, name="my_season"):
            ...
    """

    _registry: ClassVar[dict[str, type[SeasonalityComponent]]] = {}
    component_name: ClassVar[str]

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if name is not None:
            cls.component_name = name
            SeasonalityComponent._registry[name] = cls

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        sig = inspect.signature(self.__class__.__init__)
        params = {}
        for p_name, p in sig.parameters.items():
            if p_name == "self":
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if hasattr(self, p_name):
                params[p_name] = getattr(self, p_name)
        return {"type": self.component_name, "params": params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeasonalityComponent:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        type_name = data["type"]
        if type_name not in cls._registry:
            raise ValueError(
                f"Unknown SeasonalityComponent type: {type_name!r}. "
                f"Available: {sorted(cls._registry)}"
            )
        klass = cls._registry[type_name]
        return klass(**data.get("params", {}))

    @abstractmethod
    def sample_params(self, node_key: str) -> dict[str, Any]:
        """Register NumPyro sample sites. Return param dict."""

    @abstractmethod
    def contribute(self, params: dict[str, Any], total_T: int) -> jnp.ndarray:
        """Return ``(total_T,)`` seasonal contribution."""

    @abstractmethod
    def extract_posterior(
        self, node_key: str, samples: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        """Extract this component's params from the flat posterior dict."""

    @abstractmethod
    def forecast_contribute(
        self, params: dict[str, Any], T_hist: int, horizon: int,
    ) -> jnp.ndarray:
        """Return ``(horizon,)`` seasonal contribution for forecast period."""


class RegressionComponent(ABC):
    """Stateless, data-driven regression component.

    Register a custom regression by subclassing with a ``name`` keyword::

        class MyReg(RegressionComponent, name="my_reg"):
            ...
    """

    _registry: ClassVar[dict[str, type[RegressionComponent]]] = {}
    component_name: ClassVar[str]

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if name is not None:
            cls.component_name = name
            RegressionComponent._registry[name] = cls

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        sig = inspect.signature(self.__class__.__init__)
        params = {}
        for p_name, p in sig.parameters.items():
            if p_name == "self":
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if hasattr(self, p_name):
                params[p_name] = getattr(self, p_name)
        return {"type": self.component_name, "params": params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegressionComponent:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        type_name = data["type"]
        if type_name not in cls._registry:
            raise ValueError(
                f"Unknown RegressionComponent type: {type_name!r}. "
                f"Available: {sorted(cls._registry)}"
            )
        klass = cls._registry[type_name]
        return klass(**data.get("params", {}))

    @abstractmethod
    def sample_params(self, node_key: str, n_predictors: int) -> dict[str, Any]:
        """Register NumPyro sample sites. Return param dict."""

    @abstractmethod
    def contribute(
        self, params: dict[str, Any], X: jnp.ndarray,
    ) -> jnp.ndarray:
        """X is ``(total_T, n_edges)``, return ``(total_T,)``."""

    @abstractmethod
    def extract_posterior(
        self, node_key: str, samples: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        """Extract this component's params from the flat posterior dict."""

    @abstractmethod
    def forecast_contribute(
        self, params: dict[str, Any], X_future: jnp.ndarray,
    ) -> jnp.ndarray:
        """X_future is ``(horizon, n_edges)``, return ``(horizon,)``."""


# ---------------------------------------------------------------------------
# Built-in trend components
# ---------------------------------------------------------------------------


class LocalLinearTrend(TrendComponent, name="local_linear_trend"):
    """Standard local linear trend: level + slope with Gaussian innovations."""

    def sample_params(self, node_key: str) -> dict[str, Any]:
        level_sigma = numpyro.sample(f"llt_level_sigma_{node_key}", dist.HalfNormal(0.5))
        slope_sigma = numpyro.sample(f"llt_slope_sigma_{node_key}", dist.HalfNormal(0.05))
        level_init = numpyro.sample(f"llt_level_init_{node_key}", dist.Normal(0.0, 1.0))
        slope_init = numpyro.sample(f"llt_slope_init_{node_key}", dist.Normal(0.0, 0.1))
        return {
            "level_sigma": level_sigma,
            "slope_sigma": slope_sigma,
            "level_init": level_init,
            "slope_init": slope_init,
        }

    def sample_innovations(self, node_key: str, T: int) -> dict[str, jnp.ndarray]:
        lev_inn = numpyro.sample(f"llt_lev_inn_{node_key}", dist.Normal(0, 1).expand([T]))
        slp_inn = numpyro.sample(f"llt_slp_inn_{node_key}", dist.Normal(0, 1).expand([T]))
        return {"lev_inn": lev_inn, "slp_inn": slp_inn}

    def init_state(self, params: dict[str, Any]) -> jnp.ndarray:
        return jnp.array([params["level_init"], params["slope_init"]])

    def transition_fn(
        self,
        carry: jnp.ndarray,
        innovations: dict[str, jnp.ndarray],
        params: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        level, slope = carry[0], carry[1]
        new_level = level + slope + params["level_sigma"] * innovations["lev_inn"]
        new_slope = slope + params["slope_sigma"] * innovations["slp_inn"]
        new_carry = jnp.array([new_level, new_slope])
        return new_carry, new_level

    def extract_posterior(
        self, node_key: str, samples: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        return {
            "level_sigma": samples[f"llt_level_sigma_{node_key}"],
            "slope_sigma": samples[f"llt_slope_sigma_{node_key}"],
            "level_init": samples[f"llt_level_init_{node_key}"],
            "slope_init": samples[f"llt_slope_init_{node_key}"],
        }

    def forecast_from_state(
        self,
        final_state: jnp.ndarray,
        params: dict[str, Any],
        horizon: int,
        rng_key: jax.Array,
    ) -> jnp.ndarray:
        k1, k2 = jr.split(rng_key)
        lev_inn = jr.normal(k1, (horizon,))
        slp_inn = jr.normal(k2, (horizon,))

        def scan_fn(carry, inputs):
            li, si = inputs
            level, slope = carry[0], carry[1]
            new_level = level + slope + params["level_sigma"] * li
            new_slope = slope + params["slope_sigma"] * si
            return jnp.array([new_level, new_slope]), new_level

        _, levels = jax.lax.scan(scan_fn, final_state, (lev_inn, slp_inn))
        return levels


class DampedLocalLinearTrend(TrendComponent, name="damped_local_linear_trend"):
    """Local linear trend with damped slope (phi < 1)."""

    def sample_params(self, node_key: str) -> dict[str, Any]:
        level_sigma = numpyro.sample(f"dllt_level_sigma_{node_key}", dist.HalfNormal(0.5))
        slope_sigma = numpyro.sample(f"dllt_slope_sigma_{node_key}", dist.HalfNormal(0.05))
        level_init = numpyro.sample(f"dllt_level_init_{node_key}", dist.Normal(0.0, 1.0))
        slope_init = numpyro.sample(f"dllt_slope_init_{node_key}", dist.Normal(0.0, 0.1))
        phi = numpyro.sample(f"dllt_phi_{node_key}", dist.Beta(8.0, 2.0))
        return {
            "level_sigma": level_sigma,
            "slope_sigma": slope_sigma,
            "level_init": level_init,
            "slope_init": slope_init,
            "phi": phi,
        }

    def sample_innovations(self, node_key: str, T: int) -> dict[str, jnp.ndarray]:
        lev_inn = numpyro.sample(f"dllt_lev_inn_{node_key}", dist.Normal(0, 1).expand([T]))
        slp_inn = numpyro.sample(f"dllt_slp_inn_{node_key}", dist.Normal(0, 1).expand([T]))
        return {"lev_inn": lev_inn, "slp_inn": slp_inn}

    def init_state(self, params: dict[str, Any]) -> jnp.ndarray:
        return jnp.array([params["level_init"], params["slope_init"]])

    def transition_fn(
        self,
        carry: jnp.ndarray,
        innovations: dict[str, jnp.ndarray],
        params: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        level, slope = carry[0], carry[1]
        new_level = level + slope + params["level_sigma"] * innovations["lev_inn"]
        new_slope = params["phi"] * slope + params["slope_sigma"] * innovations["slp_inn"]
        new_carry = jnp.array([new_level, new_slope])
        return new_carry, new_level

    def extract_posterior(
        self, node_key: str, samples: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        return {
            "level_sigma": samples[f"dllt_level_sigma_{node_key}"],
            "slope_sigma": samples[f"dllt_slope_sigma_{node_key}"],
            "level_init": samples[f"dllt_level_init_{node_key}"],
            "slope_init": samples[f"dllt_slope_init_{node_key}"],
            "phi": samples[f"dllt_phi_{node_key}"],
        }

    def forecast_from_state(
        self,
        final_state: jnp.ndarray,
        params: dict[str, Any],
        horizon: int,
        rng_key: jax.Array,
    ) -> jnp.ndarray:
        k1, k2 = jr.split(rng_key)
        lev_inn = jr.normal(k1, (horizon,))
        slp_inn = jr.normal(k2, (horizon,))

        def scan_fn(carry, inputs):
            li, si = inputs
            level, slope = carry[0], carry[1]
            new_level = level + slope + params["level_sigma"] * li
            new_slope = params["phi"] * slope + params["slope_sigma"] * si
            return jnp.array([new_level, new_slope]), new_level

        _, levels = jax.lax.scan(scan_fn, final_state, (lev_inn, slp_inn))
        return levels


class OUMeanReversion(TrendComponent, name="ou_mean_reversion"):
    """Ornstein-Uhlenbeck mean-reverting process.

    ``level_{t+1} = level_t + theta * (mu - level_t) + sigma * eps``

    State: ``[level]`` (1-dimensional).
    """

    @property
    def state_dim(self) -> int:
        return 1

    def sample_params(self, node_key: str) -> dict[str, Any]:
        theta = numpyro.sample(f"ou_theta_{node_key}", dist.HalfNormal(0.5))
        mu = numpyro.sample(f"ou_mu_{node_key}", dist.Normal(0.0, 1.0))
        sigma = numpyro.sample(f"ou_sigma_{node_key}", dist.HalfNormal(0.5))
        level_init = numpyro.sample(f"ou_level_init_{node_key}", dist.Normal(0.0, 1.0))
        return {"theta": theta, "mu": mu, "sigma": sigma, "level_init": level_init}

    def sample_innovations(self, node_key: str, T: int) -> dict[str, jnp.ndarray]:
        inn = numpyro.sample(f"ou_inn_{node_key}", dist.Normal(0, 1).expand([T]))
        return {"inn": inn}

    def init_state(self, params: dict[str, Any]) -> jnp.ndarray:
        return jnp.array([params["level_init"]])

    def transition_fn(
        self,
        carry: jnp.ndarray,
        innovations: dict[str, jnp.ndarray],
        params: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        level = carry[0]
        new_level = level + params["theta"] * (params["mu"] - level) + params["sigma"] * innovations["inn"]
        return jnp.array([new_level]), new_level

    def extract_posterior(
        self, node_key: str, samples: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        return {
            "theta": samples[f"ou_theta_{node_key}"],
            "mu": samples[f"ou_mu_{node_key}"],
            "sigma": samples[f"ou_sigma_{node_key}"],
            "level_init": samples[f"ou_level_init_{node_key}"],
        }

    def forecast_from_state(
        self,
        final_state: jnp.ndarray,
        params: dict[str, Any],
        horizon: int,
        rng_key: jax.Array,
    ) -> jnp.ndarray:
        inn = jr.normal(rng_key, (horizon,))

        def scan_fn(carry, eps):
            level = carry[0]
            new_level = level + params["theta"] * (params["mu"] - level) + params["sigma"] * eps
            return jnp.array([new_level]), new_level

        _, levels = jax.lax.scan(scan_fn, final_state, inn)
        return levels


# ---------------------------------------------------------------------------
# Built-in seasonality components
# ---------------------------------------------------------------------------


class FourierSeasonality(SeasonalityComponent, name="fourier_seasonality"):
    """Fourier harmonic seasonality with configurable period and harmonics."""

    def __init__(self, n_harmonics: int = 2, period: float = 12.0) -> None:
        self.n_harmonics = n_harmonics
        self.period = period

    def sample_params(self, node_key: str) -> dict[str, Any]:
        coeffs = numpyro.sample(
            f"fourier_season_{node_key}",
            dist.Normal(0, 0.5).expand([2 * self.n_harmonics]),
        )
        return {"coeffs": coeffs}

    def contribute(self, params: dict[str, Any], total_T: int) -> jnp.ndarray:
        t_idx = jnp.arange(total_T, dtype=jnp.float32)
        coeffs = params["coeffs"]
        result = jnp.zeros(total_T)
        for h in range(self.n_harmonics):
            freq = 2.0 * jnp.pi * (h + 1) * t_idx / self.period
            result = result + coeffs[2 * h] * jnp.sin(freq) + coeffs[2 * h + 1] * jnp.cos(freq)
        return result

    def extract_posterior(
        self, node_key: str, samples: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        return {"coeffs": samples[f"fourier_season_{node_key}"]}

    def forecast_contribute(
        self, params: dict[str, Any], T_hist: int, horizon: int,
    ) -> jnp.ndarray:
        t_idx = jnp.arange(T_hist, T_hist + horizon, dtype=jnp.float32)
        coeffs = params["coeffs"]
        result = jnp.zeros(horizon)
        for h in range(self.n_harmonics):
            freq = 2.0 * jnp.pi * (h + 1) * t_idx / self.period
            result = result + coeffs[2 * h] * jnp.sin(freq) + coeffs[2 * h + 1] * jnp.cos(freq)
        return result


class MultiplicativeFourierSeasonality(SeasonalityComponent, name="multiplicative_fourier_seasonality"):
    """Fourier seasonality normalised for multiplicative composition.

    Uses Fourier harmonics to parameterise the seasonal shape, then
    applies **softmax** over one full period to produce positive weights
    that sum to 1.  The weights are scaled by the period length *P* so
    that the mean seasonal factor equals 1 over a complete cycle.

    The component outputs ``factor − 1``, so when paired with
    :class:`MultiplicativeSeasonality` (which applies ``1 + s``), the
    effective multiplier is the normalised factor itself.

    Parameters
    ----------
    n_harmonics : int
        Number of Fourier harmonic pairs (sin + cos).
    period : int
        Length of one seasonal cycle (e.g. 12 for monthly data).
    """

    def __init__(self, n_harmonics: int = 2, period: int = 12) -> None:
        self.n_harmonics = n_harmonics
        self.period = period

    # -- internals ---------------------------------------------------------

    def _raw_fourier(self, coeffs: jnp.ndarray, t_idx: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the raw (un-normalised) Fourier pattern."""
        result = jnp.zeros_like(t_idx, dtype=jnp.float32)
        for h in range(self.n_harmonics):
            freq = 2.0 * jnp.pi * (h + 1) * t_idx / self.period
            result = result + coeffs[2 * h] * jnp.sin(freq) + coeffs[2 * h + 1] * jnp.cos(freq)
        return result

    def _factors(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Return ``(period,)`` positive factors with mean 1."""
        t_period = jnp.arange(self.period, dtype=jnp.float32)
        raw = self._raw_fourier(coeffs, t_period)
        weights = jax.nn.softmax(raw)          # sum = 1
        return self.period * weights            # mean = 1

    # -- SeasonalityComponent interface ------------------------------------

    def sample_params(self, node_key: str) -> dict[str, Any]:
        coeffs = numpyro.sample(
            f"mult_fourier_season_{node_key}",
            dist.Normal(0, 0.5).expand([2 * self.n_harmonics]),
        )
        return {"coeffs": coeffs}

    def contribute(self, params: dict[str, Any], total_T: int) -> jnp.ndarray:
        factors = self._factors(params["coeffs"])       # (P,)
        idx = jnp.arange(total_T) % self.period
        return factors[idx] - 1.0                       # 1 + output = factor

    def extract_posterior(
        self, node_key: str, samples: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        return {"coeffs": samples[f"mult_fourier_season_{node_key}"]}

    def forecast_contribute(
        self, params: dict[str, Any], T_hist: int, horizon: int,
    ) -> jnp.ndarray:
        factors = self._factors(params["coeffs"])       # (P,)
        idx = jnp.arange(T_hist, T_hist + horizon) % self.period
        return factors[idx] - 1.0


class MonthlySeasonality(SeasonalityComponent, name="monthly_seasonality"):
    """12 month-level dummy effects with soft sum-to-zero constraint."""

    def sample_params(self, node_key: str) -> dict[str, Any]:
        raw = numpyro.sample(
            f"monthly_raw_{node_key}",
            dist.Normal(0, 1.0).expand([12]),
        )
        centred = raw - jnp.mean(raw)
        return {"month_effects": centred, "raw": raw}

    def contribute(self, params: dict[str, Any], total_T: int) -> jnp.ndarray:
        month_idx = jnp.arange(total_T) % 12
        return params["month_effects"][month_idx]

    def extract_posterior(
        self, node_key: str, samples: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        raw = samples[f"monthly_raw_{node_key}"]  # (S, 12)
        centred = raw - jnp.mean(raw, axis=-1, keepdims=True)
        return {"month_effects": centred, "raw": raw}

    def forecast_contribute(
        self, params: dict[str, Any], T_hist: int, horizon: int,
    ) -> jnp.ndarray:
        month_idx = jnp.arange(T_hist, T_hist + horizon) % 12
        return params["month_effects"][month_idx]


class MultiplicativeMonthlySeasonality(SeasonalityComponent, name="multiplicative_monthly_seasonality"):
    """Per-period dummy seasonality normalised for multiplicative composition.

    Samples *N* free effects (one per period slot), applies **softmax** to
    produce positive weights that sum to 1, then scales by *N* so that the
    mean factor over one full cycle equals 1.

    Outputs ``factor − 1`` for pairing with
    :class:`MultiplicativeSeasonality` (which applies ``1 + s``).

    Parameters
    ----------
    period : int
        Number of seasonal slots (e.g. 12 for monthly, 4 for quarterly).
    """

    def __init__(self, period: int = 12) -> None:
        self.period = period

    def _factors(self, raw: jnp.ndarray) -> jnp.ndarray:
        """Return ``(period,)`` positive factors with mean 1."""
        weights = jax.nn.softmax(raw)       # sum = 1
        return self.period * weights        # mean = 1

    def sample_params(self, node_key: str) -> dict[str, Any]:
        raw = numpyro.sample(
            f"mult_monthly_raw_{node_key}",
            dist.Normal(0, 1.0).expand([self.period]),
        )
        return {"raw": raw}

    def contribute(self, params: dict[str, Any], total_T: int) -> jnp.ndarray:
        factors = self._factors(params["raw"])          # (period,)
        idx = jnp.arange(total_T) % self.period
        return factors[idx] - 1.0

    def extract_posterior(
        self, node_key: str, samples: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        return {"raw": samples[f"mult_monthly_raw_{node_key}"]}

    def forecast_contribute(
        self, params: dict[str, Any], T_hist: int, horizon: int,
    ) -> jnp.ndarray:
        factors = self._factors(params["raw"])          # (period,)
        idx = jnp.arange(T_hist, T_hist + horizon) % self.period
        return factors[idx] - 1.0


# ---------------------------------------------------------------------------
# Built-in regression component
# ---------------------------------------------------------------------------


class ExternalRegression(RegressionComponent, name="external_regression"):
    """Linear regression on external predictors."""

    def sample_params(self, node_key: str, n_predictors: int) -> dict[str, Any]:
        betas = numpyro.sample(
            f"reg_beta_{node_key}",
            dist.Normal(0, 1.0).expand([n_predictors]),
        )
        return {"betas": betas}

    def contribute(
        self, params: dict[str, Any], X: jnp.ndarray,
    ) -> jnp.ndarray:
        # X: (total_T, n_edges), betas: (n_edges,)
        return jnp.dot(X, params["betas"])

    def extract_posterior(
        self, node_key: str, samples: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        return {"betas": samples[f"reg_beta_{node_key}"]}

    def forecast_contribute(
        self, params: dict[str, Any], X_future: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.dot(X_future, params["betas"])


# ---------------------------------------------------------------------------
# Component resolution
# ---------------------------------------------------------------------------

# Union type for convenience
Component = TrendComponent | SeasonalityComponent | RegressionComponent


def resolve_components(cfg: NodeConfig) -> list[Component]:
    """Return the component list for *cfg*.

    When ``cfg.components`` is ``None``, defaults to a single
    :class:`LocalLinearTrend`.
    """
    if cfg.components is not None:
        return list(cfg.components)
    return [LocalLinearTrend()]


# ---------------------------------------------------------------------------
# Role mapping — component type → role string
# ---------------------------------------------------------------------------

ROLE_MAP: dict[type, str] = {
    TrendComponent: "trend",
    SeasonalityComponent: "seasonality",
    RegressionComponent: "regression",
}


def component_role(comp: Component) -> str:
    """Return the role string for a component instance."""
    for base_type, role in ROLE_MAP.items():
        if isinstance(comp, base_type):
            return role
    return "unknown"


# ---------------------------------------------------------------------------
# Component library — unified registry browser
# ---------------------------------------------------------------------------


class ComponentLibrary:
    """Unified registry for all component and aggregator types.

    Components register themselves automatically via ``__init_subclass__``
    when they declare a ``name`` keyword::

        class MyTrend(TrendComponent, name="my_trend"):
            ...

    Use :meth:`all_components` to browse the full library, or
    :meth:`from_dict` to deserialize any component/aggregator from a
    dict produced by ``to_dict()``.
    """

    _BASE_CLASSES: ClassVar[dict[str, type]] = {
        "trend": TrendComponent,
        "seasonality": SeasonalityComponent,
        "regression": RegressionComponent,
        "aggregator": Aggregator,
    }

    @classmethod
    def all_components(cls) -> dict[str, dict[str, Any]]:
        """Return the full library as a dict keyed by component name.

        Each entry contains ``role``, ``class_name``, ``description``,
        and ``params`` (with defaults and type annotations).
        """
        result: dict[str, dict[str, Any]] = {}
        for role, base_cls in cls._BASE_CLASSES.items():
            for comp_name, comp_cls in base_cls._registry.items():
                sig = inspect.signature(comp_cls.__init__)
                params: dict[str, dict[str, Any]] = {}
                for p_name, param in sig.parameters.items():
                    if p_name == "self":
                        continue
                    if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                        continue
                    default = (
                        param.default
                        if param.default is not inspect.Parameter.empty
                        else None
                    )
                    annotation = (
                        param.annotation.__name__
                        if hasattr(param.annotation, "__name__")
                        else str(param.annotation)
                        if param.annotation is not inspect.Parameter.empty
                        else "Any"
                    )
                    params[p_name] = {"default": default, "annotation": annotation}

                doc = (comp_cls.__doc__ or "").strip().split("\n")[0]
                result[comp_name] = {
                    "role": role,
                    "class_name": comp_cls.__name__,
                    "description": doc,
                    "params": params,
                }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Deserialize any component or aggregator from a dict.

        Searches all base class registries until a match is found.
        """
        type_name = data["type"]
        for base_cls in cls._BASE_CLASSES.values():
            if type_name in base_cls._registry:
                return base_cls.from_dict(data)
        raise ValueError(
            f"Unknown component type: {type_name!r}. "
            f"Available: {sorted(n for bc in cls._BASE_CLASSES.values() for n in bc._registry)}"
        )
