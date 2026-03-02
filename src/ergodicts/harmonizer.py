"""Harmonizer — post-hoc reconciliation of hierarchical forecasts.

Reconciles forecasts so they satisfy hierarchical consistency
(parent = sum of children), price consistency (ASP × qty = dollars),
and elasticity relationships using NumPyro/JAX.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from ergodicts.reducer import DependencyGraph, ModelKey

if TYPE_CHECKING:
    import pandas as pd

    from ergodicts.forecaster import HierarchicalForecaster

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ForecastBelief — universal input type
# ---------------------------------------------------------------------------

_SUPPORTED_DISTRIBUTIONS: dict[str, type[dist.Distribution]] = {
    "normal": dist.Normal,
    "studentt": dist.StudentT,
    "lognormal": dist.LogNormal,
    "laplace": dist.Laplace,
}


@dataclass
class ForecastBelief:
    """What we believe about a single time series, from any source.

    Provide *either* ``samples`` (existing posterior draws) *or*
    ``distribution`` + ``params`` (parametric specification).

    Parameters
    ----------
    samples
        Pre-existing posterior draws, shape ``(num_samples, T)``.
    distribution
        Name of a ``numpyro.distributions`` class — one of
        ``"normal"``, ``"studentt"``, ``"lognormal"``, ``"laplace"``.
    params
        Distribution parameters as numpy arrays, each of shape ``(T,)``.
        For ``normal``: ``{"loc": ..., "scale": ...}``.
        For ``studentt``: ``{"df": ..., "loc": ..., "scale": ...}``.
    trust
        Higher values mean tighter prior (scales down the std).
        Default ``1.0`` (no adjustment).
    """

    samples: jnp.ndarray | None = None
    distribution: str | None = None
    params: dict[str, np.ndarray] | None = None
    trust: float = 1.0

    def __post_init__(self) -> None:
        has_samples = self.samples is not None
        has_dist = self.distribution is not None and self.params is not None
        if has_samples == has_dist:
            raise ValueError(
                "Provide exactly one of `samples` or (`distribution` + `params`)."
            )
        if self.distribution is not None and self.distribution not in _SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution {self.distribution!r}. "
                f"Choose from {list(_SUPPORTED_DISTRIBUTIONS)}."
            )

    def draw(self, num_samples: int, rng_key: jax.Array) -> jnp.ndarray:
        """Produce ``(num_samples, T)`` samples from this belief."""
        if self.samples is not None:
            existing = jnp.asarray(self.samples)
            n, T = existing.shape
            if n == num_samples:
                return existing
            # resample with replacement
            idx = jax.random.choice(rng_key, n, shape=(num_samples,), replace=True)
            return existing[idx]

        # parametric path
        assert self.distribution is not None and self.params is not None
        dist_cls = _SUPPORTED_DISTRIBUTIONS[self.distribution]
        # broadcast params to (num_samples, T)
        param_arrays = {k: jnp.asarray(v) for k, v in self.params.items()}
        d = dist_cls(**param_arrays)
        return d.sample(rng_key, sample_shape=(num_samples,))


# ---------------------------------------------------------------------------
# Constraint — abstract protocol
# ---------------------------------------------------------------------------


class Constraint(ABC):
    """Abstract constraint that relates multiple forecast series."""

    @abstractmethod
    def nodes(self) -> list[ModelKey]:
        """All ModelKeys involved in this constraint."""

    @abstractmethod
    def apply_potential(
        self,
        factors: dict[ModelKey, Any],
        lambda_weight: float,
    ) -> Any:
        """Return a NumPyro Potential log-prob term (a scalar)."""

    @abstractmethod
    def is_linear(self) -> bool:
        """True if this constraint can be expressed as A @ x = 0."""

    def linear_coefficients(self, ordered_keys: list[ModelKey]) -> jnp.ndarray | None:
        """Return coefficient row for the constraint matrix. None if nonlinear."""
        return None


# ---------------------------------------------------------------------------
# AdditiveConstraint — parent = sum(children)
# ---------------------------------------------------------------------------


class AdditiveConstraint(Constraint):
    """Enforces ``parent = sum(children)`` via a quadratic penalty."""

    def __init__(self, parent: ModelKey, children: list[ModelKey]) -> None:
        self.parent = parent
        self.children = list(children)

    def nodes(self) -> list[ModelKey]:
        return [self.parent, *self.children]

    def is_linear(self) -> bool:
        return True

    def linear_coefficients(self, ordered_keys: list[ModelKey]) -> jnp.ndarray:
        coeffs = np.zeros(len(ordered_keys))
        for i, key in enumerate(ordered_keys):
            if key == self.parent:
                coeffs[i] = 1.0
            elif key in self.children:
                coeffs[i] = -1.0
        return jnp.array(coeffs)

    def apply_potential(
        self,
        factors: dict[ModelKey, Any],
        lambda_weight: float,
    ) -> Any:
        parent_val = factors[self.parent]
        children_sum = sum(factors[c] for c in self.children)
        residual = parent_val - children_sum
        # normalise by scale of the parent
        scale = jnp.maximum(jnp.abs(parent_val).mean(), 1e-6)
        return -lambda_weight * jnp.sum((residual / scale) ** 2)

    def __repr__(self) -> str:
        kids = ", ".join(str(c) for c in self.children)
        return f"AdditiveConstraint({self.parent} = sum[{kids}])"


# ---------------------------------------------------------------------------
# PriceConstraint — ASP × qty ≈ dollars
# ---------------------------------------------------------------------------


class PriceConstraint(Constraint):
    """Enforces ``asp * qty ≈ dollars`` via a quadratic penalty."""

    def __init__(
        self,
        asp_key: ModelKey,
        qty_key: ModelKey,
        dollar_key: ModelKey,
    ) -> None:
        self.asp_key = asp_key
        self.qty_key = qty_key
        self.dollar_key = dollar_key

    def nodes(self) -> list[ModelKey]:
        return [self.asp_key, self.qty_key, self.dollar_key]

    def is_linear(self) -> bool:
        return False

    def apply_potential(
        self,
        factors: dict[ModelKey, Any],
        lambda_weight: float,
    ) -> Any:
        asp = factors[self.asp_key]
        qty = factors[self.qty_key]
        dollars = factors[self.dollar_key]
        residual = asp * qty - dollars
        # Normalise by per-timestep dollar magnitude so the penalty
        # is scale-free (a 1% residual gets the same penalty regardless
        # of whether dollars are ~100 or ~100k).
        scale = jnp.maximum(jnp.abs(dollars), 1e-6)
        return -lambda_weight * jnp.sum((residual / scale) ** 2)

    def __repr__(self) -> str:
        return (
            f"PriceConstraint({self.asp_key} × {self.qty_key} "
            f"= {self.dollar_key})"
        )


# ---------------------------------------------------------------------------
# ElasticityConstraint — log(qty) ~ elasticity × log(asp) + intercept
# ---------------------------------------------------------------------------


class ElasticityConstraint(Constraint):
    """Penalises deviations from a log-linear price–demand relationship.

    Parameters
    ----------
    asp_key, qty_key
        ModelKeys for average selling price and quantity.
    elasticity_prior
        ``(mean, std)`` for the elasticity coefficient. Typical: ``(-1.0, 0.5)``
        (unit elastic with moderate uncertainty).
    """

    def __init__(
        self,
        asp_key: ModelKey,
        qty_key: ModelKey,
        elasticity_prior: tuple[float, float] = (-1.0, 0.5),
    ) -> None:
        self.asp_key = asp_key
        self.qty_key = qty_key
        self.elasticity_mean, self.elasticity_std = elasticity_prior

    def nodes(self) -> list[ModelKey]:
        return [self.asp_key, self.qty_key]

    def is_linear(self) -> bool:
        return False

    def apply_potential(
        self,
        factors: dict[ModelKey, Any],
        lambda_weight: float,
        elasticity: Any | None = None,
    ) -> Any:
        asp = factors[self.asp_key]
        qty = factors[self.qty_key]
        log_asp = jnp.log(jnp.maximum(asp, 1e-8))
        log_qty = jnp.log(jnp.maximum(qty, 1e-8))

        if elasticity is None:
            elasticity = self.elasticity_mean

        # intercept from means
        intercept = log_qty.mean() - elasticity * log_asp.mean()
        residual = log_qty - (intercept + elasticity * log_asp)
        return -lambda_weight * jnp.sum(residual**2)

    def __repr__(self) -> str:
        return (
            f"ElasticityConstraint({self.qty_key} ~ "
            f"ε({self.elasticity_mean:.2f}±{self.elasticity_std:.2f}) "
            f"× {self.asp_key})"
        )


# ---------------------------------------------------------------------------
# HarmonizedResult — output container
# ---------------------------------------------------------------------------


@dataclass
class HarmonizedResult:
    """Output of :meth:`Harmonizer.harmonize`.

    Attributes
    ----------
    samples
        Reconciled posterior samples, ``{ModelKey: (num_samples, T)}``.
    constraint_violations
        Per-constraint residuals post-harmonization, ``{name: (T,)}``.
    """

    samples: dict[ModelKey, np.ndarray]
    constraint_violations: dict[str, np.ndarray] = field(default_factory=dict)

    def mean(self) -> dict[ModelKey, np.ndarray]:
        """Per-series mean forecast, shape ``(T,)``."""
        return {k: np.asarray(v).mean(axis=0) for k, v in self.samples.items()}

    def quantiles(self, q: list[float]) -> dict[ModelKey, np.ndarray]:
        """Per-series quantiles, shape ``(len(q), T)``."""
        return {
            k: np.quantile(np.asarray(v), q, axis=0) for k, v in self.samples.items()
        }

    def summary(self) -> pd.DataFrame:
        """Tidy DataFrame with mean, std, and quantiles per series per timestep."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for key, samp in self.samples.items():
            arr = np.asarray(samp)
            mu = arr.mean(axis=0)
            std = arr.std(axis=0)
            q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9], axis=0)
            for t in range(arr.shape[1]):
                rows.append(
                    {
                        "key": str(key),
                        "label": key.label,
                        "t": t,
                        "mean": mu[t],
                        "std": std[t],
                        "q10": q10[t],
                        "median": q50[t],
                        "q90": q90[t],
                    }
                )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Harmonizer — main class
# ---------------------------------------------------------------------------


class Harmonizer:
    """Post-hoc reconciliation of hierarchical forecasts.

    Reconciles a set of :class:`ForecastBelief` objects so they satisfy
    additive hierarchy, price identity, and elasticity constraints.

    Parameters
    ----------
    constraints
        Explicit constraint list. If *hierarchy* is also given, the
        auto-generated :class:`AdditiveConstraint` instances are prepended.
    hierarchy
        A :class:`DependencyGraph` — one :class:`AdditiveConstraint`
        is created for every parent automatically.
    """

    def __init__(
        self,
        constraints: list[Constraint] | None = None,
        hierarchy: DependencyGraph | None = None,
    ) -> None:
        built: list[Constraint] = []
        if hierarchy is not None:
            for parent in hierarchy.all_parents:
                children = sorted(hierarchy.children_of(parent))
                if children:
                    built.append(AdditiveConstraint(parent, children))
        if constraints:
            built.extend(constraints)
        self.constraints = built

    # -- convenience ----------------------------------------------------------

    @staticmethod
    def from_forecaster(
        forecaster: HierarchicalForecaster,
        horizon: int,
        **forecast_kwargs: Any,
    ) -> dict[ModelKey, ForecastBelief]:
        """Extract :class:`ForecastBelief` objects from a fitted forecaster."""
        results = forecaster.forecast(horizon, **forecast_kwargs)
        return {
            key: ForecastBelief(samples=jnp.array(samples), trust=1.0)
            for key, samples in results.items()
        }

    # -- harmonize ------------------------------------------------------------

    def harmonize(
        self,
        beliefs: dict[ModelKey, ForecastBelief],
        *,
        method: Literal["mcmc", "analytical"] = "mcmc",
        num_samples: int = 1000,
        lambda_hierarchy: float = 1.0,
        lambda_price: float = 1.0,
        lambda_elasticity: float = 1.0,
        mcmc_kwargs: dict[str, Any] | None = None,
        rng_seed: int = 42,
    ) -> HarmonizedResult:
        """Reconcile *beliefs* subject to the configured constraints.

        Parameters
        ----------
        beliefs
            One :class:`ForecastBelief` per series to reconcile.
        method
            ``"mcmc"`` (full NumPyro NUTS) or ``"analytical"``
            (closed-form MinT/WLS projection — only for linear+Gaussian).
        num_samples
            Number of posterior samples to draw / retain.
        lambda_hierarchy, lambda_price, lambda_elasticity
            Weights for the respective constraint families.
        mcmc_kwargs
            Extra keyword arguments forwarded to ``numpyro.infer.MCMC``
            (e.g. ``num_warmup``, ``num_chains``).
        rng_seed
            JAX PRNG seed.
        """
        if method == "analytical":
            return self._harmonize_analytical(
                beliefs, num_samples=num_samples, rng_seed=rng_seed,
                lambda_hierarchy=lambda_hierarchy,
            )
        return self._harmonize_mcmc(
            beliefs,
            num_samples=num_samples,
            lambda_hierarchy=lambda_hierarchy,
            lambda_price=lambda_price,
            lambda_elasticity=lambda_elasticity,
            mcmc_kwargs=mcmc_kwargs or {},
            rng_seed=rng_seed,
        )

    # -- MCMC path ------------------------------------------------------------

    def _harmonize_mcmc(
        self,
        beliefs: dict[ModelKey, ForecastBelief],
        *,
        num_samples: int,
        lambda_hierarchy: float,
        lambda_price: float,
        lambda_elasticity: float,
        mcmc_kwargs: dict[str, Any],
        rng_seed: int,
    ) -> HarmonizedResult:
        rng = jax.random.PRNGKey(rng_seed)

        # 1. draw samples from each belief to get empirical mu/std
        ordered_keys = sorted(beliefs.keys())
        belief_stats: dict[ModelKey, tuple[jnp.ndarray, jnp.ndarray]] = {}
        for key in ordered_keys:
            rng, sub = jax.random.split(rng)
            draws = beliefs[key].draw(num_samples, sub)  # (S, T)
            mu = jnp.mean(draws, axis=0)
            std = jnp.std(draws, axis=0)
            std = jnp.maximum(std, 1e-6)
            trust = beliefs[key].trust
            belief_stats[key] = (mu, std / trust)

        T = belief_stats[ordered_keys[0]][0].shape[0]

        # 2. build numpyro model
        def _lambda_for(c: Constraint) -> float:
            if isinstance(c, AdditiveConstraint):
                return lambda_hierarchy
            if isinstance(c, PriceConstraint):
                return lambda_price
            if isinstance(c, ElasticityConstraint):
                return lambda_elasticity
            return 1.0

        def model() -> None:
            factors: dict[ModelKey, Any] = {}
            # Non-centered parameterization: sample unit-normal deltas,
            # then transform to original scale.  This ensures all latent
            # variables live on comparable scales, which dramatically
            # improves NUTS geometry when series span different magnitudes
            # (e.g. ASP ~50, Qty ~1000, Revenue ~50000).
            for key in ordered_keys:
                mu, std = belief_stats[key]
                z = numpyro.sample(
                    f"z_{key}",
                    dist.Normal(jnp.zeros_like(mu), 1.0).to_event(1),
                )
                factors[key] = numpyro.deterministic(f"f_{key}", mu + std * z)

            # elasticity parameters
            elasticities: dict[int, Any] = {}
            for i, c in enumerate(self.constraints):
                if isinstance(c, ElasticityConstraint):
                    elasticities[i] = numpyro.sample(
                        f"elasticity_{i}",
                        dist.Normal(c.elasticity_mean, c.elasticity_std),
                    )

            # potentials
            for i, c in enumerate(self.constraints):
                lam = _lambda_for(c)
                if isinstance(c, ElasticityConstraint):
                    pot = c.apply_potential(factors, lam, elasticity=elasticities[i])
                else:
                    pot = c.apply_potential(factors, lam)
                numpyro.factor(f"constraint_{i}", pot)

        # 3. run NUTS
        default_mcmc = {"num_warmup": 500, "num_chains": 1}
        default_mcmc.update(mcmc_kwargs)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_samples=num_samples, **default_mcmc)
        rng, sub = jax.random.split(rng)
        mcmc.run(sub)
        posterior = mcmc.get_samples()

        # 4. extract results
        result_samples: dict[ModelKey, np.ndarray] = {}
        for key in ordered_keys:
            result_samples[key] = np.asarray(posterior[f"f_{key}"])

        violations = self._compute_violations(result_samples)
        return HarmonizedResult(samples=result_samples, constraint_violations=violations)

    # -- Analytical path ------------------------------------------------------

    def _harmonize_analytical(
        self,
        beliefs: dict[ModelKey, ForecastBelief],
        *,
        num_samples: int,
        rng_seed: int,
        lambda_hierarchy: float,
    ) -> HarmonizedResult:
        # check all constraints are linear
        nonlinear = [c for c in self.constraints if not c.is_linear()]
        if nonlinear:
            warnings.warn(
                f"Analytical method requires all-linear constraints but found "
                f"{len(nonlinear)} nonlinear constraint(s). Falling back to MCMC.",
                stacklevel=2,
            )
            return self._harmonize_mcmc(
                beliefs,
                num_samples=num_samples,
                lambda_hierarchy=lambda_hierarchy,
                lambda_price=1.0,
                lambda_elasticity=1.0,
                mcmc_kwargs={},
                rng_seed=rng_seed,
            )

        if not self.constraints:
            # nothing to reconcile
            rng = jax.random.PRNGKey(rng_seed)
            result_samples: dict[ModelKey, np.ndarray] = {}
            ordered_keys = sorted(beliefs.keys())
            for key in ordered_keys:
                rng, sub = jax.random.split(rng)
                result_samples[key] = np.asarray(beliefs[key].draw(num_samples, sub))
            return HarmonizedResult(samples=result_samples)

        rng = jax.random.PRNGKey(rng_seed)
        ordered_keys = sorted(beliefs.keys())
        N = len(ordered_keys)

        # draw samples from each belief → (S, T)
        all_draws: dict[ModelKey, jnp.ndarray] = {}
        for key in ordered_keys:
            rng, sub = jax.random.split(rng)
            all_draws[key] = beliefs[key].draw(num_samples, sub)

        T = all_draws[ordered_keys[0]].shape[1]

        # build constraint matrix A: (num_constraints, N)
        A_rows = []
        for c in self.constraints:
            row = c.linear_coefficients(ordered_keys)
            if row is None:
                raise RuntimeError("linear_coefficients returned None for linear constraint")
            A_rows.append(row)
        A = jnp.stack(A_rows)  # (C, N)

        # compute per-series variance (diagonal covariance, WLS-style)
        # use empirical variance across samples, averaged over time
        variances = []
        for key in ordered_keys:
            trust = beliefs[key].trust
            v = jnp.var(all_draws[key], axis=0).mean()
            variances.append(v / (trust**2))
        W_diag = jnp.array(variances)  # (N,)
        W = jnp.diag(W_diag)  # (N, N)

        # MinT projection: x_rec = x - W A' (A W A')^{-1} (A x)
        AWt = W @ A.T  # (N, C)
        AWAt = A @ AWt  # (C, C)
        AWAt_inv = jnp.linalg.inv(AWAt + 1e-8 * jnp.eye(AWAt.shape[0]))
        P = AWt @ AWAt_inv  # (N, C)

        # reconcile each sample, each timestep
        result_samples = {}
        # stack all draws: (S, T, N)
        X = jnp.stack([all_draws[k] for k in ordered_keys], axis=-1)

        # A x for each sample and timestep: (S, T, C)
        Ax = jnp.einsum("cn,stn->stc", A, X)
        # correction: (S, T, N)
        correction = jnp.einsum("nc,stc->stn", P, Ax)
        X_rec = X - correction

        for i, key in enumerate(ordered_keys):
            result_samples[key] = np.asarray(X_rec[:, :, i])

        violations = self._compute_violations(result_samples)
        return HarmonizedResult(samples=result_samples, constraint_violations=violations)

    # -- helpers --------------------------------------------------------------

    def _compute_violations(
        self, samples: dict[ModelKey, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Compute mean absolute residual per constraint over samples."""
        violations: dict[str, np.ndarray] = {}
        for i, c in enumerate(self.constraints):
            name = f"constraint_{i}_{type(c).__name__}"
            try:
                if isinstance(c, AdditiveConstraint):
                    parent = samples[c.parent].mean(axis=0)
                    child_sum = sum(samples[ch].mean(axis=0) for ch in c.children)
                    violations[name] = np.abs(parent - child_sum)
                elif isinstance(c, PriceConstraint):
                    asp = samples[c.asp_key].mean(axis=0)
                    qty = samples[c.qty_key].mean(axis=0)
                    dol = samples[c.dollar_key].mean(axis=0)
                    violations[name] = np.abs(asp * qty - dol)
                elif isinstance(c, ElasticityConstraint):
                    asp = samples[c.asp_key].mean(axis=0)
                    qty = samples[c.qty_key].mean(axis=0)
                    log_asp = np.log(np.maximum(asp, 1e-8))
                    log_qty = np.log(np.maximum(qty, 1e-8))
                    intercept = log_qty.mean() - c.elasticity_mean * log_asp.mean()
                    violations[name] = np.abs(
                        log_qty - (intercept + c.elasticity_mean * log_asp)
                    )
            except KeyError:
                pass
        return violations
