"""Hierarchical Bayesian structural time-series forecaster (NumPyro)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from ergodicts.causal_dag import CausalDAG, EdgeSpec, ExternalNode, NodeConfig
from ergodicts.components import (
    Component,
    ExternalRegression,
    RegressionComponent,
    SeasonalityComponent,
    TrendComponent,
    component_role,
    resolve_aggregator,
    resolve_components,
)
from ergodicts.reducer import DependencyGraph, ModelKey


# ---------------------------------------------------------------------------
# ForecastData — pre-processed arrays for the NumPyro model
# ---------------------------------------------------------------------------


@dataclass
class ForecastData:
    """Pre-processed arrays ready for the NumPyro model.

    Internal series are ratio-scaled (divided by the median) so that
    values are centred around 1.0 and remain strictly positive.  This
    preserves the multiplicative structure of the data and is natural
    for revenue / sales series.  Use :meth:`unstandardize` to map back
    to the original scale.

    External predictors are z-score standardised (they may be negative).
    """

    y: dict[ModelKey, jnp.ndarray]
    x: dict[ExternalNode, jnp.ndarray]
    T: int
    leaf_nodes: list[ModelKey]
    root_nodes: list[ModelKey]
    children_map: dict[ModelKey, list[ModelKey]]
    predictor_edges: dict[ModelKey, list[EdgeSpec]]
    y_median: dict[ModelKey, float] = field(default_factory=dict)
    time_index: np.ndarray | None = None
    """Optional array of ISO date strings or datetime64 values, length ``T``."""

    def unstandardize(
        self, key: ModelKey, values: np.ndarray,
    ) -> np.ndarray:
        """Map ratio-scaled *values* back to the original scale."""
        return values * self.y_median[key]


# ---------------------------------------------------------------------------
# Decomposition — per-component forecast contributions
# ---------------------------------------------------------------------------


@dataclass
class Decomposition:
    """Per-component contributions alongside forecasts.

    Attributes
    ----------
    contributions
        ``{node_key: {"trend": (S, H), "seasonality_fourier": (S, H), ...}}``
    regression_coefficients
        ``{node_key: {"predictor_name": (S,)}}`` — full posterior over betas.
    aggregator_type
        ``{node_key: "additive"}`` — tells UI how components combine.
    """

    contributions: dict[ModelKey, dict[str, np.ndarray]]
    regression_coefficients: dict[ModelKey, dict[str, np.ndarray]]
    aggregator_type: dict[ModelKey, str]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _forward_fill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values in a 1-D array."""
    out = arr.copy()
    mask = np.isnan(out)
    if mask.all():
        return np.zeros_like(out)
    # Fill leading NaN with first valid value
    first_valid = np.where(~mask)[0][0]
    out[:first_valid] = out[first_valid]
    # Forward fill remaining
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    return out


def prepare_data(
    hierarchy: DependencyGraph,
    causal_dag: CausalDAG,
    y_data: dict[ModelKey, np.ndarray],
    x_data: dict[ExternalNode, np.ndarray],
    time_index: np.ndarray | None = None,
) -> ForecastData:
    """Validate and pre-process raw data into JAX arrays.

    Parameters
    ----------
    hierarchy
        Parent → children aggregation graph.
    causal_dag
        Predictive / causal edges.
    y_data
        Internal series, shape ``(T,)`` each.  Must include all nodes
        that appear in the hierarchy.
    x_data
        External predictor series, shape ``(T,)`` each.

    Returns
    -------
    ForecastData
    """
    # --- Determine hierarchy structure -----------------------------------
    all_parents = hierarchy.all_parents
    all_children = hierarchy.all_children
    all_internal = all_parents | all_children

    # Roots = parents that are not children of anything
    root_nodes = sorted(all_parents - all_children, key=str)
    # Leaves = children that are not parents of anything
    leaf_nodes = sorted(all_children - all_parents, key=str)

    # If hierarchy is empty, treat all y_data keys as leaves
    if not leaf_nodes:
        leaf_nodes = sorted(y_data.keys(), key=str)
        root_nodes = []

    # Build children_map as transitive leaf descendants so that
    # root = sum(leaves) works for hierarchies of any depth.
    leaf_set = set(leaf_nodes)

    def _descendant_leaves(node: ModelKey) -> set[ModelKey]:
        direct = hierarchy.children_of(node)
        if not direct:
            return {node} if node in leaf_set else set()
        result: set[ModelKey] = set()
        for child in direct:
            result |= _descendant_leaves(child)
        return result

    children_map: dict[ModelKey, list[ModelKey]] = {}
    for parent in root_nodes:
        children_map[parent] = sorted(_descendant_leaves(parent), key=str)

    # --- Validate y_data -------------------------------------------------
    required = set(leaf_nodes) | set(root_nodes)
    missing = required - set(y_data)
    if missing:
        raise ValueError(f"y_data is missing series for: {missing}")

    # --- Common length ---------------------------------------------------
    lengths = {k: len(v) for k, v in y_data.items()}
    T = min(lengths.values())

    # --- Ratio-scale y (divide by median) --------------------------------
    y: dict[ModelKey, jnp.ndarray] = {}
    y_median: dict[ModelKey, float] = {}

    for key in list(leaf_nodes) + list(root_nodes):
        arr = np.asarray(y_data[key][:T], dtype=np.float64)
        med = float(np.nanmedian(arr))
        if abs(med) < 1e-8:
            med = 1.0
        y[key] = jnp.array(arr / med, dtype=jnp.float32)
        y_median[key] = med

    # --- Forward-fill and standardise x ----------------------------------
    x: dict[ExternalNode, jnp.ndarray] = {}
    for ext, arr in x_data.items():
        filled = _forward_fill(np.asarray(arr[:T], dtype=np.float64))
        m, s = float(np.mean(filled)), float(np.std(filled))
        if s < 1e-8:
            s = 1.0
        x[ext] = jnp.array((filled - m) / s, dtype=jnp.float32)

    # --- Predictor edges per leaf node -----------------------------------
    predictor_edges: dict[ModelKey, list[EdgeSpec]] = {}
    for node in leaf_nodes:
        predictor_edges[node] = causal_dag.parents_of(node)

    # --- Slice time_index to match T ------------------------------------
    ti = None
    if time_index is not None:
        ti_arr = np.asarray(time_index)
        if len(ti_arr) >= T:
            ti = ti_arr[:T]
        else:
            ti = ti_arr  # shorter than T — keep as-is

    return ForecastData(
        y=y,
        x=x,
        T=T,
        leaf_nodes=leaf_nodes,
        root_nodes=root_nodes,
        children_map=children_map,
        predictor_edges=predictor_edges,
        y_median=y_median,
        time_index=ti,
    )


# ---------------------------------------------------------------------------
# Helper: per-node predictor matrix
# ---------------------------------------------------------------------------


def _build_node_predictors(
    edges: list[EdgeSpec],
    x_full: dict[ExternalNode, jnp.ndarray],
    total_T: int,
) -> jnp.ndarray:
    """Build predictor matrix for a single node → ``(total_T, n_edges)``.

    Applies lag shifting per :class:`EdgeSpec`.
    """
    if not edges:
        return jnp.zeros((total_T, 0))

    cols: list[jnp.ndarray] = []
    for edge in edges:
        src = edge.source
        if not isinstance(src, ExternalNode):
            cols.append(jnp.zeros(total_T))
            continue
        x_vals = x_full[src]
        lag = edge.lag
        if lag > 0:
            padded = jnp.concatenate([
                jnp.full(lag, x_vals[0]),
                x_vals[: total_T - lag],
            ])
        else:
            padded = x_vals[:total_T]
        cols.append(padded)
    return jnp.stack(cols, axis=-1)  # (total_T, n_edges)


# ---------------------------------------------------------------------------
# NumPyro model (component-based)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Batched scan helpers — vectorize trend scans across nodes of the same type
# ---------------------------------------------------------------------------


def _llt_transition(carry, inn_t, level_sigma, slope_sigma):
    """Single-step LocalLinearTrend transition (vmappable)."""
    level, slope = carry[0], carry[1]
    new_level = level + slope + level_sigma * inn_t[0]
    new_slope = slope + slope_sigma * inn_t[1]
    return jnp.array([new_level, new_slope]), new_level


def _dllt_transition(carry, inn_t, level_sigma, slope_sigma, phi):
    """Single-step DampedLocalLinearTrend transition (vmappable)."""
    level, slope = carry[0], carry[1]
    new_level = level + slope + level_sigma * inn_t[0]
    new_slope = phi * slope + slope_sigma * inn_t[1]
    return jnp.array([new_level, new_slope]), new_level


def _ou_transition(carry, inn_t, theta, mu, sigma):
    """Single-step OUMeanReversion transition (vmappable)."""
    level = carry[0]
    new_level = level + theta * (mu - level) + sigma * inn_t[0]
    return jnp.array([new_level]), new_level


def _batched_llt_scan(inits, innovations, level_sigmas, slope_sigmas, total_T):
    """Run LLT scan batched over N nodes via vmap.

    Parameters: inits (N, 2), innovations (N, total_T, 2),
    level_sigmas (N,), slope_sigmas (N,).

    Returns: final_carries (N, 2), all_levels (N, total_T).
    """
    def _single_scan(init, inns, lsig, ssig):
        def step(carry, inn_t):
            return _llt_transition(carry, inn_t, lsig, ssig)
        return jax.lax.scan(step, init, inns)

    return jax.vmap(_single_scan)(inits, innovations, level_sigmas, slope_sigmas)


def _batched_dllt_scan(inits, innovations, level_sigmas, slope_sigmas, phis, total_T):
    """Run DampedLLT scan batched over N nodes via vmap."""
    def _single_scan(init, inns, lsig, ssig, phi):
        def step(carry, inn_t):
            return _dllt_transition(carry, inn_t, lsig, ssig, phi)
        return jax.lax.scan(step, init, inns)

    return jax.vmap(_single_scan)(inits, innovations, level_sigmas, slope_sigmas, phis)


def _batched_ou_scan(inits, innovations, thetas, mus, sigmas, total_T):
    """Run OU scan batched over N nodes via vmap."""
    def _single_scan(init, inns, theta, mu, sigma):
        def step(carry, inn_t):
            return _ou_transition(carry, inn_t, theta, mu, sigma)
        return jax.lax.scan(step, init, inns)

    return jax.vmap(_single_scan)(inits, innovations, thetas, mus, sigmas)


# ---------------------------------------------------------------------------
# NumPyro model (vectorized component-based)
# ---------------------------------------------------------------------------


def _forecaster_model(
    data: ForecastData,
    node_configs: dict[ModelKey, NodeConfig],
    reconciliation: str,
    horizon: int = 0,
    x_future: dict[ExternalNode, jnp.ndarray] | None = None,
) -> None:
    """NumPyro structural time-series model (vectorized, bottom-up hierarchy).

    Nodes with the same trend component type are batched into a single
    ``jax.vmap``-ed ``jax.lax.scan``, dramatically reducing XLA
    compilation time for large hierarchies.
    """
    T = data.T
    total_T = T + horizon
    leaf_nodes = data.leaf_nodes
    root_nodes = data.root_nodes
    N = len(leaf_nodes)

    # --- Extend external predictors for forecast horizon -----------------
    x_full: dict[ExternalNode, jnp.ndarray] = {}
    for ext, x_hist in data.x.items():
        if horizon > 0 and x_future and ext in x_future:
            x_full[ext] = jnp.concatenate([x_hist, x_future[ext]])
        elif horizon > 0:
            ext_sigma = numpyro.sample(
                f"ext_sigma_{ext}", dist.HalfNormal(0.1),
            )
            ext_inn = numpyro.sample(
                f"ext_inn_{ext}", dist.Normal(0, 1).expand([horizon]),
            )
            last_val = x_hist[-1]
            future = last_val + jnp.cumsum(ext_sigma * ext_inn)
            x_full[ext] = jnp.concatenate([x_hist, future])
        else:
            x_full[ext] = x_hist

    # =====================================================================
    # Phase 1: Sample all numpyro sites per-node (Python loop)
    # =====================================================================
    obs_sigmas: dict[ModelKey, Any] = {}

    # Per-node sampled data, grouped by trend component type
    # "llt" -> list of (node_index, node_key, params, innovations)
    trend_groups: dict[str, list[tuple]] = {}
    # Per-node seasonal contributions (computed immediately, no scan needed)
    seasonal_contribs: dict[int, list[jnp.ndarray]] = {}
    # Per-node regression contributions
    regression_contribs: dict[int, jnp.ndarray] = {}
    # Per-node aggregator and component config
    node_aggs: dict[int, Any] = {}

    for i, k in enumerate(leaf_nodes):
        cfg = node_configs.get(k, NodeConfig())
        node_key = str(k)
        components = resolve_components(cfg)
        agg = resolve_aggregator(cfg)
        node_aggs[i] = agg

        obs_sigmas[k] = numpyro.sample(
            f"obs_sigma_{k}", dist.HalfNormal(1.0),
        )

        for comp in components:
            role = component_role(comp)
            if isinstance(comp, TrendComponent):
                # Sample params and innovations
                params = comp.sample_params(node_key)
                innovations_hist = comp.sample_innovations(node_key, T)

                if horizon > 0:
                    innovations_fut = comp.sample_innovations(f"{node_key}_fut", horizon)
                    all_innovations: dict[str, jnp.ndarray] = {}
                    for inn_key in innovations_hist:
                        all_innovations[inn_key] = jnp.concatenate([
                            innovations_hist[inn_key],
                            innovations_fut[inn_key],
                        ])
                else:
                    all_innovations = innovations_hist

                init = comp.init_state(params)
                trend_type = comp.component_name  # "local_linear_trend", "damped_local_linear_trend", "ou_mean_reversion"
                trend_groups.setdefault(trend_type, []).append(
                    (i, node_key, params, all_innovations, init)
                )

            elif isinstance(comp, SeasonalityComponent):
                params = comp.sample_params(node_key)
                seasonal = comp.contribute(params, total_T)
                seasonal_contribs.setdefault(i, []).append(seasonal)

        # Regression (from DAG edges)
        edges = data.predictor_edges.get(k, [])
        if edges:
            reg_comp = ExternalRegression()
            X_node = _build_node_predictors(edges, x_full, total_T)
            reg_params = reg_comp.sample_params(node_key, len(edges))
            reg_contrib = reg_comp.contribute(reg_params, X_node)
            regression_contribs[i] = reg_contrib

    # =====================================================================
    # Phase 2: Batched trend scans (one vmap per trend type)
    # =====================================================================
    # Maps node_index -> (final_carry, levels)
    trend_results: dict[int, tuple[jnp.ndarray, jnp.ndarray]] = {}

    for trend_type, group in trend_groups.items():
        indices = [g[0] for g in group]
        node_keys = [g[1] for g in group]

        if trend_type == "local_linear_trend":
            # Stack params
            level_sigmas = jnp.stack([g[2]["level_sigma"] for g in group])
            slope_sigmas = jnp.stack([g[2]["slope_sigma"] for g in group])
            # Stack innovations: (N, total_T, 2) — [lev_inn, slp_inn]
            inns = jnp.stack([
                jnp.stack([g[3]["lev_inn"], g[3]["slp_inn"]], axis=-1)
                for g in group
            ])
            inits = jnp.stack([g[4] for g in group])

            final_carries, all_levels = _batched_llt_scan(
                inits, inns, level_sigmas, slope_sigmas, total_T,
            )
            for j, idx in enumerate(indices):
                trend_results[idx] = (final_carries[j], all_levels[j])

        elif trend_type == "damped_local_linear_trend":
            level_sigmas = jnp.stack([g[2]["level_sigma"] for g in group])
            slope_sigmas = jnp.stack([g[2]["slope_sigma"] for g in group])
            phis = jnp.stack([g[2]["phi"] for g in group])
            inns = jnp.stack([
                jnp.stack([g[3]["lev_inn"], g[3]["slp_inn"]], axis=-1)
                for g in group
            ])
            inits = jnp.stack([g[4] for g in group])

            final_carries, all_levels = _batched_dllt_scan(
                inits, inns, level_sigmas, slope_sigmas, phis, total_T,
            )
            for j, idx in enumerate(indices):
                trend_results[idx] = (final_carries[j], all_levels[j])

        elif trend_type == "ou_mean_reversion":
            thetas = jnp.stack([g[2]["theta"] for g in group])
            mus = jnp.stack([g[2]["mu"] for g in group])
            sigmas = jnp.stack([g[2]["sigma"] for g in group])
            inns = jnp.stack([
                g[3]["inn"][:, None]  # (total_T, 1)
                for g in group
            ])
            inits = jnp.stack([g[4] for g in group])

            final_carries, all_levels = _batched_ou_scan(
                inits, inns, thetas, mus, sigmas, total_T,
            )
            for j, idx in enumerate(indices):
                trend_results[idx] = (final_carries[j], all_levels[j])

        else:
            # Fallback: per-node scan for unknown trend types
            for g in group:
                idx, node_key, params, all_innovations, init = g
                comp_instance = resolve_components(
                    node_configs.get(leaf_nodes[idx], NodeConfig())
                )[0]

                def _make_scan_fn(component, parameters, innovations):
                    def scan_fn(carry, t_idx):
                        inn_t = {
                            inn_key: all_inn[t_idx]
                            for inn_key, all_inn in innovations.items()
                        }
                        new_carry, level_t = component.transition_fn(carry, inn_t, parameters)
                        return new_carry, level_t
                    return scan_fn

                scan_fn = _make_scan_fn(comp_instance, params, all_innovations)
                final_carry, levels = jax.lax.scan(
                    scan_fn, init, jnp.arange(total_T),
                )
                trend_results[idx] = (final_carry, levels)

        # Register deterministic final_state sites
        for g in group:
            idx, node_key = g[0], g[1]
            numpyro.deterministic(f"final_state_{node_key}", trend_results[idx][0])

    # =====================================================================
    # Phase 3: Aggregate contributions per node
    # =====================================================================
    all_mus_list: list[jnp.ndarray] = []

    for i, k in enumerate(leaf_nodes):
        agg = node_aggs[i]
        contributions: dict[str, list[jnp.ndarray]] = {}

        # Trend
        if i in trend_results:
            contributions.setdefault("trend", []).append(trend_results[i][1])

        # Seasonality
        if i in seasonal_contribs:
            contributions["seasonality"] = seasonal_contribs[i]

        # Regression
        if i in regression_contribs:
            contributions.setdefault("regression", []).append(regression_contribs[i])

        mu_node = agg.aggregate(contributions, (total_T,))
        all_mus_list.append(mu_node)

    # Stack all node means: (total_T, N)
    all_mus = jnp.stack(all_mus_list, axis=-1)

    # --- Root = sum of leaves (for bottom_up and soft) --------------------
    has_root = reconciliation in ("bottom_up", "soft") and root_nodes
    if has_root:
        total_mu = jnp.sum(all_mus, axis=-1)  # (total_T,)

    # --- Observations (historical only) ----------------------------------
    for i, k in enumerate(leaf_nodes):
        numpyro.sample(
            f"y_{k}",
            dist.Normal(all_mus[:T, i], obs_sigmas[k]),
            obs=data.y[k],
        )

    if has_root:
        for root in root_nodes:
            if root in data.y:
                root_sigma = numpyro.sample(
                    f"obs_sigma_{root}", dist.HalfNormal(1.0),
                )
                numpyro.sample(
                    f"y_{root}",
                    dist.Normal(total_mu[:T], root_sigma),
                    obs=data.y[root],
                )

    # --- Deterministic forecast outputs ----------------------------------
    for i, k in enumerate(leaf_nodes):
        numpyro.deterministic(f"mu_{k}", all_mus[:, i])
    if has_root:
        for root in root_nodes:
            numpyro.deterministic(f"mu_{root}", total_mu)


# ---------------------------------------------------------------------------
# HierarchicalForecaster
# ---------------------------------------------------------------------------


class HierarchicalForecaster:
    """Hierarchical Bayesian structural time-series forecaster.

    Parameters
    ----------
    hierarchy : DependencyGraph
        Parent → children aggregation graph (from the reducer).
    causal_dag : CausalDAG
        Predictive / causal edges between nodes.
    node_configs : dict[ModelKey, NodeConfig]
        Per-node configuration.
    reconciliation : str
        ``"bottom_up"`` (sum leaves deterministically),
        ``"soft"`` (Gaussian potential), or ``"none"``.

    Examples
    --------
    ```python
    model = HierarchicalForecaster(hierarchy, dag, configs)
    model.fit(y_data, x_data, num_warmup=500, num_samples=1000)
    forecasts = model.forecast(horizon=12)
    ```
    """

    def __init__(
        self,
        hierarchy: DependencyGraph,
        causal_dag: CausalDAG,
        node_configs: dict[ModelKey, NodeConfig] | None = None,
        reconciliation: Literal["bottom_up", "soft", "none"] = "bottom_up",
    ) -> None:
        self.hierarchy = hierarchy
        self.causal_dag = causal_dag
        self.node_configs = node_configs or {}
        self.reconciliation = reconciliation

        self._data: ForecastData | None = None
        self._mcmc: MCMC | None = None
        self._samples: dict[str, jnp.ndarray] | None = None
        self._inference: str | None = None

    # -- public API --------------------------------------------------------

    def fit(
        self,
        y_data: dict[ModelKey, np.ndarray],
        x_data: dict[ExternalNode, np.ndarray] | None = None,
        *,
        inference: Literal["nuts", "svi"] = "nuts",
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        rng_seed: int = 0,
        # SVI-specific parameters
        svi_steps: int = 5000,
        svi_lr: float = 0.005,
        svi_progress_bar: bool = True,
    ) -> HierarchicalForecaster:
        """Run Bayesian inference.

        Parameters
        ----------
        inference
            ``"nuts"`` for full MCMC (accurate but slow) or ``"svi"``
            for stochastic variational inference (fast approximate).
        num_warmup
            NUTS warmup iterations (ignored for SVI).
        num_samples
            Number of posterior samples to draw.
        num_chains
            Number of MCMC chains (ignored for SVI).
        rng_seed
            JAX PRNG seed.
        svi_steps
            Number of SVI optimisation steps (ignored for NUTS).
        svi_lr
            Adam learning rate for SVI (ignored for NUTS).
        svi_progress_bar
            Show progress bar for SVI training.

        Returns *self* for method chaining.
        """
        x_data = x_data or {}
        self._data = prepare_data(
            self.hierarchy, self.causal_dag, y_data, x_data,
        )
        self._inference = inference

        model_kwargs = dict(
            data=self._data,
            node_configs=self.node_configs,
            reconciliation=self.reconciliation,
            horizon=0,
            x_future=None,
        )
        rng_key = jr.PRNGKey(rng_seed)

        if inference == "nuts":
            kernel = NUTS(
                _forecaster_model,
                target_accept_prob=0.8,
                max_tree_depth=10,
            )
            self._mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                progress_bar=True,
            )
            self._mcmc.run(rng_key, **model_kwargs)
            self._samples = self._mcmc.get_samples()

        elif inference == "svi":
            guide = AutoNormal(_forecaster_model)
            optimizer = numpyro.optim.Adam(step_size=svi_lr)
            svi = SVI(_forecaster_model, guide, optimizer, loss=Trace_ELBO())

            svi_result = svi.run(
                rng_key,
                svi_steps,
                progress_bar=svi_progress_bar,
                **model_kwargs,
            )

            # Step 1: draw latent samples from the trained guide
            k1, k2 = jr.split(rng_key)
            guide_predictive = Predictive(
                guide,
                params=svi_result.params,
                num_samples=num_samples,
            )
            latent_samples = guide_predictive(k1, **model_kwargs)

            # Step 2: run model conditioned on latent samples to
            # get deterministic sites (mu_*, final_state_*)
            model_predictive = Predictive(
                _forecaster_model,
                posterior_samples=latent_samples,
            )
            det_samples = model_predictive(k2, **model_kwargs)

            # Merge latent + deterministic into one dict
            self._samples = {**latent_samples, **det_samples}
        else:
            raise ValueError(f"Unknown inference method: {inference!r}")

        return self

    def forecast(
        self,
        horizon: int,
        x_future: dict[ExternalNode, np.ndarray] | None = None,
        *,
        rng_seed: int = 1,
    ) -> dict[ModelKey, np.ndarray]:
        """Generate posterior predictive forecasts.

        Parameters
        ----------
        horizon
            Number of future time steps to forecast.
        x_future
            Future external predictor values, shape ``(horizon,)`` each.
            If omitted, external series are extended via random walk.
        rng_seed
            JAX PRNG seed.

        Returns
        -------
        dict[ModelKey, ndarray]
            Posterior predictive samples of shape
            ``(num_samples, horizon)`` per node, on the **original**
            (unstandardised) scale.
        """
        if self._samples is None or self._data is None:
            raise RuntimeError("Must call .fit() before .forecast()")

        data = self._data
        leaf_nodes = data.leaf_nodes
        N = len(leaf_nodes)
        T = data.T
        samples = self._samples

        # Infer num_samples from obs_sigma (always present)
        num_samp = samples[f"obs_sigma_{leaf_nodes[0]}"].shape[0]

        rng_key = jr.PRNGKey(rng_seed)

        # --- Build future external predictor values ----------------------
        x_fut: dict[ExternalNode, jnp.ndarray] = {}
        if any(data.predictor_edges.get(k, []) for k in leaf_nodes):
            for ext, x_hist in data.x.items():
                if x_future and ext in x_future:
                    x_fut[ext] = jnp.array(
                        x_future[ext][:horizon], dtype=jnp.float32,
                    )
                else:
                    k1, rng_key = jr.split(rng_key)
                    inn = jr.normal(k1, (horizon,))
                    x_fut[ext] = x_hist[-1] + jnp.cumsum(0.1 * inn)

        # --- Per-node forecast -------------------------------------------
        forecast_per_node: list[jnp.ndarray] = []  # (S, horizon) per node

        for i, k in enumerate(leaf_nodes):
            cfg = self.node_configs.get(k, NodeConfig())
            node_key = str(k)
            components = resolve_components(cfg)
            agg = resolve_aggregator(cfg)

            contributions: dict[str, list[jnp.ndarray]] = {}

            for comp in components:
                role = component_role(comp)
                if isinstance(comp, TrendComponent):
                    # Extract posterior params
                    post_params = comp.extract_posterior(node_key, samples)
                    final_state_key = f"final_state_{node_key}"
                    final_state = samples[final_state_key]  # (S, state_dim)

                    # Forecast: vmap over sample dimension
                    k1, rng_key = jr.split(rng_key)
                    rng_keys = jr.split(k1, num_samp)

                    # Build per-sample param dict for vmap
                    trend_levels = jax.vmap(
                        lambda fs, rk, *p_vals: comp.forecast_from_state(
                            fs,
                            dict(zip(post_params.keys(), p_vals)),
                            horizon,
                            rk,
                        ),
                    )(final_state, rng_keys, *post_params.values())
                    contributions.setdefault(role, []).append(trend_levels)

                elif isinstance(comp, SeasonalityComponent):
                    post_params = comp.extract_posterior(node_key, samples)

                    # vmap over sample dimension
                    seasonal = jax.vmap(
                        lambda *p_vals: comp.forecast_contribute(
                            dict(zip(post_params.keys(), p_vals)),
                            T,
                            horizon,
                        ),
                    )(*post_params.values())
                    contributions.setdefault(role, []).append(seasonal)

            # Regression
            edges = data.predictor_edges.get(k, [])
            if edges:
                reg_comp = ExternalRegression()
                reg_post = reg_comp.extract_posterior(node_key, samples)

                # Build future X for this node
                x_cols: list[jnp.ndarray] = []
                for edge in edges:
                    src = edge.source
                    if not isinstance(src, ExternalNode):
                        x_cols.append(jnp.zeros(horizon))
                        continue
                    lag = edge.lag
                    x_hist = data.x[src]
                    if src in x_fut:
                        x_extended = jnp.concatenate([x_hist, x_fut[src]])
                    else:
                        x_extended = x_hist
                    x_fc = x_extended[T - lag: T - lag + horizon]
                    x_cols.append(x_fc)

                X_future_node = jnp.stack(x_cols, axis=-1)  # (horizon, n_edges)
                reg_forecast = jax.vmap(
                    lambda *p_vals: reg_comp.forecast_contribute(
                        dict(zip(reg_post.keys(), p_vals)),
                        X_future_node,
                    ),
                )(*reg_post.values())
                contributions.setdefault("regression", []).append(reg_forecast)

            node_forecast = agg.aggregate(contributions, (num_samp, horizon))
            forecast_per_node.append(node_forecast)

        # --- Stack: (S, horizon, N) and add obs noise --------------------
        forecast_stack = jnp.stack(forecast_per_node, axis=-1)  # (S, horizon, N)

        obs_sigma_arr = jnp.stack(
            [samples[f"obs_sigma_{k}"] for k in leaf_nodes], axis=-1,
        )  # (S, N)

        k3, rng_key = jr.split(rng_key)
        obs_noise = jr.normal(k3, forecast_stack.shape) * obs_sigma_arr[:, None, :]
        forecast_samples = forecast_stack + obs_noise

        # --- Extract per-node and unstandardise ---------------------------
        results: dict[ModelKey, np.ndarray] = {}
        for i, k in enumerate(leaf_nodes):
            raw = np.asarray(forecast_samples[:, :, i])
            results[k] = data.unstandardize(k, raw)

        # Root nodes = sum of leaves (on original scale)
        if self.reconciliation in ("bottom_up", "soft") and data.root_nodes:
            for root in data.root_nodes:
                results[root] = sum(results[k] for k in data.children_map[root])

        return results

    def forecast_decomposed(
        self,
        horizon: int,
        x_future: dict[ExternalNode, np.ndarray] | None = None,
        *,
        rng_seed: int = 1,
    ) -> tuple[dict[ModelKey, np.ndarray], Decomposition]:
        """Generate forecasts with per-component decomposition.

        Returns the same forecasts as :meth:`forecast` plus a
        :class:`Decomposition` capturing each component's contribution.
        """
        if self._samples is None or self._data is None:
            raise RuntimeError("Must call .fit() before .forecast_decomposed()")

        data = self._data
        leaf_nodes = data.leaf_nodes
        T = data.T
        samples = self._samples

        num_samp = samples[f"obs_sigma_{leaf_nodes[0]}"].shape[0]
        rng_key = jr.PRNGKey(rng_seed)

        # Build future external predictors
        x_fut: dict[ExternalNode, jnp.ndarray] = {}
        if any(data.predictor_edges.get(k, []) for k in leaf_nodes):
            for ext, x_hist in data.x.items():
                if x_future and ext in x_future:
                    x_fut[ext] = jnp.array(x_future[ext][:horizon], dtype=jnp.float32)
                else:
                    k1, rng_key = jr.split(rng_key)
                    inn = jr.normal(k1, (horizon,))
                    x_fut[ext] = x_hist[-1] + jnp.cumsum(0.1 * inn)

        decomp_contribs: dict[ModelKey, dict[str, np.ndarray]] = {}
        decomp_reg_coeffs: dict[ModelKey, dict[str, np.ndarray]] = {}
        decomp_agg_type: dict[ModelKey, str] = {}
        forecast_per_node: list[jnp.ndarray] = []

        for i, k in enumerate(leaf_nodes):
            cfg = self.node_configs.get(k, NodeConfig())
            node_key = str(k)
            components = resolve_components(cfg)
            agg = resolve_aggregator(cfg)
            decomp_agg_type[k] = agg.component_name
            node_decomp: dict[str, np.ndarray] = {}

            contributions: dict[str, list[jnp.ndarray]] = {}

            for comp in components:
                role = component_role(comp)
                if isinstance(comp, TrendComponent):
                    post_params = comp.extract_posterior(node_key, samples)
                    final_state_key = f"final_state_{node_key}"
                    final_state = samples[final_state_key]

                    k1, rng_key = jr.split(rng_key)
                    rng_keys = jr.split(k1, num_samp)

                    trend_levels = jax.vmap(
                        lambda fs, rk, *p_vals: comp.forecast_from_state(
                            fs, dict(zip(post_params.keys(), p_vals)), horizon, rk,
                        ),
                    )(final_state, rng_keys, *post_params.values())
                    contributions.setdefault(role, []).append(trend_levels)
                    comp_label = f"trend_{comp.component_name}"
                    node_decomp[comp_label] = np.asarray(trend_levels)

                elif isinstance(comp, SeasonalityComponent):
                    post_params = comp.extract_posterior(node_key, samples)
                    seasonal = jax.vmap(
                        lambda *p_vals: comp.forecast_contribute(
                            dict(zip(post_params.keys(), p_vals)), T, horizon,
                        ),
                    )(*post_params.values())
                    contributions.setdefault(role, []).append(seasonal)
                    comp_label = f"seasonality_{comp.component_name}"
                    node_decomp[comp_label] = np.asarray(seasonal)

            # Regression
            edges = data.predictor_edges.get(k, [])
            if edges:
                reg_comp = ExternalRegression()
                reg_post = reg_comp.extract_posterior(node_key, samples)

                x_cols: list[jnp.ndarray] = []
                edge_names: list[str] = []
                for edge in edges:
                    src = edge.source
                    if not isinstance(src, ExternalNode):
                        x_cols.append(jnp.zeros(horizon))
                        edge_names.append(str(src))
                        continue
                    edge_names.append(src.name)
                    lag = edge.lag
                    x_hist = data.x[src]
                    if src in x_fut:
                        x_extended = jnp.concatenate([x_hist, x_fut[src]])
                    else:
                        x_extended = x_hist
                    x_fc = x_extended[T - lag: T - lag + horizon]
                    x_cols.append(x_fc)

                X_future_node = jnp.stack(x_cols, axis=-1)
                reg_forecast = jax.vmap(
                    lambda *p_vals: reg_comp.forecast_contribute(
                        dict(zip(reg_post.keys(), p_vals)), X_future_node,
                    ),
                )(*reg_post.values())
                contributions.setdefault("regression", []).append(reg_forecast)
                node_decomp["regression_total"] = np.asarray(reg_forecast)

                # Per-predictor attribution: betas[:, j] * X_future[:, j]
                betas = np.asarray(reg_post["betas"])  # (S, n_edges)
                reg_coeffs: dict[str, np.ndarray] = {}
                for j, ename in enumerate(edge_names):
                    # Per-predictor contribution: (S, H)
                    attr = betas[:, j:j+1] * np.asarray(X_future_node[:, j])[None, :]
                    node_decomp[f"regression_{ename}"] = attr
                    reg_coeffs[ename] = betas[:, j]
                decomp_reg_coeffs[k] = reg_coeffs

            decomp_contribs[k] = node_decomp
            node_forecast = agg.aggregate(contributions, (num_samp, horizon))
            forecast_per_node.append(node_forecast)

        # Stack and add obs noise
        forecast_stack = jnp.stack(forecast_per_node, axis=-1)
        obs_sigma_arr = jnp.stack(
            [samples[f"obs_sigma_{k}"] for k in leaf_nodes], axis=-1,
        )
        k3, rng_key = jr.split(rng_key)
        obs_noise = jr.normal(k3, forecast_stack.shape) * obs_sigma_arr[:, None, :]
        forecast_samples = forecast_stack + obs_noise

        results: dict[ModelKey, np.ndarray] = {}
        for i, k in enumerate(leaf_nodes):
            raw = np.asarray(forecast_samples[:, :, i])
            results[k] = data.unstandardize(k, raw)

        if self.reconciliation in ("bottom_up", "soft") and data.root_nodes:
            for root in data.root_nodes:
                results[root] = sum(results[k] for k in data.children_map[root])

        decomposition = Decomposition(
            contributions=decomp_contribs,
            regression_coefficients=decomp_reg_coeffs,
            aggregator_type=decomp_agg_type,
        )
        return results, decomposition

    def summary(self) -> None:
        """Print MCMC diagnostics."""
        if self._mcmc is None:
            raise RuntimeError("Must call .fit() first")
        self._mcmc.print_summary()
