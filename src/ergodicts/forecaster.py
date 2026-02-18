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
from numpyro.infer import MCMC, NUTS, Predictive

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

    All internal series are z-score standardised.  Use
    :meth:`unstandardize` to map back to the original scale.
    """

    y: dict[ModelKey, jnp.ndarray]
    x: dict[ExternalNode, jnp.ndarray]
    T: int
    leaf_nodes: list[ModelKey]
    root_nodes: list[ModelKey]
    children_map: dict[ModelKey, list[ModelKey]]
    predictor_edges: dict[ModelKey, list[EdgeSpec]]
    y_mean: dict[ModelKey, float] = field(default_factory=dict)
    y_std: dict[ModelKey, float] = field(default_factory=dict)

    def unstandardize(
        self, key: ModelKey, values: np.ndarray,
    ) -> np.ndarray:
        """Map standardised *values* back to the original scale."""
        return values * self.y_std[key] + self.y_mean[key]


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

    children_map: dict[ModelKey, list[ModelKey]] = {}
    for parent in root_nodes:
        children_map[parent] = sorted(hierarchy.children_of(parent), key=str)

    # --- Validate y_data -------------------------------------------------
    required = set(leaf_nodes) | set(root_nodes)
    missing = required - set(y_data)
    if missing:
        raise ValueError(f"y_data is missing series for: {missing}")

    # --- Common length ---------------------------------------------------
    lengths = {k: len(v) for k, v in y_data.items()}
    T = min(lengths.values())

    # --- Standardise y ---------------------------------------------------
    y: dict[ModelKey, jnp.ndarray] = {}
    y_mean: dict[ModelKey, float] = {}
    y_std: dict[ModelKey, float] = {}

    for key in list(leaf_nodes) + list(root_nodes):
        arr = np.asarray(y_data[key][:T], dtype=np.float64)
        m, s = float(np.nanmean(arr)), float(np.nanstd(arr))
        if s < 1e-8:
            s = 1.0
        y[key] = jnp.array((arr - m) / s, dtype=jnp.float32)
        y_mean[key] = m
        y_std[key] = s

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

    return ForecastData(
        y=y,
        x=x,
        T=T,
        leaf_nodes=leaf_nodes,
        root_nodes=root_nodes,
        children_map=children_map,
        predictor_edges=predictor_edges,
        y_mean=y_mean,
        y_std=y_std,
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


def _forecaster_model(
    data: ForecastData,
    node_configs: dict[ModelKey, NodeConfig],
    reconciliation: str,
    horizon: int = 0,
    x_future: dict[ExternalNode, jnp.ndarray] | None = None,
) -> None:
    """NumPyro structural time-series model (component-based, bottom-up hierarchy).

    This is a module-level function (not a method) to avoid JAX tracer
    issues with closures over ``self``.
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

    # --- Per-node component loop -----------------------------------------
    all_mus_list: list[jnp.ndarray] = []  # (total_T,) per node
    obs_sigmas: dict[ModelKey, Any] = {}

    for i, k in enumerate(leaf_nodes):
        cfg = node_configs.get(k, NodeConfig())
        node_key = str(k)
        components = resolve_components(cfg)
        agg = resolve_aggregator(cfg)

        obs_sigmas[k] = numpyro.sample(
            f"obs_sigma_{k}", dist.HalfNormal(1.0),
        )

        contributions: dict[str, list[jnp.ndarray]] = {}

        for comp in components:
            role = component_role(comp)
            if isinstance(comp, TrendComponent):
                # Sample params and innovations
                params = comp.sample_params(node_key)
                innovations_hist = comp.sample_innovations(node_key, T)

                if horizon > 0:
                    innovations_fut = comp.sample_innovations(f"{node_key}_fut", horizon)
                    # Concatenate historical + future innovations
                    all_innovations: dict[str, jnp.ndarray] = {}
                    for inn_key in innovations_hist:
                        all_innovations[inn_key] = jnp.concatenate([
                            innovations_hist[inn_key],
                            innovations_fut[inn_key],
                        ])
                else:
                    all_innovations = innovations_hist

                # Run scan
                init = comp.init_state(params)

                def _make_scan_fn(component, parameters):
                    """Create a scan function closed over component and params."""
                    def scan_fn(carry, t_idx):
                        # Slice innovations for this time step
                        inn_t = {
                            inn_key: all_inn[t_idx]
                            for inn_key, all_inn in all_innovations.items()
                        }
                        new_carry, level_t = component.transition_fn(carry, inn_t, parameters)
                        return new_carry, level_t
                    return scan_fn

                scan_fn = _make_scan_fn(comp, params)
                final_carry, levels = jax.lax.scan(
                    scan_fn, init, jnp.arange(total_T),
                )
                numpyro.deterministic(f"final_state_{node_key}", final_carry)
                contributions.setdefault(role, []).append(levels)

            elif isinstance(comp, SeasonalityComponent):
                params = comp.sample_params(node_key)
                seasonal = comp.contribute(params, total_T)
                contributions.setdefault(role, []).append(seasonal)

        # Regression (from DAG edges)
        edges = data.predictor_edges.get(k, [])
        if edges:
            reg_comp = ExternalRegression()
            X_node = _build_node_predictors(edges, x_full, total_T)
            reg_params = reg_comp.sample_params(node_key, len(edges))
            reg_contrib = reg_comp.contribute(reg_params, X_node)
            contributions.setdefault("regression", []).append(reg_contrib)

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

    # -- public API --------------------------------------------------------

    def fit(
        self,
        y_data: dict[ModelKey, np.ndarray],
        x_data: dict[ExternalNode, np.ndarray] | None = None,
        *,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        rng_seed: int = 0,
    ) -> HierarchicalForecaster:
        """Run NUTS inference.

        Returns *self* for method chaining.
        """
        x_data = x_data or {}
        self._data = prepare_data(
            self.hierarchy, self.causal_dag, y_data, x_data,
        )

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

        rng_key = jr.PRNGKey(rng_seed)
        self._mcmc.run(
            rng_key,
            data=self._data,
            node_configs=self.node_configs,
            reconciliation=self.reconciliation,
            horizon=0,
            x_future=None,
        )
        self._samples = self._mcmc.get_samples()
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

    def summary(self) -> None:
        """Print MCMC diagnostics."""
        if self._mcmc is None:
            raise RuntimeError("Must call .fit() first")
        self._mcmc.print_summary()
