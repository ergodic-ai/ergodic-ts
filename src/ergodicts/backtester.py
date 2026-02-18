"""Backtesting module for hierarchical time-series forecasters."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from ergodicts.causal_dag import CausalDAG, ExternalNode, NodeConfig
from ergodicts.forecaster import HierarchicalForecaster
from ergodicts.reducer import DependencyGraph, ModelKey


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Returns ``NaN`` if any actual value is zero.
    """
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0–200 scale)."""
    denom = np.abs(actual) + np.abs(predicted)
    mask = denom != 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def coverage(
    actual: np.ndarray,
    samples: np.ndarray,
    level: float = 0.90,
) -> float:
    """Empirical coverage of prediction interval.

    Parameters
    ----------
    actual
        Shape ``(horizon,)``.
    samples
        Posterior predictive samples, shape ``(num_samples, horizon)``.
    level
        Nominal coverage level (default 0.90 → 5th/95th percentiles).

    Returns
    -------
    float
        Fraction of actual values inside the interval.
    """
    alpha = (1 - level) / 2
    lo = np.percentile(samples, alpha * 100, axis=0)
    hi = np.percentile(samples, (1 - alpha) * 100, axis=0)
    return float(np.mean((actual >= lo) & (actual <= hi)))


def rolling_mape(actual: np.ndarray, predicted: np.ndarray, window: int) -> float:
    """MAPE of rolling sums over *window*-step windows.

    For each window position, sums both actual and predicted over the window,
    then computes ``|sum(predicted) - sum(actual)| / sum(actual)``.  The final
    result is the mean of these per-window percentage errors.

    Returns ``NaN`` if the series is shorter than *window* or all window
    sums of actuals are zero.
    """
    h = len(actual)
    if h < window:
        return float("nan")
    mapes = []
    for start in range(h - window + 1):
        end = start + window
        sum_a = actual[start:end].sum()
        sum_p = predicted[start:end].sum()
        if sum_a != 0:
            mapes.append(float(abs(sum_p - sum_a) / abs(sum_a)) * 100)
    return float(np.mean(mapes)) if mapes else float("nan")


def accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean directional accuracy, clipped to [0, 1].

    ``accuracy = mean(1 - clip(|y_hat - y| / |y|, 0, 1))``

    A value of 1.0 means perfect point forecasts; 0.0 means every
    forecast is off by 100 % or more.  Stored as a fraction (like
    coverage).

    Returns ``NaN`` if all actuals are zero.
    """
    mask = actual != 0
    if not mask.any():
        return float("nan")
    abs_pct_err = np.abs((predicted[mask] - actual[mask]) / actual[mask])
    return float(np.mean(1 - np.clip(abs_pct_err, 0, 1)))


def rolling_accuracy(actual: np.ndarray, predicted: np.ndarray, window: int) -> float:
    """Accuracy computed on rolling sums over *window*-step windows.

    For each window, sums actual and predicted, then computes
    ``1 - clip(|sum_hat - sum_y| / |sum_y|, 0, 1)``.  The final result
    is the mean across windows.  Stored as a fraction.

    Returns ``NaN`` if the series is shorter than *window*.
    """
    h = len(actual)
    if h < window:
        return float("nan")
    accs = []
    for start in range(h - window + 1):
        end = start + window
        sum_a = actual[start:end].sum()
        sum_p = predicted[start:end].sum()
        if sum_a != 0:
            pct_err = abs(sum_p - sum_a) / abs(sum_a)
            accs.append(1.0 - min(pct_err, 1.0))
    return float(np.mean(accs)) if accs else float("nan")


def crps(actual: np.ndarray, samples: np.ndarray) -> float:
    """Continuous Ranked Probability Score (lower is better).

    Uses the energy form:
    ``CRPS = E|X - y| - 0.5 * E|X - X'|``
    where X, X' are independent draws from the forecast distribution.

    Parameters
    ----------
    actual
        Shape ``(horizon,)``.
    samples
        Shape ``(num_samples, horizon)``.
    """
    # E|X - y|  — average over samples and time
    term1 = np.mean(np.abs(samples - actual[None, :]))
    # E|X - X'| — average over pairs and time
    # For efficiency, use the sorted-samples formula per time step
    n = samples.shape[0]
    scores = []
    for t in range(actual.shape[0]):
        s = np.sort(samples[:, t])
        # E|X - X'| = 2/(n^2) * sum_i (2i - n - 1) * s_i
        weights = 2 * np.arange(1, n + 1) - n - 1
        term2_t = np.sum(weights * s) / (n * n)
        term1_t = np.mean(np.abs(s - actual[t]))
        scores.append(term1_t - term2_t)
    return float(np.mean(scores))


def compute_metrics(
    actual: np.ndarray,
    samples: np.ndarray,
    coverage_level: float = 0.90,
) -> dict[str, float]:
    """Compute all standard forecast metrics.

    Parameters
    ----------
    actual
        Ground truth, shape ``(horizon,)``.
    samples
        Posterior predictive samples, shape ``(num_samples, horizon)``.
    coverage_level
        Nominal coverage for interval evaluation.

    Returns
    -------
    dict
        Keys: ``mae``, ``rmse``, ``mape``, ``smape``, ``coverage``,
        ``crps``.
    """
    median = np.median(samples, axis=0)
    h = len(actual)
    return {
        "mae": mae(actual, median),
        "rmse": rmse(actual, median),
        "mape": mape(actual, median),
        "smape": smape(actual, median),
        "rolling_mape_3": rolling_mape(actual, median, 3),
        "rolling_mape_6": rolling_mape(actual, median, 6),
        "mape_full": rolling_mape(actual, median, h),
        "accuracy": accuracy(actual, median),
        "rolling_accuracy_3": rolling_accuracy(actual, median, 3),
        "rolling_accuracy_6": rolling_accuracy(actual, median, 6),
        "accuracy_full": rolling_accuracy(actual, median, h),
        "coverage": coverage(actual, samples, level=coverage_level),
        "crps": crps(actual, samples),
    }


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _model_key_to_dict(key: ModelKey) -> dict[str, Any]:
    return {"dimensions": list(key.dimensions), "values": list(key.values)}


def _model_key_from_dict(d: dict[str, Any]) -> ModelKey:
    return ModelKey(dimensions=tuple(d["dimensions"]), values=tuple(d["values"]))


def _external_node_to_dict(node: ExternalNode) -> dict[str, Any]:
    return {"name": node.name, "dynamics": node.dynamics, "integrated": node.integrated}


def _external_node_from_dict(d: dict[str, Any]) -> ExternalNode:
    return ExternalNode(name=d["name"], dynamics=d.get("dynamics", "rw"), integrated=d.get("integrated", False))


def _node_key_to_str(key: ModelKey) -> str:
    """Stable string key for numpy archive names (no special chars)."""
    return "__".join(key.dimensions) + "___" + "__".join(key.values)


def _node_key_from_str(s: str) -> ModelKey:
    dims_part, vals_part = s.split("___", 1)
    return ModelKey(dimensions=tuple(dims_part.split("__")), values=tuple(vals_part.split("__")))


def _serialize_hierarchy(hierarchy: DependencyGraph) -> list[dict[str, Any]]:
    edges = []
    for parent in sorted(hierarchy.all_parents, key=str):
        for child in sorted(hierarchy.children_of(parent), key=str):
            edges.append({"parent": _model_key_to_dict(parent), "child": _model_key_to_dict(child)})
    return edges


def _deserialize_hierarchy(edges: list[dict[str, Any]]) -> DependencyGraph:
    g = DependencyGraph()
    for e in edges:
        g.add(_model_key_from_dict(e["parent"]), _model_key_from_dict(e["child"]))
    return g


def _serialize_causal_dag(dag: CausalDAG) -> list[dict[str, Any]]:
    edges = []
    for node in dag.all_nodes:
        for edge in dag.children_of(node):
            src = edge.source
            tgt = edge.target
            edges.append({
                "source": _external_node_to_dict(src) if isinstance(src, ExternalNode) else _model_key_to_dict(src),
                "source_type": "external" if isinstance(src, ExternalNode) else "model_key",
                "target": _model_key_to_dict(tgt) if isinstance(tgt, ModelKey) else _external_node_to_dict(tgt),
                "target_type": "model_key" if isinstance(tgt, ModelKey) else "external",
                "lag": edge.lag,
                "contemporaneous": edge.contemporaneous,
            })
    return edges


def _deserialize_causal_dag(edges: list[dict[str, Any]]) -> CausalDAG:
    dag = CausalDAG()
    for e in edges:
        src = _external_node_from_dict(e["source"]) if e["source_type"] == "external" else _model_key_from_dict(e["source"])
        tgt = _model_key_from_dict(e["target"]) if e["target_type"] == "model_key" else _external_node_from_dict(e["target"])
        dag.add_edge(src, tgt, lag=e["lag"], contemporaneous=e.get("contemporaneous", False))
    return dag


def _serialize_node_configs(configs: dict[ModelKey, NodeConfig]) -> dict[str, Any]:
    return {str(k): v.model_dump() for k, v in configs.items()}


def _deserialize_node_configs(data: dict[str, Any], key_lookup: dict[str, ModelKey]) -> dict[ModelKey, NodeConfig]:
    result = {}
    for k_str, cfg_dict in data.items():
        key = key_lookup.get(k_str)
        if key is not None:
            result[key] = NodeConfig(**cfg_dict)
    return result


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Results from a single backtest fold.

    Attributes
    ----------
    cutoff
        Index of the last training observation.
    horizon
        Number of forecast steps.
    metrics
        Per-node metric dictionaries.
    forecasts
        Per-node posterior predictive samples ``(num_samples, horizon)``.
    actuals
        Per-node ground truth ``(horizon,)``.
    """

    cutoff: int
    horizon: int
    metrics: dict[ModelKey, dict[str, float]]
    forecasts: dict[ModelKey, np.ndarray]
    actuals: dict[ModelKey, np.ndarray]


@dataclass
class BacktestSummary:
    """Aggregated results across all backtest folds.

    Attributes
    ----------
    folds
        Individual :class:`BacktestResult` objects.
    summary_df
        DataFrame with mean metrics across folds, indexed by node.
    run_config
        Configuration dict capturing all parameters needed to reproduce
        this run.  Populated automatically by :meth:`Backtester.run`.
    """

    folds: list[BacktestResult]
    summary_df: pd.DataFrame
    run_config: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        n = len(self.folds)
        nodes = len(self.summary_df)
        name = self.run_config.get("run_name", "")
        tag = f", name={name!r}" if name else ""
        return f"BacktestSummary(folds={n}, nodes={nodes}{tag})"

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save the full backtest to disk.

        Directory layout::

            path/
                meta.json          # run_config, metadata, per-fold metrics
                summary.csv        # summary_df
                folds/
                    fold_000.npz   # forecasts + actuals arrays
                    fold_001.npz
                    ...

        Parameters
        ----------
        path
            Directory to write into.  Created if it doesn't exist.

        Returns
        -------
        Path
            The directory that was written to.
        """
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        folds_dir = root / "folds"
        folds_dir.mkdir(exist_ok=True)

        # -- meta.json: config + per-fold metrics --------------------------
        fold_meta = []
        for i, fold in enumerate(self.folds):
            metrics_ser = {str(k): v for k, v in fold.metrics.items()}
            fold_meta.append({
                "fold_index": i,
                "cutoff": fold.cutoff,
                "horizon": fold.horizon,
                "metrics": metrics_ser,
            })

        meta = {
            "run_config": self.run_config,
            "n_folds": len(self.folds),
            "folds": fold_meta,
        }
        (root / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

        # -- summary.csv ---------------------------------------------------
        self.summary_df.to_csv(root / "summary.csv")

        # -- per-fold arrays -----------------------------------------------
        for i, fold in enumerate(self.folds):
            arrays: dict[str, np.ndarray] = {}
            for key, arr in fold.forecasts.items():
                arrays[f"forecast__{_node_key_to_str(key)}"] = arr
            for key, arr in fold.actuals.items():
                arrays[f"actual__{_node_key_to_str(key)}"] = arr
            np.savez_compressed(folds_dir / f"fold_{i:03d}.npz", **arrays)

        print(f"Saved backtest to {root}")
        return root

    @classmethod
    def load(cls, path: str | Path) -> BacktestSummary:
        """Load a previously saved backtest.

        Parameters
        ----------
        path
            Directory written by :meth:`save`.

        Returns
        -------
        BacktestSummary
        """
        root = Path(path)
        meta = json.loads((root / "meta.json").read_text())
        summary_df = pd.read_csv(root / "summary.csv", index_col="node")
        run_config = meta.get("run_config", {})

        folds: list[BacktestResult] = []
        for fold_info in meta["folds"]:
            i = fold_info["fold_index"]
            npz_path = root / "folds" / f"fold_{i:03d}.npz"
            data = np.load(npz_path)

            forecasts: dict[ModelKey, np.ndarray] = {}
            actuals: dict[ModelKey, np.ndarray] = {}
            for arr_name in data.files:
                prefix, key_str = arr_name.split("__", 1)
                # key_str still has the leading _ from the double __
                # Actually the split gives us: "forecast" and "_<key>"
                # because the separator is "__" but split("__", 1)
                # handles it correctly
                node_key = _node_key_from_str(key_str)
                if prefix == "forecast":
                    forecasts[node_key] = data[arr_name]
                elif prefix == "actual":
                    actuals[node_key] = data[arr_name]

            # Deserialize metrics: str keys -> ModelKey
            metrics: dict[ModelKey, dict[str, float]] = {}
            for k_str, m in fold_info["metrics"].items():
                # Match the ModelKey from the arrays (which we already parsed)
                for mk in forecasts:
                    if str(mk) == k_str:
                        metrics[mk] = m
                        break

            folds.append(BacktestResult(
                cutoff=fold_info["cutoff"],
                horizon=fold_info["horizon"],
                metrics=metrics,
                forecasts=forecasts,
                actuals=actuals,
            ))

        return cls(folds=folds, summary_df=summary_df, run_config=run_config)

    @staticmethod
    def load_y_data(path: str | Path) -> dict[ModelKey, np.ndarray] | None:
        """Load y_data saved alongside a backtest run.

        Returns ``None`` if no ``y_data.npz`` file exists.
        """
        npz_path = Path(path) / "y_data.npz"
        if not npz_path.exists():
            return None
        data = np.load(npz_path)
        return {_node_key_from_str(name): data[name] for name in data.files}

    def reproduce_config(self) -> dict[str, Any]:
        """Return the config dict needed to re-run this backtest.

        Includes deserialized hierarchy, dag, and node_configs ready to
        pass into :class:`Backtester`.
        """
        cfg = self.run_config
        if not cfg:
            raise ValueError("No run_config stored — cannot reproduce.")

        hierarchy = _deserialize_hierarchy(cfg.get("hierarchy_edges", []))
        dag = _deserialize_causal_dag(cfg.get("dag_edges", []))

        # Build key lookup from hierarchy + dag
        key_lookup: dict[str, ModelKey] = {}
        for k in hierarchy.all_parents | hierarchy.all_children:
            key_lookup[str(k)] = k
        for k in dag.internal_nodes:
            key_lookup[str(k)] = k

        node_configs = _deserialize_node_configs(cfg.get("node_configs", {}), key_lookup)

        return {
            "hierarchy": hierarchy,
            "causal_dag": dag,
            "node_configs": node_configs,
            "reconciliation": cfg.get("reconciliation", "bottom_up"),
            "run_kwargs": cfg.get("run_kwargs", {}),
        }

    def plot(
        self,
        y_data: dict[ModelKey, np.ndarray],
        *,
        nodes: list[ModelKey] | None = None,
        dates: pd.DatetimeIndex | np.ndarray | None = None,
        figsize: tuple[float, float] | None = None,
        save_path: str | None = None,
    ):
        """Plot actuals vs forecasts for each fold.

        Parameters
        ----------
        y_data
            Full historical series (same dict passed to ``Backtester.run``).
        nodes
            Subset of nodes to plot. Defaults to all nodes in the results.
        dates
            Date index aligned with ``y_data`` arrays. If ``None``, integer
            indices are used on the x-axis.
        figsize
            Figure size ``(width, height)``. Auto-sized if ``None``.
        save_path
            If given, save the figure to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if nodes is None:
            nodes = sorted(self.folds[0].metrics.keys(), key=str)

        n_folds = len(self.folds)
        n_nodes = len(nodes)
        ncols = min(n_nodes, 3)
        nrows = (n_nodes + ncols - 1) // ncols
        if figsize is None:
            figsize = (6 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False)
        colors = plt.cm.tab10.colors

        for idx, node in enumerate(nodes):
            ax = axes[idx // ncols][idx % ncols]
            full_y = y_data[node]
            T = len(full_y)
            x_full = dates if dates is not None else np.arange(T)

            # Full observed series
            ax.plot(x_full, full_y, "k-", linewidth=0.8, label="Observed")

            for fi, fold in enumerate(self.folds):
                if node not in fold.forecasts:
                    continue

                cutoff = fold.cutoff
                horizon = fold.horizon
                samples = fold.forecasts[node]
                median = np.median(samples, axis=0)
                lo = np.percentile(samples, 5, axis=0)
                hi = np.percentile(samples, 95, axis=0)

                # x-axis for forecast window
                if dates is not None:
                    x_fc = dates[cutoff: cutoff + horizon]
                    x_cut = dates[cutoff]
                else:
                    x_fc = np.arange(cutoff, cutoff + horizon)
                    x_cut = cutoff

                color = colors[fi % len(colors)]
                label_prefix = f"Fold {fi + 1}" if n_folds > 1 else "Forecast"

                ax.plot(x_fc, median, "-", color=color, linewidth=1.5, label=f"{label_prefix} median")
                ax.fill_between(x_fc, lo, hi, alpha=0.2, color=color, label=f"{label_prefix} 90% CI")

                # Train / test divider
                ax.axvline(x_cut, color=color, linestyle="--", linewidth=0.8, alpha=0.7)

            ax.set_title(str(node), fontsize=11)
            ax.legend(fontsize=7, loc="best")
            if dates is not None:
                ax.tick_params(axis="x", rotation=30)

        # Hide unused subplots
        for idx in range(n_nodes, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(
            f"Backtest: {n_folds} fold{'s' if n_folds > 1 else ''}, "
            f"horizon={self.folds[0].horizon}",
            fontsize=13,
        )

        if save_path:
            fig.savefig(save_path, dpi=150)

        return fig


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class Backtester:
    """Backtest a :class:`HierarchicalForecaster` on historical data.

    Supports single-split and expanding-window evaluation.

    Parameters
    ----------
    hierarchy
        Parent → children aggregation graph.
    causal_dag
        Causal / predictive edges.
    node_configs
        Per-node forecaster configuration.
    reconciliation
        Reconciliation strategy passed to the forecaster.

    Examples
    --------
    ```python
    bt = Backtester(hierarchy, dag, node_configs)

    # Single split
    result = bt.run(
        y_data, x_data,
        mode="single",
        test_size=12,
    )

    # Expanding window
    result = bt.run(
        y_data, x_data,
        mode="expanding",
        test_size=12,
        n_splits=3,
    )

    print(result.summary_df)
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

    def run(
        self,
        y_data: dict[ModelKey, np.ndarray],
        x_data: dict[ExternalNode, np.ndarray] | None = None,
        *,
        mode: Literal["single", "expanding"] = "single",
        test_size: int = 12,
        n_splits: int = 1,
        min_train_size: int | None = None,
        num_warmup: int = 200,
        num_samples: int = 500,
        num_chains: int = 1,
        rng_seed: int = 0,
        coverage_level: float = 0.90,
        run_name: str | None = None,
        run_path: str | Path | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> BacktestSummary:
        """Run the backtest.

        Parameters
        ----------
        y_data
            Internal series, shape ``(T,)`` each.
        x_data
            External predictor series, shape ``(T,)`` each.
        mode
            ``"single"`` for one train/test split, ``"expanding"`` for
            multiple expanding-window folds.
        test_size
            Number of time steps in each test window (= forecast
            horizon).
        n_splits
            Number of folds for ``"expanding"`` mode.  Ignored when
            ``mode="single"``.
        min_train_size
            Minimum training observations.  Defaults to ``2 * test_size``.
        num_warmup
            NUTS warmup iterations per fold.
        num_samples
            Posterior samples per fold.
        num_chains
            MCMC chains per fold.
        rng_seed
            Base PRNG seed (incremented per fold).
        coverage_level
            Nominal coverage for interval metrics.
        run_name
            Optional human-readable name for this run.
        run_path
            If given, save results to this directory after completion.

        Returns
        -------
        BacktestSummary
        """
        import ergodicts

        t0 = time.time()
        x_data = x_data or {}
        if min_train_size is None:
            min_train_size = 2 * test_size

        # Determine total length from shortest series
        T = min(len(v) for v in y_data.values())

        # --- Compute cutoff points ---
        cutoffs = self._compute_cutoffs(
            T=T,
            mode=mode,
            test_size=test_size,
            n_splits=n_splits,
            min_train_size=min_train_size,
        )

        # --- Run each fold ---
        folds: list[BacktestResult] = []
        for fold_idx, cutoff in enumerate(cutoffs):
            print(
                f"Fold {fold_idx + 1}/{len(cutoffs)}: "
                f"train=[0:{cutoff}], test=[{cutoff}:{cutoff + test_size}]"
            )
            if progress_callback is not None:
                progress_callback("fold_start", {"fold": fold_idx, "total": len(cutoffs)})
            result = self._run_fold(
                y_data=y_data,
                x_data=x_data,
                cutoff=cutoff,
                horizon=test_size,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                rng_seed=rng_seed + fold_idx,
                coverage_level=coverage_level,
            )
            folds.append(result)
            if progress_callback is not None:
                fold_metrics = {str(k): v for k, v in result.metrics.items()}
                progress_callback("fold_done", {"fold": fold_idx, "metrics": fold_metrics})

        elapsed = time.time() - t0

        # --- Build run_config for reproducibility -------------------------
        run_config: dict[str, Any] = {
            "run_name": run_name or "",
            "version": ergodicts.__version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": round(elapsed, 1),
            "hierarchy_edges": _serialize_hierarchy(self.hierarchy),
            "dag_edges": _serialize_causal_dag(self.causal_dag),
            "node_configs": _serialize_node_configs(self.node_configs),
            "reconciliation": self.reconciliation,
            "run_kwargs": {
                "mode": mode,
                "test_size": test_size,
                "n_splits": n_splits,
                "min_train_size": min_train_size,
                "num_warmup": num_warmup,
                "num_samples": num_samples,
                "num_chains": num_chains,
                "rng_seed": rng_seed,
                "coverage_level": coverage_level,
            },
            "data_info": {
                "T": T,
                "n_internal_series": len(y_data),
                "n_external_series": len(x_data),
                "internal_keys": [str(k) for k in sorted(y_data.keys(), key=str)],
                "external_keys": [str(k) for k in sorted(x_data.keys(), key=str)],
            },
        }

        # --- Aggregate ---
        summary_df = self._aggregate_folds(folds)
        summary = BacktestSummary(folds=folds, summary_df=summary_df, run_config=run_config)

        # --- Auto-save if run_path provided -------------------------------
        if run_path is not None:
            summary.save(run_path)
            # Save y_data alongside for plotting
            y_arrays = {_node_key_to_str(k): v for k, v in y_data.items()}
            np.savez_compressed(Path(run_path) / "y_data.npz", **y_arrays)

        return summary

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _compute_cutoffs(
        T: int,
        mode: str,
        test_size: int,
        n_splits: int,
        min_train_size: int,
    ) -> list[int]:
        """Return sorted list of cutoff indices."""
        if mode == "single":
            cutoff = T - test_size
            if cutoff < min_train_size:
                raise ValueError(
                    f"Not enough data: T={T}, test_size={test_size}, "
                    f"min_train_size={min_train_size}"
                )
            return [cutoff]

        # Expanding window: place n_splits cutoffs
        last_cutoff = T - test_size
        first_cutoff = max(min_train_size, last_cutoff - (n_splits - 1) * test_size)

        if first_cutoff > last_cutoff:
            raise ValueError(
                f"Not enough data for {n_splits} expanding splits: "
                f"T={T}, test_size={test_size}, min_train_size={min_train_size}"
            )

        step = max(1, (last_cutoff - first_cutoff) // max(n_splits - 1, 1))
        cutoffs = list(range(first_cutoff, last_cutoff + 1, step))
        # Ensure we include the last cutoff
        if cutoffs[-1] != last_cutoff:
            cutoffs.append(last_cutoff)
        return cutoffs[:n_splits]

    def _run_fold(
        self,
        y_data: dict[ModelKey, np.ndarray],
        x_data: dict[ExternalNode, np.ndarray],
        cutoff: int,
        horizon: int,
        num_warmup: int,
        num_samples: int,
        num_chains: int,
        rng_seed: int,
        coverage_level: float,
    ) -> BacktestResult:
        """Fit on [0:cutoff], forecast [cutoff:cutoff+horizon], score."""
        # Slice training data
        y_train = {k: v[:cutoff] for k, v in y_data.items()}
        x_train = {k: v[:cutoff] for k, v in x_data.items()}

        # Fit
        model = HierarchicalForecaster(
            hierarchy=self.hierarchy,
            causal_dag=self.causal_dag,
            node_configs=self.node_configs,
            reconciliation=self.reconciliation,
        )
        model.fit(
            y_data=y_train,
            x_data=x_train,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            rng_seed=rng_seed,
        )

        # Forecast
        forecasts = model.forecast(horizon=horizon, rng_seed=rng_seed + 1000)

        # Ground truth
        actuals: dict[ModelKey, np.ndarray] = {}
        for k in forecasts:
            actuals[k] = y_data[k][cutoff: cutoff + horizon]

        # Metrics
        metrics: dict[ModelKey, dict[str, float]] = {}
        for k in forecasts:
            metrics[k] = compute_metrics(
                actual=actuals[k],
                samples=forecasts[k],
                coverage_level=coverage_level,
            )

        return BacktestResult(
            cutoff=cutoff,
            horizon=horizon,
            metrics=metrics,
            forecasts=forecasts,
            actuals=actuals,
        )

    @staticmethod
    def _aggregate_folds(folds: list[BacktestResult]) -> pd.DataFrame:
        """Average metrics across folds into a summary DataFrame."""
        if not folds:
            return pd.DataFrame()

        # Collect all (node, metric, value) tuples
        all_nodes = set()
        for f in folds:
            all_nodes.update(f.metrics.keys())

        rows = []
        for node in sorted(all_nodes, key=str):
            node_metrics: dict[str, list[float]] = {}
            for f in folds:
                if node in f.metrics:
                    for metric_name, val in f.metrics[node].items():
                        node_metrics.setdefault(metric_name, []).append(val)

            row = {"node": str(node)}
            for metric_name, vals in node_metrics.items():
                row[metric_name] = float(np.nanmean(vals))
            rows.append(row)

        df = pd.DataFrame(rows).set_index("node")
        return df
