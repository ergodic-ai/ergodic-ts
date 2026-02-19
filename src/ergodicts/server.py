"""Backtest dashboard server — FastAPI backend + static frontend."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ergodicts.backtester import BacktestSummary, _node_key_from_str, _node_key_to_str

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunListItem(BaseModel):
    id: str
    run_name: str
    timestamp: str
    n_folds: int
    elapsed_seconds: float
    n_series: int


class LaunchRequest(BaseModel):
    run_name: str = "unnamed"
    data_path: str
    mode: str = "single"
    test_size: int = 12
    n_splits: int = 1
    num_warmup: int = 200
    num_samples: int = 500
    num_chains: int = 1
    rng_seed: int = 0
    reconciliation: str = "bottom_up"


class LaunchResponse(BaseModel):
    run_id: str
    status: str


# ---------------------------------------------------------------------------
# In-memory state for active runs
# ---------------------------------------------------------------------------

_active_runs: dict[str, asyncio.Queue] = {}
_run_status: dict[str, str] = {}  # run_id -> "running" | "completed" | "failed"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(runs_dir: Path) -> FastAPI:
    """Create the FastAPI app bound to *runs_dir*."""

    app = FastAPI(title="Ergodicts Backtest Dashboard")
    runs_dir.mkdir(parents=True, exist_ok=True)

    # -- helpers -----------------------------------------------------------

    def _scan_runs() -> list[dict[str, Any]]:
        items = []
        for d in sorted(runs_dir.iterdir(), key=lambda p: p.name):
            meta_path = d / "meta.json"
            if d.is_dir() and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                cfg = meta.get("run_config", {})
                items.append({
                    "id": d.name,
                    "run_name": cfg.get("run_name", d.name),
                    "timestamp": cfg.get("timestamp", ""),
                    "n_folds": meta.get("n_folds", 0),
                    "elapsed_seconds": cfg.get("elapsed_seconds", 0),
                    "n_series": cfg.get("data_info", {}).get("n_internal_series", 0),
                })
        # Sort newest first
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return items

    def _get_run_dir(run_id: str) -> Path:
        run_path = runs_dir / run_id
        if not run_path.is_dir() or not (run_path / "meta.json").exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return run_path

    # -- API endpoints -----------------------------------------------------

    @app.get("/api/components")
    async def list_components() -> dict[str, Any]:
        """Return the full component library for the dashboard."""
        from ergodicts.components import ComponentLibrary
        return ComponentLibrary.all_components()

    @app.get("/api/runs")
    async def list_runs() -> list[dict[str, Any]]:
        runs = _scan_runs()
        # Append any in-progress runs that don't have meta.json yet
        for run_id, status in _run_status.items():
            if status == "running" and not any(r["id"] == run_id for r in runs):
                runs.insert(0, {
                    "id": run_id,
                    "run_name": run_id,
                    "timestamp": "",
                    "n_folds": 0,
                    "elapsed_seconds": 0,
                    "n_series": 0,
                    "status": "running",
                })
        return runs

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        # Check if it's an active run
        if run_id in _run_status and _run_status[run_id] == "running":
            return {"id": run_id, "status": "running"}
        run_path = _get_run_dir(run_id)
        meta = json.loads((run_path / "meta.json").read_text())
        meta["id"] = run_id
        meta["status"] = _run_status.get(run_id, "completed")
        return meta

    @app.get("/api/runs/{run_id}/summary")
    async def get_summary(run_id: str) -> list[dict[str, Any]]:
        run_path = _get_run_dir(run_id)
        import pandas as pd

        df = pd.read_csv(run_path / "summary.csv")
        return df.to_dict(orient="records")

    @app.get("/api/runs/{run_id}/charts")
    async def get_charts(run_id: str) -> dict[str, Any]:
        """Return Plotly JSON figures — one per node — built from fold data."""
        run_path = _get_run_dir(run_id)
        summary = BacktestSummary.load(run_path)
        y_data = BacktestSummary.load_y_data(run_path)

        # Read time_index from meta if available
        time_index: list[str] | None = summary.time_index

        colors = [
            "#2563eb", "#dc2626", "#16a34a", "#9333ea",
            "#ea580c", "#0891b2", "#ca8a04", "#db2777",
        ]

        if not summary.folds:
            return {"charts": {}}

        # Discover all nodes
        all_nodes = set()
        for fold in summary.folds:
            all_nodes.update(fold.metrics.keys())
        all_nodes = sorted(all_nodes, key=str)

        charts: dict[str, Any] = {}
        for node in all_nodes:
            node_label = str(node)
            traces = []

            # Full observed series if y_data available
            if y_data is not None and node in y_data:
                full_y = y_data[node]
                x_obs = time_index[:len(full_y)] if time_index and len(time_index) >= len(full_y) else list(range(len(full_y)))
                traces.append({
                    "x": x_obs if isinstance(x_obs, list) else list(x_obs),
                    "y": full_y.tolist(),
                    "mode": "lines",
                    "name": "Observed",
                    "line": {"color": "black", "width": 1.5},
                })

            for fi, fold in enumerate(summary.folds):
                if node not in fold.forecasts:
                    continue
                samples = fold.forecasts[node]
                median = np.median(samples, axis=0)
                p5 = np.percentile(samples, 5, axis=0)
                p95 = np.percentile(samples, 95, axis=0)
                actual = fold.actuals.get(node)
                cutoff = fold.cutoff
                horizon = fold.horizon

                # Use time labels if available, otherwise integer indices
                if fold.time_labels:
                    x_fc = fold.time_labels[:horizon]
                elif time_index and cutoff + horizon <= len(time_index):
                    x_fc = time_index[cutoff:cutoff + horizon]
                else:
                    x_fc = list(range(cutoff, cutoff + horizon))

                color = colors[fi % len(colors)]
                n_folds = len(summary.folds)
                label = f"Fold {fi + 1}" if n_folds > 1 else "Forecast"

                # Actual for this fold window
                if actual is not None:
                    traces.append({
                        "x": x_fc,
                        "y": actual.tolist(),
                        "mode": "lines+markers",
                        "name": f"{label} actual",
                        "line": {"color": color, "width": 1, "dash": "dot"},
                        "marker": {"size": 4},
                    })

                # Median
                traces.append({
                    "x": x_fc,
                    "y": median.tolist(),
                    "mode": "lines",
                    "name": f"{label} median",
                    "line": {"color": color, "width": 2},
                })

                # 90% CI band (upper)
                traces.append({
                    "x": x_fc,
                    "y": p95.tolist(),
                    "mode": "lines",
                    "name": f"{label} 95th",
                    "line": {"width": 0},
                    "showlegend": False,
                })
                # 90% CI band (lower, fills to upper)
                traces.append({
                    "x": x_fc,
                    "y": p5.tolist(),
                    "mode": "lines",
                    "name": f"{label} 5th",
                    "line": {"width": 0},
                    "fill": "tonexty",
                    "fillcolor": color.replace(")", ", 0.15)").replace("rgb", "rgba")
                    if color.startswith("rgb") else color + "26",
                    "showlegend": False,
                })

                # Cutoff line via shapes (added in layout)

            # Determine cutoff x-values for shapes
            shapes = []
            for fi, fold in enumerate(summary.folds):
                if node not in fold.forecasts:
                    continue
                if fold.cutoff_label:
                    cutoff_x = fold.cutoff_label
                elif time_index and fold.cutoff < len(time_index):
                    cutoff_x = time_index[fold.cutoff]
                else:
                    cutoff_x = fold.cutoff
                shapes.append({
                    "type": "line",
                    "x0": cutoff_x, "x1": cutoff_x,
                    "y0": 0, "y1": 1,
                    "yref": "paper",
                    "line": {"color": colors[fi % len(colors)], "dash": "dash", "width": 1},
                })

            x_title = "Date" if time_index else "Time index"
            layout = {
                "title": {"text": node_label, "font": {"size": 14}},
                "xaxis": {"title": x_title},
                "yaxis": {"title": "Value"},
                "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
                "legend": {"font": {"size": 11}},
                "hovermode": "x unified",
                "shapes": shapes,
            }

            charts[node_label] = {"data": traces, "layout": layout}

        return {"charts": charts}

    @app.get("/api/runs/{run_id}/fold/{fold_idx}")
    async def get_fold(run_id: str, fold_idx: int) -> dict[str, Any]:
        run_path = _get_run_dir(run_id)
        meta = json.loads((run_path / "meta.json").read_text())

        if fold_idx < 0 or fold_idx >= meta["n_folds"]:
            raise HTTPException(status_code=404, detail=f"Fold {fold_idx} not found")

        npz_path = run_path / "folds" / f"fold_{fold_idx:03d}.npz"
        data = np.load(npz_path)

        fold_info = meta["folds"][fold_idx]
        result: dict[str, Any] = {
            "fold_index": fold_idx,
            "cutoff": fold_info["cutoff"],
            "horizon": fold_info["horizon"],
            "metrics": fold_info["metrics"],
            "series": {},
        }
        # Include time labels if available
        if fold_info.get("cutoff_label"):
            result["cutoff_label"] = fold_info["cutoff_label"]
        if fold_info.get("time_labels"):
            result["time_labels"] = fold_info["time_labels"]
        # Include full time_index if available
        ti = meta.get("run_config", {}).get("data_info", {}).get("time_index")
        if ti:
            result["time_index"] = ti

        # Load y_data if available (for full observed series context)
        y_data = BacktestSummary.load_y_data(run_path)

        for arr_name in data.files:
            prefix, key_str = arr_name.split("__", 1)
            node_key = _node_key_from_str(key_str)
            node_label = str(node_key)
            if prefix == "forecast":
                samples = data[arr_name]
                result["series"].setdefault(node_label, {})["forecast_median"] = np.median(samples, axis=0).tolist()
                result["series"][node_label]["forecast_p5"] = np.percentile(samples, 5, axis=0).tolist()
                result["series"][node_label]["forecast_p95"] = np.percentile(samples, 95, axis=0).tolist()
                result["series"][node_label]["forecast_p25"] = np.percentile(samples, 25, axis=0).tolist()
                result["series"][node_label]["forecast_p75"] = np.percentile(samples, 75, axis=0).tolist()
            elif prefix == "actual":
                result["series"].setdefault(node_label, {})["actual"] = data[arr_name].tolist()

        # Attach full observed series if available
        if y_data is not None:
            for node_label in list(result["series"].keys()):
                for nk in y_data:
                    if str(nk) == node_label:
                        result["series"][node_label]["observed_full"] = y_data[nk].tolist()
                        break

        # Load parameter summaries if available
        params_path = run_path / "folds" / f"params_{fold_idx:03d}.json"
        if params_path.exists():
            result["param_summary"] = json.loads(params_path.read_text())

        # Load convergence diagnostics if available
        diag_path = run_path / "folds" / f"diag_{fold_idx:03d}.json"
        if diag_path.exists():
            result["diagnostics"] = json.loads(diag_path.read_text())

        return result

    @app.get("/api/compare")
    async def compare_runs(run_a: str, run_b: str) -> dict[str, Any]:
        """Compare two runs side-by-side with per-metric deltas."""
        import pandas as pd

        path_a = _get_run_dir(run_a)
        path_b = _get_run_dir(run_b)

        df_a = pd.read_csv(path_a / "summary.csv", index_col="node")
        df_b = pd.read_csv(path_b / "summary.csv", index_col="node")

        meta_a = json.loads((path_a / "meta.json").read_text())
        meta_b = json.loads((path_b / "meta.json").read_text())

        cfg_a = meta_a.get("run_config", {})
        cfg_b = meta_b.get("run_config", {})

        # Find common nodes and metrics
        common_nodes = sorted(set(df_a.index) & set(df_b.index))
        common_metrics = sorted(set(df_a.columns) & set(df_b.columns))

        rows = []
        for node in common_nodes:
            row: dict[str, Any] = {"node": node}
            for metric in common_metrics:
                val_a = float(df_a.loc[node, metric]) if node in df_a.index else None
                val_b = float(df_b.loc[node, metric]) if node in df_b.index else None
                row[f"{metric}_a"] = val_a
                row[f"{metric}_b"] = val_b
                if val_a is not None and val_b is not None:
                    row[f"{metric}_delta"] = val_b - val_a
                else:
                    row[f"{metric}_delta"] = None
            rows.append(row)

        return {
            "run_a": {"id": run_a, "name": cfg_a.get("run_name", run_a)},
            "run_b": {"id": run_b, "name": cfg_b.get("run_name", run_b)},
            "metrics": common_metrics,
            "rows": rows,
        }

    @app.get("/api/runs/{run_id}/fold/{fold_idx}/param-cards")
    async def get_param_cards(run_id: str, fold_idx: int) -> dict[str, Any]:
        """Return structured parameter cards for each node's components."""
        from ergodicts.causal_dag import NodeConfig
        from ergodicts.components import (
            ExternalRegression,
            RegressionComponent,
            SeasonalityComponent,
            TrendComponent,
            resolve_components,
        )
        from ergodicts.reducer import ModelKey

        run_path = _get_run_dir(run_id)
        meta = json.loads((run_path / "meta.json").read_text())

        if fold_idx < 0 or fold_idx >= meta["n_folds"]:
            raise HTTPException(status_code=404, detail=f"Fold {fold_idx} not found")

        # Load parameter summary for this fold
        params_path = run_path / "folds" / f"params_{fold_idx:03d}.json"
        if not params_path.exists():
            return {"nodes": {}}

        param_summary = json.loads(params_path.read_text())
        cfg = meta.get("run_config", {})

        # Try to reconstruct node configs for describe_params
        try:
            summary = BacktestSummary.load(run_path)
            repro = summary.reproduce_config()
            node_configs = repro.get("node_configs", {})
            hierarchy = repro.get("hierarchy")
            dag = repro.get("causal_dag")
        except Exception:
            return {"nodes": {}, "param_summary": param_summary}

        # Build a fake samples dict from param_summary for describe_params
        # (This is approximate — uses mean values only, not full posterior)
        fake_samples: dict[str, Any] = {}
        for node_str, params in param_summary.items():
            for param_name, stats in params.items():
                full_key = f"{param_name}_{node_str}" if node_str != "_global" else param_name
                # Create a 1D array with the mean (describe_params uses extract_posterior)
                fake_samples[full_key] = np.array([stats["mean"]])

        result_nodes: dict[str, list[dict[str, Any]]] = {}
        for mk, ncfg in node_configs.items():
            node_str = str(mk)
            components = resolve_components(ncfg)
            cards_list = []
            for comp in components:
                try:
                    if isinstance(comp, TrendComponent):
                        desc = comp.describe_params(node_str, fake_samples)
                    elif isinstance(comp, SeasonalityComponent):
                        desc = comp.describe_params(node_str, fake_samples)
                    elif isinstance(comp, RegressionComponent):
                        # Get predictor names from DAG
                        pred_names = None
                        if dag:
                            edges = dag.parents_of(mk)
                            from ergodicts.causal_dag import ExternalNode as EN
                            pred_names = [e.source.name if isinstance(e.source, EN) else str(e.source) for e in edges]
                        desc = comp.describe_params(node_str, fake_samples, predictor_names=pred_names)
                    else:
                        desc = {"type": "unknown", "display_name": "Unknown", "cards": []}
                    cards_list.append(desc)
                except Exception:
                    cards_list.append({"type": comp.component_name, "display_name": comp.component_name, "cards": [], "error": "Failed to describe"})

            if cards_list:
                result_nodes[node_str] = cards_list

        return {"nodes": result_nodes}

    @app.get("/api/runs/{run_id}/fold/{fold_idx}/decomposition")
    async def get_decomposition(run_id: str, fold_idx: int) -> dict[str, Any]:
        """Return per-node, per-component decomposition for a fold."""
        run_path = _get_run_dir(run_id)
        meta = json.loads((run_path / "meta.json").read_text())

        if fold_idx < 0 or fold_idx >= meta["n_folds"]:
            raise HTTPException(status_code=404, detail=f"Fold {fold_idx} not found")

        decomp_path = run_path / "folds" / f"decomp_{fold_idx:03d}.npz"
        if not decomp_path.exists():
            raise HTTPException(status_code=404, detail="No decomposition data for this fold")

        decomp_data = np.load(decomp_path)
        nodes: dict[str, dict[str, Any]] = {}
        for arr_name in decomp_data.files:
            parts = arr_name.split("__", 1)
            if len(parts) != 2:
                continue
            comp_label, node_str = parts
            arr = decomp_data[arr_name]
            # Compute summary stats: median, p5, p95 across samples
            median = np.median(arr, axis=0).tolist()
            p5 = np.percentile(arr, 5, axis=0).tolist()
            p95 = np.percentile(arr, 95, axis=0).tolist()
            nodes.setdefault(node_str, {})[comp_label] = {
                "median": median, "p5": p5, "p95": p95,
            }

        # Load metadata (aggregator types + regression coefficients)
        result: dict[str, Any] = {"nodes": nodes}
        decomp_meta_path = run_path / "folds" / f"decomp_meta_{fold_idx:03d}.json"
        if decomp_meta_path.exists():
            decomp_meta = json.loads(decomp_meta_path.read_text())
            result["aggregator_type"] = decomp_meta.get("aggregator_type", {})
            result["regression_coefficients"] = decomp_meta.get("regression_coefficients", {})
        else:
            result["aggregator_type"] = {}
            result["regression_coefficients"] = {}

        # Include time labels if available
        fold_info = meta["folds"][fold_idx]
        if fold_info.get("time_labels"):
            result["time_labels"] = fold_info["time_labels"]

        return result

    @app.post("/api/runs")
    async def launch_run(req: LaunchRequest) -> LaunchResponse:
        from ergodicts.backtester import Backtester, _node_key_from_str

        data_path = Path(req.data_path)
        if not data_path.exists():
            raise HTTPException(status_code=400, detail=f"Data file not found: {req.data_path}")

        run_id = req.run_name.replace(" ", "_") + "_" + uuid.uuid4().hex[:8]
        run_path = runs_dir / run_id

        # Load data
        npz = np.load(data_path, allow_pickle=True)

        # Expect keys: y_data (dict saved via np.savez), hierarchy, dag, node_configs
        # For simplicity, load a BacktestSummary to get the reproduce_config
        config_path = npz["config_path"].item() if "config_path" in npz.files else None

        # Simple approach: data_path is an NPZ with y_<key>=array pairs
        # and a 'config_run_path' string pointing to a reference run for hierarchy/dag
        if "config_run_path" in npz.files:
            ref_path = Path(str(npz["config_run_path"].item()))
            ref_summary = BacktestSummary.load(ref_path)
            repro = ref_summary.reproduce_config()
        else:
            raise HTTPException(
                status_code=400,
                detail="Data NPZ must contain 'config_run_path' pointing to a reference run",
            )

        y_data = {}
        for name in npz.files:
            if name.startswith("y__"):
                key = _node_key_from_str(name[2:])  # strip "y_" prefix -> "_key"
                y_data[key] = npz[name]

        if not y_data:
            raise HTTPException(status_code=400, detail="No y_data arrays found in NPZ (expected y__<key> entries)")

        x_data = {}
        # TODO: support x_data loading

        queue: asyncio.Queue = asyncio.Queue()
        _active_runs[run_id] = queue
        _run_status[run_id] = "running"

        loop = asyncio.get_event_loop()

        def _progress_callback(event: str, payload: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, {"event": event, **payload})

        def _run_backtest():
            try:
                bt = Backtester(
                    hierarchy=repro["hierarchy"],
                    causal_dag=repro["causal_dag"],
                    node_configs=repro["node_configs"],
                    reconciliation=req.reconciliation,
                )
                bt.run(
                    y_data=y_data,
                    x_data=x_data,
                    mode=req.mode,
                    test_size=req.test_size,
                    n_splits=req.n_splits,
                    num_warmup=req.num_warmup,
                    num_samples=req.num_samples,
                    num_chains=req.num_chains,
                    rng_seed=req.rng_seed,
                    run_name=req.run_name,
                    run_path=run_path,
                    progress_callback=_progress_callback,
                )
                _run_status[run_id] = "completed"
                loop.call_soon_threadsafe(queue.put_nowait, {"event": "complete"})
            except Exception as exc:
                _run_status[run_id] = "failed"
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"event": "error", "message": str(exc)}
                )

        thread = threading.Thread(target=_run_backtest, daemon=True)
        thread.start()

        return LaunchResponse(run_id=run_id, status="running")

    @app.get("/api/runs/{run_id}/progress")
    async def stream_progress(run_id: str):
        if run_id not in _active_runs:
            # Already completed — send a single "complete" event
            if (runs_dir / run_id / "meta.json").exists():
                async def _done():
                    yield f"data: {json.dumps({'event': 'complete'})}\n\n"
                return StreamingResponse(_done(), media_type="text/event-stream")
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        queue = _active_runs[run_id]

        async def _generate():
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"
                    continue

                yield f"data: {json.dumps(msg, default=str)}\n\n"

                if msg.get("event") in ("complete", "error"):
                    _active_runs.pop(run_id, None)
                    break

        return StreamingResponse(_generate(), media_type="text/event-stream")

    # -- Static files (index.html) -----------------------------------------

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        @app.get("/")
        async def index():
            return FileResponse(static_dir / "index.html")

        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def serve(runs_dir: str = "runs", host: str = "0.0.0.0", port: int = 8765) -> None:
    """Launch the backtest dashboard server."""
    import uvicorn

    app = create_app(Path(runs_dir))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ergodicts Backtest Dashboard")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing saved runs")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(runs_dir=args.runs_dir, host=args.host, port=args.port)
