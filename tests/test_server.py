"""Tests for the backtest dashboard server."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ergodicts.backtester import (
    BacktestResult,
    BacktestSummary,
    _node_key_to_str,
)
from ergodicts.reducer import ModelKey
from ergodicts.server import create_app

# Inline import to avoid top-level fastapi dependency issues
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def node_a():
    return ModelKey(("LEVEL",), ("A",))


@pytest.fixture
def node_b():
    return ModelKey(("LEVEL",), ("B",))


@pytest.fixture
def fake_run(tmp_path, node_a, node_b):
    """Create a minimal saved run on disk."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    folds_dir = run_dir / "folds"
    folds_dir.mkdir()

    rng = np.random.default_rng(0)

    # Build a BacktestResult
    forecasts_a = rng.normal(100, 5, (10, 6))
    forecasts_b = rng.normal(50, 3, (10, 6))
    actuals_a = rng.normal(100, 5, 6)
    actuals_b = rng.normal(50, 3, 6)

    metrics_a = {"mae": 3.2, "rmse": 4.1, "mape": 3.5, "smape": 3.4, "coverage": 0.9, "crps": 2.1}
    metrics_b = {"mae": 1.8, "rmse": 2.3, "mape": 4.0, "smape": 3.9, "coverage": 0.85, "crps": 1.5}

    fold = BacktestResult(
        cutoff=48,
        horizon=6,
        metrics={node_a: metrics_a, node_b: metrics_b},
        forecasts={node_a: forecasts_a, node_b: forecasts_b},
        actuals={node_a: actuals_a, node_b: actuals_b},
    )

    import pandas as pd

    summary_df = pd.DataFrame([
        {"node": str(node_a), **metrics_a},
        {"node": str(node_b), **metrics_b},
    ]).set_index("node")

    summary = BacktestSummary(
        folds=[fold],
        summary_df=summary_df,
        run_config={
            "run_name": "test_run",
            "timestamp": "2026-01-01T00:00:00",
            "elapsed_seconds": 10.5,
            "data_info": {"n_internal_series": 2, "n_external_series": 0},
            "run_kwargs": {"mode": "single", "test_size": 6},
        },
    )
    summary.save(run_dir)

    # Save y_data too
    y_arrays = {
        _node_key_to_str(node_a): rng.normal(100, 5, 60),
        _node_key_to_str(node_b): rng.normal(50, 3, 60),
    }
    np.savez_compressed(run_dir / "y_data.npz", **y_arrays)

    return tmp_path


@pytest.fixture
def client(fake_run):
    app = create_app(fake_run)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_returns_list(self, client):
        res = client.get("/api/runs")
        assert res.status_code == 200
        runs = res.json()
        assert isinstance(runs, list)
        assert len(runs) == 1
        assert runs[0]["id"] == "test_run"
        assert runs[0]["run_name"] == "test_run"

    def test_sorted_newest_first(self, fake_run):
        # Create a second run with a later timestamp
        run2 = fake_run / "run2"
        run2.mkdir()
        meta = {
            "run_config": {"run_name": "run2", "timestamp": "2026-06-01T00:00:00", "elapsed_seconds": 5},
            "n_folds": 0,
            "folds": [],
        }
        (run2 / "meta.json").write_text(json.dumps(meta))

        app = create_app(fake_run)
        client = TestClient(app)
        res = client.get("/api/runs")
        runs = res.json()
        assert len(runs) == 2
        assert runs[0]["id"] == "run2"  # newer first


class TestGetRun:
    def test_returns_meta(self, client):
        res = client.get("/api/runs/test_run")
        assert res.status_code == 200
        data = res.json()
        assert data["run_config"]["run_name"] == "test_run"
        assert data["n_folds"] == 1

    def test_not_found(self, client):
        res = client.get("/api/runs/nonexistent")
        assert res.status_code == 404


class TestGetSummary:
    def test_returns_rows(self, client):
        res = client.get("/api/runs/test_run/summary")
        assert res.status_code == 200
        rows = res.json()
        assert len(rows) == 2
        assert "mae" in rows[0]


class TestGetCharts:
    def test_returns_plotly_json(self, client):
        res = client.get("/api/runs/test_run/charts")
        assert res.status_code == 200
        data = res.json()
        assert "charts" in data
        charts = data["charts"]
        assert len(charts) == 2  # node_a and node_b
        for node_name, fig in charts.items():
            assert "data" in fig
            assert "layout" in fig
            assert len(fig["data"]) > 0

    def test_charts_without_y_data(self, fake_run):
        """Charts should still work even without y_data.npz."""
        import os
        y_data_path = fake_run / "test_run" / "y_data.npz"
        if y_data_path.exists():
            os.remove(y_data_path)
        app = create_app(fake_run)
        client = TestClient(app)
        res = client.get("/api/runs/test_run/charts")
        assert res.status_code == 200
        charts = res.json()["charts"]
        assert len(charts) == 2


class TestGetFold:
    def test_returns_fold_detail(self, client):
        res = client.get("/api/runs/test_run/fold/0")
        assert res.status_code == 200
        data = res.json()
        assert data["fold_index"] == 0
        assert data["cutoff"] == 48
        assert data["horizon"] == 6
        assert "series" in data
        assert len(data["series"]) > 0
        # Check Plotly-friendly fields
        for node_name, s in data["series"].items():
            assert "forecast_median" in s
            assert "forecast_p5" in s
            assert "forecast_p95" in s
            assert "forecast_p25" in s
            assert "forecast_p75" in s
            assert "actual" in s
            assert "observed_full" in s  # y_data.npz is present in fixture

    def test_fold_out_of_range(self, client):
        res = client.get("/api/runs/test_run/fold/99")
        assert res.status_code == 404


class TestComponents:
    def test_returns_library(self, client):
        res = client.get("/api/components")
        assert res.status_code == 200
        lib = res.json()
        assert isinstance(lib, dict)
        assert "local_linear_trend" in lib
        assert "fourier_seasonality" in lib
        assert lib["fourier_seasonality"]["role"] == "seasonality"
        assert "n_harmonics" in lib["fourier_seasonality"]["params"]


class TestStaticFiles:
    def test_index_served(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "Ergodicts" in res.text


class TestLaunchRun:
    def test_missing_data_path(self, client):
        res = client.post("/api/runs", json={
            "run_name": "bad_run",
            "data_path": "/nonexistent/data.npz",
        })
        assert res.status_code == 400


class TestProgress:
    def test_completed_run_sends_complete(self, client):
        res = client.get("/api/runs/test_run/progress")
        assert res.status_code == 200
        # SSE stream — read the first event
        text = res.text
        assert "complete" in text

    def test_nonexistent_run(self, client):
        res = client.get("/api/runs/nonexistent/progress")
        assert res.status_code == 404
