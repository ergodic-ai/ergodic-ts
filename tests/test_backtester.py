"""Tests for the backtester module."""

from __future__ import annotations

import numpy as np
import pytest

from ergodicts.backtester import (
    Backtester,
    BacktestResult,
    BacktestSummary,
    compute_metrics,
    coverage,
    crps,
    mae,
    mape,
    rmse,
    smape,
)
from ergodicts.causal_dag import CausalDAG, ExternalNode, NodeConfig
from ergodicts.components import FourierSeasonality, LocalLinearTrend
from ergodicts.forecaster import HierarchicalForecaster
from ergodicts.reducer import DependencyGraph, ModelKey


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def parent():
    return ModelKey(("LEVEL",), ("Parent",))


@pytest.fixture
def child_a():
    return ModelKey(("LEVEL",), ("A",))


@pytest.fixture
def child_b():
    return ModelKey(("LEVEL",), ("B",))


@pytest.fixture
def hierarchy(parent, child_a, child_b):
    g = DependencyGraph()
    g.add(parent, child_a)
    g.add(parent, child_b)
    return g


@pytest.fixture
def ext_x1():
    return ExternalNode("X1", dynamics="ar1")


@pytest.fixture
def dag(ext_x1, child_a, child_b):
    d = CausalDAG()
    d.add_edge(ext_x1, child_a, lag=1)
    d.add_edge(ext_x1, child_b, lag=1)
    return d


@pytest.fixture
def y_data(parent, child_a, child_b, rng):
    T = 60
    t = np.arange(T)
    y_a = 100.0 + 10.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, T)
    y_b = 50.0 + 5.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, T)
    return {parent: y_a + y_b, child_a: y_a, child_b: y_b}


@pytest.fixture
def x_data(ext_x1, rng):
    T = 60
    return {ext_x1: np.cumsum(rng.normal(0, 0.1, T)) + 100}


# ---------------------------------------------------------------------------
# TestMetrics
# ---------------------------------------------------------------------------


class TestMAE:
    def test_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mae(a, a) == 0.0

    def test_known(self):
        a = np.array([1.0, 2.0, 3.0])
        p = np.array([2.0, 3.0, 4.0])
        assert mae(a, p) == 1.0


class TestRMSE:
    def test_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert rmse(a, a) == 0.0

    def test_known(self):
        a = np.array([0.0, 0.0])
        p = np.array([3.0, 4.0])
        assert rmse(a, p) == pytest.approx(np.sqrt(12.5))


class TestMAPE:
    def test_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mape(a, a) == 0.0

    def test_zero_actual(self):
        a = np.array([0.0, 0.0])
        p = np.array([1.0, 2.0])
        assert np.isnan(mape(a, p))

    def test_known(self):
        a = np.array([100.0, 200.0])
        p = np.array([110.0, 220.0])
        assert mape(a, p) == 10.0


class TestSMAPE:
    def test_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert smape(a, a) == 0.0

    def test_symmetric(self):
        a = np.array([100.0])
        p = np.array([120.0])
        # SMAPE = 2*20 / (100+120) * 100 = 18.18...
        assert smape(a, p) == pytest.approx(smape(p, a))

    def test_both_zero(self):
        a = np.array([0.0])
        p = np.array([0.0])
        assert smape(a, p) == 0.0


class TestCoverage:
    def test_perfect_coverage(self):
        actual = np.array([5.0, 5.0, 5.0])
        # All samples are 5.0 -> interval is [5, 5]
        samples = np.full((100, 3), 5.0)
        assert coverage(actual, samples) == 1.0

    def test_zero_coverage(self):
        actual = np.array([100.0])
        # All samples are 0
        samples = np.zeros((100, 1))
        assert coverage(actual, samples) == 0.0

    def test_partial_coverage(self, rng):
        actual = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        samples = rng.normal(0, 1, (1000, 5))
        # 90% CI should cover ~90% of standard normal draws at 0
        cov = coverage(actual, samples, level=0.90)
        assert 0.8 <= cov <= 1.0


class TestCRPS:
    def test_perfect(self):
        actual = np.array([5.0])
        samples = np.full((100, 1), 5.0)
        assert crps(actual, samples) == pytest.approx(0.0, abs=1e-10)

    def test_wider_is_worse(self, rng):
        actual = np.array([0.0, 0.0, 0.0])
        narrow = rng.normal(0, 0.1, (500, 3))
        wide = rng.normal(0, 10.0, (500, 3))
        assert crps(actual, narrow) < crps(actual, wide)


class TestComputeMetrics:
    def test_returns_all_keys(self, rng):
        actual = np.array([1.0, 2.0, 3.0])
        samples = rng.normal(0, 1, (100, 3)) + actual[None, :]
        result = compute_metrics(actual, samples)
        assert set(result.keys()) == {"mae", "rmse", "mape", "smape", "rolling_mape_3", "rolling_mape_6", "mape_full", "accuracy", "rolling_accuracy_3", "rolling_accuracy_6", "accuracy_full", "coverage", "crps"}

    def test_values_are_finite(self, rng):
        actual = np.array([10.0, 20.0, 30.0])
        samples = rng.normal(0, 1, (100, 3)) + actual[None, :]
        result = compute_metrics(actual, samples)
        for k, v in result.items():
            # rolling metrics with window=6 return NaN when horizon < 6
            if k in ("rolling_mape_6", "rolling_accuracy_6"):
                continue
            assert np.isfinite(v), f"{k} is not finite: {v}"


# ---------------------------------------------------------------------------
# TestBacktesterCutoffs
# ---------------------------------------------------------------------------


class TestCutoffs:
    def test_single(self):
        cutoffs = Backtester._compute_cutoffs(
            T=60, mode="single", test_size=12, n_splits=1, min_train_size=24,
        )
        assert cutoffs == [48]

    def test_single_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Not enough data"):
            Backtester._compute_cutoffs(
                T=30, mode="single", test_size=12, n_splits=1, min_train_size=24,
            )

    def test_expanding(self):
        cutoffs = Backtester._compute_cutoffs(
            T=60, mode="expanding", test_size=6, n_splits=3, min_train_size=24,
        )
        assert len(cutoffs) == 3
        # All cutoffs should be valid
        for c in cutoffs:
            assert c >= 24
            assert c + 6 <= 60
        # Should be in ascending order
        assert cutoffs == sorted(cutoffs)

    def test_expanding_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Not enough data"):
            Backtester._compute_cutoffs(
                T=20, mode="expanding", test_size=12, n_splits=5, min_train_size=24,
            )


# ---------------------------------------------------------------------------
# TestBacktester (integration — tiny MCMC)
# ---------------------------------------------------------------------------


class TestBacktester:
    def test_single_split(self, hierarchy, dag, y_data, x_data, child_a, child_b, parent):
        configs = {
            parent: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_a: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_b: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        bt = Backtester(hierarchy, dag, configs)
        result = bt.run(
            y_data, x_data,
            mode="single",
            test_size=6,
            num_warmup=5,
            num_samples=5,
        )
        assert isinstance(result, BacktestSummary)
        assert len(result.folds) == 1
        assert not result.summary_df.empty

        # Check metrics exist for all nodes
        fold = result.folds[0]
        assert child_a in fold.metrics
        assert child_b in fold.metrics
        assert "mae" in fold.metrics[child_a]
        assert fold.forecasts[child_a].shape == (5, 6)
        assert fold.actuals[child_a].shape == (6,)

    def test_expanding_window(self, hierarchy, dag, y_data, x_data, parent, child_a, child_b):
        configs = {
            parent: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_a: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_b: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        bt = Backtester(hierarchy, dag, configs)
        result = bt.run(
            y_data, x_data,
            mode="expanding",
            test_size=6,
            n_splits=2,
            num_warmup=5,
            num_samples=5,
        )
        assert len(result.folds) == 2

        # Summary should average across folds
        assert "mae" in result.summary_df.columns
        assert len(result.summary_df) >= 2  # at least child_a and child_b

    def test_summary_repr(self, hierarchy, dag, y_data, x_data, parent, child_a, child_b):
        configs = {
            parent: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_a: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_b: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        bt = Backtester(hierarchy, dag, configs)
        result = bt.run(
            y_data, x_data,
            mode="single",
            test_size=6,
            num_warmup=5,
            num_samples=5,
        )
        r = repr(result)
        assert "folds=1" in r


# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_roundtrip(self, hierarchy, dag, y_data, x_data, parent, child_a, child_b, tmp_path):
        configs = {
            parent: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_a: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_b: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        bt = Backtester(hierarchy, dag, configs)
        result = bt.run(
            y_data, x_data,
            mode="single",
            test_size=6,
            num_warmup=5,
            num_samples=5,
            run_name="test_roundtrip",
            run_path=tmp_path / "run1",
        )

        # Verify files exist
        assert (tmp_path / "run1" / "meta.json").exists()
        assert (tmp_path / "run1" / "summary.csv").exists()
        assert (tmp_path / "run1" / "folds" / "fold_000.npz").exists()

        # Load back
        loaded = BacktestSummary.load(tmp_path / "run1")
        assert len(loaded.folds) == len(result.folds)
        assert loaded.run_config["run_name"] == "test_roundtrip"
        assert not loaded.summary_df.empty

        # Check metrics match
        for k in result.folds[0].metrics:
            orig = result.folds[0].metrics[k]
            load = loaded.folds[0].metrics[k]
            for metric_name in orig:
                assert abs(orig[metric_name] - load[metric_name]) < 1e-6

        # Check arrays match
        for k in result.folds[0].forecasts:
            np.testing.assert_array_almost_equal(
                result.folds[0].forecasts[k],
                loaded.folds[0].forecasts[k],
            )
            np.testing.assert_array_almost_equal(
                result.folds[0].actuals[k],
                loaded.folds[0].actuals[k],
            )

    def test_save_load_expanding(self, hierarchy, dag, y_data, x_data, parent, child_a, child_b, tmp_path):
        configs = {
            parent: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_a: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_b: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        bt = Backtester(hierarchy, dag, configs)
        result = bt.run(
            y_data, x_data,
            mode="expanding",
            test_size=6,
            n_splits=2,
            num_warmup=5,
            num_samples=5,
            run_name="expanding_test",
            run_path=tmp_path / "run2",
        )

        loaded = BacktestSummary.load(tmp_path / "run2")
        assert len(loaded.folds) == 2
        assert (tmp_path / "run2" / "folds" / "fold_001.npz").exists()

    def test_run_config_stored(self, hierarchy, dag, y_data, x_data, parent, child_a, child_b):
        configs = {
            parent: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_a: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_b: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        bt = Backtester(hierarchy, dag, configs)
        result = bt.run(
            y_data, x_data,
            mode="single",
            test_size=6,
            num_warmup=5,
            num_samples=5,
            rng_seed=42,
            run_name="config_test",
        )

        cfg = result.run_config
        assert cfg["run_name"] == "config_test"
        assert cfg["run_kwargs"]["mode"] == "single"
        assert cfg["run_kwargs"]["test_size"] == 6
        assert cfg["run_kwargs"]["rng_seed"] == 42
        assert cfg["data_info"]["T"] == 60
        assert len(cfg["hierarchy_edges"]) == 2  # parent -> child_a, parent -> child_b
        assert cfg["reconciliation"] == "bottom_up"
        assert "timestamp" in cfg
        assert "elapsed_seconds" in cfg

    def test_reproduce_config(self, hierarchy, dag, y_data, x_data, parent, child_a, child_b, tmp_path):
        configs = {
            parent: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_a: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_b: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        bt = Backtester(hierarchy, dag, configs)
        result = bt.run(
            y_data, x_data,
            mode="single",
            test_size=6,
            num_warmup=5,
            num_samples=5,
            run_path=tmp_path / "repro",
        )

        loaded = BacktestSummary.load(tmp_path / "repro")
        repro = loaded.reproduce_config()

        # Check deserialized objects
        assert isinstance(repro["hierarchy"], DependencyGraph)
        assert isinstance(repro["causal_dag"], CausalDAG)
        assert len(repro["node_configs"]) == 3
        assert repro["reconciliation"] == "bottom_up"
        assert repro["run_kwargs"]["test_size"] == 6

    def test_repr_with_name(self, hierarchy, dag, y_data, x_data, parent, child_a, child_b):
        configs = {
            parent: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_a: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_b: NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        bt = Backtester(hierarchy, dag, configs)
        result = bt.run(
            y_data, x_data,
            mode="single",
            test_size=6,
            num_warmup=5,
            num_samples=5,
            run_name="my_run",
        )
        assert "my_run" in repr(result)
