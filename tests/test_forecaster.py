"""Tests for the forecaster module."""

from __future__ import annotations

import numpy as np
import pytest

from ergodicts.causal_dag import CausalDAG, ExternalNode, NodeConfig
from ergodicts.components import (
    DampedLocalLinearTrend,
    FourierSeasonality,
    LocalLinearTrend,
    LogAdditiveAggregator,
    MultiplicativeAggregator,
    MultiplicativeFourierSeasonality,
    MultiplicativeSeasonality,
    OUMeanReversion,
)
from ergodicts.forecaster import (
    ForecastData,
    HierarchicalForecaster,
    _forward_fill,
    prepare_data,
)
from ergodicts.reducer import DependencyGraph, ModelKey


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def tiny_dag(ext_x1, child_a, child_b):
    dag = CausalDAG()
    dag.add_edge(ext_x1, child_a, lag=1)
    dag.add_edge(ext_x1, child_b, lag=1)
    return dag


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def tiny_y(parent, child_a, child_b, rng):
    T = 48
    t = np.arange(T)
    y_a = 100.0 + 10.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, T)
    y_b = 50.0 + 5.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, T)
    y_p = y_a + y_b
    return {parent: y_p, child_a: y_a, child_b: y_b}


@pytest.fixture
def tiny_x(ext_x1, rng):
    T = 48
    return {ext_x1: np.cumsum(rng.normal(0, 0.1, T)) + 100.0}


# ---------------------------------------------------------------------------
# TestForwardFill
# ---------------------------------------------------------------------------


class TestForwardFill:
    def test_no_nan(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _forward_fill(arr)
        np.testing.assert_array_equal(result, arr)

    def test_middle_nan(self):
        arr = np.array([1.0, np.nan, 3.0])
        result = _forward_fill(arr)
        np.testing.assert_array_equal(result, [1.0, 1.0, 3.0])

    def test_leading_nan(self):
        arr = np.array([np.nan, np.nan, 3.0])
        result = _forward_fill(arr)
        np.testing.assert_array_equal(result, [3.0, 3.0, 3.0])

    def test_all_nan(self):
        arr = np.array([np.nan, np.nan])
        result = _forward_fill(arr)
        np.testing.assert_array_equal(result, [0.0, 0.0])


# ---------------------------------------------------------------------------
# TestPrepareData
# ---------------------------------------------------------------------------


class TestPrepareData:
    def test_shapes(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a, child_b, parent):
        data = prepare_data(hierarchy, tiny_dag, tiny_y, tiny_x)
        assert data.T == 48
        assert set(data.leaf_nodes) == {child_a, child_b}
        assert set(data.root_nodes) == {parent}
        assert data.y[child_a].shape == (48,)
        assert data.y[child_b].shape == (48,)
        assert data.y[parent].shape == (48,)

    def test_ratio_scaled(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a):
        data = prepare_data(hierarchy, tiny_dag, tiny_y, tiny_x)
        y_scaled = np.asarray(data.y[child_a])
        # Ratio-scaled values should be centred around 1.0 (median ≈ 1)
        assert np.median(y_scaled) > 0.5
        assert np.median(y_scaled) < 2.0
        # All values should be positive (sales data)
        assert np.all(y_scaled > 0)

    def test_unstandardize_roundtrip(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a):
        data = prepare_data(hierarchy, tiny_dag, tiny_y, tiny_x)
        original = tiny_y[child_a][:48]
        recovered = data.unstandardize(child_a, np.asarray(data.y[child_a]))
        np.testing.assert_allclose(recovered, original, atol=1e-4)

    def test_missing_y_raises(self, hierarchy, tiny_dag, tiny_x, child_a):
        # Only provide one child, missing the other and parent
        with pytest.raises(ValueError, match="missing series"):
            prepare_data(hierarchy, tiny_dag, {child_a: np.zeros(48)}, tiny_x)

    def test_predictor_edges(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a, ext_x1):
        data = prepare_data(hierarchy, tiny_dag, tiny_y, tiny_x)
        edges = data.predictor_edges[child_a]
        assert len(edges) == 1
        assert edges[0].source is ext_x1
        assert edges[0].lag == 1

    def test_children_map(self, hierarchy, tiny_dag, tiny_y, tiny_x, parent, child_a, child_b):
        data = prepare_data(hierarchy, tiny_dag, tiny_y, tiny_x)
        assert parent in data.children_map
        assert set(data.children_map[parent]) == {child_a, child_b}


# ---------------------------------------------------------------------------
# TestHierarchicalForecaster (integration — uses tiny MCMC)
# ---------------------------------------------------------------------------


class TestHierarchicalForecaster:
    @pytest.fixture
    def fitted_model(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a, child_b, parent):
        configs = {
            parent: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_a: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_b: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        model = HierarchicalForecaster(
            hierarchy=hierarchy,
            causal_dag=tiny_dag,
            node_configs=configs,
            reconciliation="bottom_up",
        )
        model.fit(
            y_data=tiny_y,
            x_data=tiny_x,
            num_warmup=5,
            num_samples=5,
            rng_seed=0,
        )
        return model

    def test_fit_returns_self(self, fitted_model):
        assert isinstance(fitted_model, HierarchicalForecaster)

    def test_samples_populated(self, fitted_model):
        assert fitted_model._samples is not None
        assert len(fitted_model._samples) > 0

    def test_forecast_shape(self, fitted_model, child_a, child_b, parent):
        forecasts = fitted_model.forecast(horizon=6)
        assert child_a in forecasts
        assert child_b in forecasts
        # Each should be (num_samples, horizon)
        assert forecasts[child_a].shape == (5, 6)
        assert forecasts[child_b].shape == (5, 6)

    def test_forecast_has_root(self, fitted_model, parent):
        forecasts = fitted_model.forecast(horizon=6)
        assert parent in forecasts
        assert forecasts[parent].shape == (5, 6)

    def test_forecast_before_fit_raises(self, hierarchy, tiny_dag):
        model = HierarchicalForecaster(hierarchy, tiny_dag)
        with pytest.raises(RuntimeError, match="Must call .fit"):
            model.forecast(horizon=6)

    def test_svi_inference(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a, child_b, parent):
        """SVI inference produces forecasts with correct shapes."""
        configs = {
            parent: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_a: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_b: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        model = HierarchicalForecaster(
            hierarchy=hierarchy,
            causal_dag=tiny_dag,
            node_configs=configs,
            reconciliation="bottom_up",
        )
        model.fit(
            y_data=tiny_y,
            x_data=tiny_x,
            inference="svi",
            num_samples=10,
            svi_steps=50,
            svi_lr=0.01,
            rng_seed=0,
        )
        assert model._samples is not None
        forecasts = model.forecast(horizon=6)
        assert child_a in forecasts
        assert child_b in forecasts
        assert parent in forecasts
        assert forecasts[child_a].shape == (10, 6)
        assert np.all(np.isfinite(forecasts[child_a]))

    def test_no_hierarchy(self, tiny_dag, child_a, child_b, rng, ext_x1):
        """Model works without hierarchy (independent nodes)."""
        empty_hierarchy = DependencyGraph()
        T = 48
        y = {
            child_a: rng.normal(100, 10, T),
            child_b: rng.normal(50, 5, T),
        }
        x = {ext_x1: np.cumsum(rng.normal(0, 0.1, T)) + 100}

        # Need edges pointing to our nodes
        dag = CausalDAG()
        dag.add_edge(ext_x1, child_a, lag=1)

        model = HierarchicalForecaster(empty_hierarchy, dag)
        model.fit(y_data=y, x_data=x, num_warmup=5, num_samples=5)
        forecasts = model.forecast(horizon=3)
        assert child_a in forecasts
        assert forecasts[child_a].shape == (5, 3)


# ---------------------------------------------------------------------------
# TestComponentBasedConfig (new tests for composable components)
# ---------------------------------------------------------------------------


class TestComponentBasedConfig:
    def test_explicit_ou_components(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a, child_b, parent):
        """Test fitting with explicit OUMeanReversion component."""
        configs = {
            parent: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
            child_a: NodeConfig(
                mode="active",
                components=(OUMeanReversion(), FourierSeasonality(n_harmonics=1)),
            ),
            child_b: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        model = HierarchicalForecaster(
            hierarchy=hierarchy,
            causal_dag=tiny_dag,
            node_configs=configs,
            reconciliation="bottom_up",
        )
        model.fit(
            y_data=tiny_y,
            x_data=tiny_x,
            num_warmup=5,
            num_samples=5,
            rng_seed=0,
        )
        forecasts = model.forecast(horizon=6)
        assert child_a in forecasts
        assert forecasts[child_a].shape == (5, 6)
        assert child_b in forecasts
        assert parent in forecasts

    def test_mixed_components_per_node(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a, child_b, parent):
        """One node OU, another LLT."""
        configs = {
            parent: NodeConfig(mode="active"),
            child_a: NodeConfig(
                components=(OUMeanReversion(),),
            ),
            child_b: NodeConfig(
                components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1)),
            ),
        }
        model = HierarchicalForecaster(
            hierarchy=hierarchy,
            causal_dag=tiny_dag,
            node_configs=configs,
            reconciliation="bottom_up",
        )
        model.fit(
            y_data=tiny_y,
            x_data=tiny_x,
            num_warmup=5,
            num_samples=5,
            rng_seed=0,
        )
        forecasts = model.forecast(horizon=4)
        assert forecasts[child_a].shape == (5, 4)
        assert forecasts[child_b].shape == (5, 4)

    def test_damped_via_components(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a, child_b, parent):
        """Test DampedLocalLinearTrend via explicit components."""
        configs = {
            parent: NodeConfig(mode="active"),
            child_a: NodeConfig(
                components=(DampedLocalLinearTrend(),),
            ),
            child_b: NodeConfig(mode="active"),
        }
        model = HierarchicalForecaster(
            hierarchy=hierarchy,
            causal_dag=tiny_dag,
            node_configs=configs,
            reconciliation="bottom_up",
        )
        model.fit(
            y_data=tiny_y,
            x_data=tiny_x,
            num_warmup=5,
            num_samples=5,
            rng_seed=0,
        )
        forecasts = model.forecast(horizon=3)
        assert forecasts[child_a].shape == (5, 3)

    def test_log_additive_aggregator(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a, child_b, parent):
        """Test LogAdditiveAggregator: components combine in log-space, result is exponentiated."""
        configs = {
            parent: NodeConfig(mode="active"),
            child_a: NodeConfig(
                components=(LocalLinearTrend(),),
                aggregator=LogAdditiveAggregator(),
            ),
            child_b: NodeConfig(mode="active"),
        }
        model = HierarchicalForecaster(
            hierarchy=hierarchy,
            causal_dag=tiny_dag,
            node_configs=configs,
            reconciliation="bottom_up",
        )
        model.fit(
            y_data=tiny_y,
            x_data=tiny_x,
            num_warmup=5,
            num_samples=5,
            rng_seed=0,
        )
        forecasts = model.forecast(horizon=3)
        assert forecasts[child_a].shape == (5, 3)
        # Log-additive output should be finite (no NaN/Inf)
        assert np.all(np.isfinite(forecasts[child_a]))

    def test_multiplicative_seasonality(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a, child_b, parent):
        """Test (trend + external) * seasonality via MultiplicativeSeasonality."""
        configs = {
            parent: NodeConfig(mode="active"),
            child_a: NodeConfig(
                components=(
                    LocalLinearTrend(),
                    FourierSeasonality(n_harmonics=2),
                ),
                aggregator=MultiplicativeSeasonality(),
            ),
            child_b: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        model = HierarchicalForecaster(
            hierarchy=hierarchy,
            causal_dag=tiny_dag,
            node_configs=configs,
            reconciliation="bottom_up",
        )
        model.fit(
            y_data=tiny_y,
            x_data=tiny_x,
            num_warmup=5,
            num_samples=5,
            rng_seed=0,
        )
        forecasts = model.forecast(horizon=6)
        assert forecasts[child_a].shape == (5, 6)
        assert np.all(np.isfinite(forecasts[child_a]))
        assert forecasts[child_b].shape == (5, 6)
        assert parent in forecasts

    def test_multiplicative_fourier_seasonality(self, hierarchy, tiny_dag, tiny_y, tiny_x, child_a, child_b, parent):
        """Test (trend + exogenous) * multiplicative_fourier_seasonality."""
        configs = {
            parent: NodeConfig(mode="active"),
            child_a: NodeConfig(
                components=(
                    LocalLinearTrend(),
                    MultiplicativeFourierSeasonality(n_harmonics=2, period=12),
                ),
                aggregator=MultiplicativeSeasonality(),
            ),
            child_b: NodeConfig(mode="active", components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=1))),
        }
        model = HierarchicalForecaster(
            hierarchy=hierarchy,
            causal_dag=tiny_dag,
            node_configs=configs,
            reconciliation="bottom_up",
        )
        model.fit(
            y_data=tiny_y,
            x_data=tiny_x,
            num_warmup=5,
            num_samples=5,
            rng_seed=0,
        )
        forecasts = model.forecast(horizon=6)
        assert forecasts[child_a].shape == (5, 6)
        assert np.all(np.isfinite(forecasts[child_a]))
        assert forecasts[child_b].shape == (5, 6)
        assert parent in forecasts
