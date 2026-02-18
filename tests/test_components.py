"""Tests for the components module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS

from ergodicts.causal_dag import NodeConfig
from ergodicts.components import (
    AdditiveAggregator,
    DampedLocalLinearTrend,
    ExternalRegression,
    FourierSeasonality,
    LocalLinearTrend,
    LogAdditiveAggregator,
    MonthlySeasonality,
    MultiplicativeAggregator,
    MultiplicativeFourierSeasonality,
    MultiplicativeSeasonality,
    OUMeanReversion,
    component_role,
    resolve_aggregator,
    resolve_components,
)


# ---------------------------------------------------------------------------
# Aggregator tests
# ---------------------------------------------------------------------------


class TestAdditiveAggregator:
    def test_empty_contributions(self):
        agg = AdditiveAggregator()
        result = agg.aggregate({}, (5,))
        np.testing.assert_array_equal(np.asarray(result), 0.0)
        assert result.shape == (5,)

    def test_single_role(self):
        agg = AdditiveAggregator()
        contribs = {"trend": [jnp.array([1.0, 2.0])]}
        result = agg.aggregate(contribs, (2,))
        np.testing.assert_allclose(np.asarray(result), [1.0, 2.0])

    def test_multiple_roles(self):
        agg = AdditiveAggregator()
        contribs = {
            "trend": [jnp.array([1.0, 2.0])],
            "seasonality": [jnp.array([3.0, 4.0])],
        }
        result = agg.aggregate(contribs, (2,))
        np.testing.assert_allclose(np.asarray(result), [4.0, 6.0])

    def test_multiple_contributions_per_role(self):
        agg = AdditiveAggregator()
        contribs = {
            "trend": [jnp.array([1.0, 2.0]), jnp.array([0.5, 0.5])],
        }
        result = agg.aggregate(contribs, (2,))
        np.testing.assert_allclose(np.asarray(result), [1.5, 2.5])


class TestMultiplicativeAggregator:
    def test_empty_contributions(self):
        agg = MultiplicativeAggregator()
        result = agg.aggregate({}, (3,))
        np.testing.assert_array_equal(np.asarray(result), 1.0)

    def test_multiple_roles(self):
        agg = MultiplicativeAggregator()
        contribs = {
            "trend": [jnp.array([2.0, 3.0])],
            "seasonality": [jnp.array([4.0, 5.0])],
        }
        result = agg.aggregate(contribs, (2,))
        np.testing.assert_allclose(np.asarray(result), [8.0, 15.0])


class TestLogAdditiveAggregator:
    def test_empty_contributions(self):
        agg = LogAdditiveAggregator()
        result = agg.aggregate({}, (4,))
        # exp(0) = 1
        np.testing.assert_allclose(np.asarray(result), 1.0)

    def test_exponentiates_sum(self):
        agg = LogAdditiveAggregator()
        contribs = {"trend": [jnp.array([0.0, 1.0])]}
        result = agg.aggregate(contribs, (2,))
        np.testing.assert_allclose(np.asarray(result), [1.0, np.e], atol=1e-5)

    def test_roundtrip(self):
        """aggregate(log values) = product of original values."""
        agg = LogAdditiveAggregator()
        contribs = {
            "trend": [jnp.log(jnp.array([2.0, 3.0]))],
            "seasonality": [jnp.log(jnp.array([4.0, 5.0]))],
        }
        result = agg.aggregate(contribs, (2,))
        np.testing.assert_allclose(np.asarray(result), [8.0, 15.0], atol=1e-5)


class TestMultiplicativeSeasonality:
    def test_trend_only(self):
        agg = MultiplicativeSeasonality()
        contribs = {"trend": [jnp.array([10.0, 20.0])]}
        result = agg.aggregate(contribs, (2,))
        # No seasonality → factor = 1, so result = trend
        np.testing.assert_allclose(np.asarray(result), [10.0, 20.0])

    def test_seasonality_only(self):
        agg = MultiplicativeSeasonality()
        contribs = {"seasonality": [jnp.array([0.1, -0.1])]}
        result = agg.aggregate(contribs, (2,))
        # base = 0 (no trend/regression), so result = 0 * factor = 0
        np.testing.assert_allclose(np.asarray(result), [0.0, 0.0])

    def test_trend_plus_seasonality(self):
        agg = MultiplicativeSeasonality()
        contribs = {
            "trend": [jnp.array([10.0, 20.0])],
            "seasonality": [jnp.array([0.1, -0.1])],
        }
        # (10, 20) * (1.1, 0.9) = (11.0, 18.0)
        result = agg.aggregate(contribs, (2,))
        np.testing.assert_allclose(np.asarray(result), [11.0, 18.0])

    def test_trend_regression_seasonality(self):
        agg = MultiplicativeSeasonality()
        contribs = {
            "trend": [jnp.array([10.0, 20.0])],
            "regression": [jnp.array([2.0, 3.0])],
            "seasonality": [jnp.array([0.1, -0.1])],
        }
        # base = (12, 23), factor = (1.1, 0.9) → (13.2, 20.7)
        result = agg.aggregate(contribs, (2,))
        np.testing.assert_allclose(np.asarray(result), [13.2, 20.7])

    def test_multiple_seasonality_components(self):
        agg = MultiplicativeSeasonality()
        contribs = {
            "trend": [jnp.array([100.0])],
            "seasonality": [jnp.array([0.1]), jnp.array([0.2])],
        }
        # factor = (1.1) * (1.2) = 1.32 → 100 * 1.32 = 132
        result = agg.aggregate(contribs, (1,))
        np.testing.assert_allclose(np.asarray(result), [132.0])

    def test_unknown_role_additive(self):
        """Unknown roles default to additive inclusion."""
        agg = MultiplicativeSeasonality()
        contribs = {
            "trend": [jnp.array([10.0])],
            "domain_expertise": [jnp.array([5.0])],
        }
        # base = 10 + 5 = 15, no seasonality → 15
        result = agg.aggregate(contribs, (1,))
        np.testing.assert_allclose(np.asarray(result), [15.0])


class TestComponentRole:
    def test_trend(self):
        assert component_role(LocalLinearTrend()) == "trend"
        assert component_role(DampedLocalLinearTrend()) == "trend"
        assert component_role(OUMeanReversion()) == "trend"

    def test_seasonality(self):
        assert component_role(FourierSeasonality()) == "seasonality"
        assert component_role(MonthlySeasonality()) == "seasonality"

    def test_regression(self):
        assert component_role(ExternalRegression()) == "regression"


class TestResolveAggregator:
    def test_default_is_additive(self):
        cfg = NodeConfig()
        agg = resolve_aggregator(cfg)
        assert isinstance(agg, AdditiveAggregator)

    def test_explicit_multiplicative(self):
        agg = MultiplicativeAggregator()
        cfg = NodeConfig(aggregator=agg)
        assert resolve_aggregator(cfg) is agg

    def test_explicit_log_additive(self):
        agg = LogAdditiveAggregator()
        cfg = NodeConfig(aggregator=agg)
        assert resolve_aggregator(cfg) is agg


# ---------------------------------------------------------------------------
# TrendComponent tests
# ---------------------------------------------------------------------------


class TestLocalLinearTrend:
    def test_sample_params_keys(self):
        comp = LocalLinearTrend()

        def model():
            params = comp.sample_params("test")
            assert "level_sigma" in params
            assert "slope_sigma" in params
            assert "level_init" in params
            assert "slope_init" in params

        with numpyro.handlers.seed(rng_seed=0):
            model()

    def test_sample_innovations_shape(self):
        comp = LocalLinearTrend()

        def model():
            inn = comp.sample_innovations("test", 24)
            assert inn["lev_inn"].shape == (24,)
            assert inn["slp_inn"].shape == (24,)

        with numpyro.handlers.seed(rng_seed=0):
            model()

    def test_init_state_shape(self):
        comp = LocalLinearTrend()
        params = {"level_init": jnp.array(0.0), "slope_init": jnp.array(0.1)}
        state = comp.init_state(params)
        assert state.shape == (2,)
        assert state[0] == 0.0
        assert state[1] == 0.1

    def test_transition_fn(self):
        comp = LocalLinearTrend()
        carry = jnp.array([1.0, 0.1])
        inn = {"lev_inn": jnp.array(0.5), "slp_inn": jnp.array(-0.2)}
        params = {"level_sigma": jnp.array(0.1), "slope_sigma": jnp.array(0.01)}
        new_carry, level = comp.transition_fn(carry, inn, params)
        # level = 1.0 + 0.1 + 0.1 * 0.5 = 1.15
        np.testing.assert_allclose(float(level), 1.15, atol=1e-5)

    def test_forecast_from_state_shape(self):
        comp = LocalLinearTrend()
        state = jnp.array([0.0, 0.1])
        params = {
            "level_sigma": jnp.array(0.1),
            "slope_sigma": jnp.array(0.01),
        }
        levels = comp.forecast_from_state(state, params, 12, jr.PRNGKey(0))
        assert levels.shape == (12,)

    def test_state_dim(self):
        assert LocalLinearTrend().state_dim == 2


class TestDampedLocalLinearTrend:
    def test_transition_damping(self):
        comp = DampedLocalLinearTrend()
        carry = jnp.array([1.0, 0.5])
        inn = {"lev_inn": jnp.array(0.0), "slp_inn": jnp.array(0.0)}
        params = {
            "level_sigma": jnp.array(0.0),
            "slope_sigma": jnp.array(0.0),
            "phi": jnp.array(0.8),
        }
        new_carry, level = comp.transition_fn(carry, inn, params)
        # level = 1.0 + 0.5 = 1.5
        np.testing.assert_allclose(float(level), 1.5, atol=1e-5)
        # new slope = 0.8 * 0.5 = 0.4
        np.testing.assert_allclose(float(new_carry[1]), 0.4, atol=1e-5)

    def test_forecast_shape(self):
        comp = DampedLocalLinearTrend()
        state = jnp.array([0.0, 0.1])
        params = {
            "level_sigma": jnp.array(0.1),
            "slope_sigma": jnp.array(0.01),
            "phi": jnp.array(0.9),
        }
        levels = comp.forecast_from_state(state, params, 6, jr.PRNGKey(42))
        assert levels.shape == (6,)


class TestOUMeanReversion:
    def test_state_dim(self):
        assert OUMeanReversion().state_dim == 1

    def test_transition_mean_reversion(self):
        comp = OUMeanReversion()
        carry = jnp.array([2.0])
        inn = {"inn": jnp.array(0.0)}
        params = {
            "theta": jnp.array(0.5),
            "mu": jnp.array(0.0),
            "sigma": jnp.array(0.0),
        }
        new_carry, level = comp.transition_fn(carry, inn, params)
        # level = 2.0 + 0.5 * (0.0 - 2.0) = 2.0 - 1.0 = 1.0
        np.testing.assert_allclose(float(level), 1.0, atol=1e-5)

    def test_forecast_shape(self):
        comp = OUMeanReversion()
        state = jnp.array([1.0])
        params = {
            "theta": jnp.array(0.3),
            "mu": jnp.array(0.0),
            "sigma": jnp.array(0.1),
            "level_init": jnp.array(1.0),
        }
        levels = comp.forecast_from_state(state, params, 10, jr.PRNGKey(0))
        assert levels.shape == (10,)


# ---------------------------------------------------------------------------
# SeasonalityComponent tests
# ---------------------------------------------------------------------------


class TestFourierSeasonality:
    def test_contribute_shape(self):
        comp = FourierSeasonality(n_harmonics=2, period=12.0)
        params = {"coeffs": jnp.array([1.0, 0.0, 0.5, 0.0])}
        result = comp.contribute(params, 24)
        assert result.shape == (24,)

    def test_period_12(self):
        comp = FourierSeasonality(n_harmonics=1, period=12.0)
        params = {"coeffs": jnp.array([1.0, 0.0])}
        result = comp.contribute(params, 24)
        np.testing.assert_allclose(
            np.asarray(result[:12]),
            np.asarray(result[12:24]),
            atol=1e-5,
        )

    def test_forecast_contribute_shape(self):
        comp = FourierSeasonality(n_harmonics=2, period=12.0)
        params = {"coeffs": jnp.array([1.0, 0.0, 0.5, 0.0])}
        result = comp.forecast_contribute(params, 48, 12)
        assert result.shape == (12,)

    def test_forecast_continue_from_history(self):
        comp = FourierSeasonality(n_harmonics=1, period=12.0)
        params = {"coeffs": jnp.array([1.0, 0.5])}
        full = comp.contribute(params, 60)
        forecast = comp.forecast_contribute(params, 48, 12)
        np.testing.assert_allclose(
            np.asarray(full[48:60]),
            np.asarray(forecast),
            atol=1e-5,
        )


class TestMonthlySeasonality:
    def test_contribute_shape(self):
        comp = MonthlySeasonality()
        effects = jnp.arange(12, dtype=jnp.float32)
        effects = effects - jnp.mean(effects)
        params = {"month_effects": effects, "raw": jnp.arange(12, dtype=jnp.float32)}
        result = comp.contribute(params, 24)
        assert result.shape == (24,)

    def test_period_12(self):
        comp = MonthlySeasonality()
        effects = jnp.arange(12, dtype=jnp.float32)
        effects = effects - jnp.mean(effects)
        params = {"month_effects": effects, "raw": jnp.arange(12, dtype=jnp.float32)}
        result = comp.contribute(params, 24)
        np.testing.assert_allclose(
            np.asarray(result[:12]),
            np.asarray(result[12:24]),
            atol=1e-5,
        )

    def test_forecast_contribute_shape(self):
        comp = MonthlySeasonality()
        effects = jnp.arange(12, dtype=jnp.float32)
        effects = effects - jnp.mean(effects)
        params = {"month_effects": effects, "raw": jnp.arange(12, dtype=jnp.float32)}
        result = comp.forecast_contribute(params, 48, 6)
        assert result.shape == (6,)


class TestMultiplicativeFourierSeasonality:
    def test_contribute_shape(self):
        comp = MultiplicativeFourierSeasonality(n_harmonics=2, period=12)
        params = {"coeffs": jnp.array([1.0, 0.0, 0.5, 0.0])}
        result = comp.contribute(params, 24)
        assert result.shape == (24,)

    def test_weights_sum_to_one(self):
        """Softmax weights over one period sum to 1."""
        comp = MultiplicativeFourierSeasonality(n_harmonics=2, period=12)
        params = {"coeffs": jnp.array([1.0, -0.5, 0.3, 0.8])}
        factors = comp._factors(params["coeffs"])
        # factors = P * softmax(raw), so sum = P and mean = 1
        np.testing.assert_allclose(float(jnp.sum(factors)), 12.0, atol=1e-5)
        np.testing.assert_allclose(float(jnp.mean(factors)), 1.0, atol=1e-5)

    def test_factors_positive(self):
        """All seasonal factors must be positive (softmax guarantees this)."""
        comp = MultiplicativeFourierSeasonality(n_harmonics=3, period=12)
        params = {"coeffs": jnp.array([2.0, -1.0, 0.5, -2.0, 1.5, 0.3])}
        factors = comp._factors(params["coeffs"])
        assert jnp.all(factors > 0)

    def test_output_recovers_factor(self):
        """1 + contribute(...) should equal the normalised factor."""
        comp = MultiplicativeFourierSeasonality(n_harmonics=1, period=12)
        params = {"coeffs": jnp.array([1.0, 0.5])}
        factors = comp._factors(params["coeffs"])      # (12,)
        output = comp.contribute(params, 12)            # factor - 1
        np.testing.assert_allclose(
            np.asarray(1.0 + output), np.asarray(factors), atol=1e-5,
        )

    def test_period_repeats(self):
        comp = MultiplicativeFourierSeasonality(n_harmonics=1, period=12)
        params = {"coeffs": jnp.array([1.0, 0.0])}
        result = comp.contribute(params, 24)
        np.testing.assert_allclose(
            np.asarray(result[:12]),
            np.asarray(result[12:24]),
            atol=1e-5,
        )

    def test_forecast_contribute_shape(self):
        comp = MultiplicativeFourierSeasonality(n_harmonics=2, period=12)
        params = {"coeffs": jnp.array([1.0, 0.0, 0.5, 0.0])}
        result = comp.forecast_contribute(params, 48, 12)
        assert result.shape == (12,)

    def test_forecast_continues_from_history(self):
        comp = MultiplicativeFourierSeasonality(n_harmonics=1, period=12)
        params = {"coeffs": jnp.array([1.0, 0.5])}
        full = comp.contribute(params, 60)
        forecast = comp.forecast_contribute(params, 48, 12)
        np.testing.assert_allclose(
            np.asarray(full[48:60]),
            np.asarray(forecast),
            atol=1e-5,
        )

    def test_component_role(self):
        comp = MultiplicativeFourierSeasonality()
        assert component_role(comp) == "seasonality"


# ---------------------------------------------------------------------------
# RegressionComponent tests
# ---------------------------------------------------------------------------


class TestExternalRegression:
    def test_contribute_shape(self):
        comp = ExternalRegression()
        X = jnp.ones((24, 3))
        params = {"betas": jnp.array([1.0, 2.0, 3.0])}
        result = comp.contribute(params, X)
        assert result.shape == (24,)
        np.testing.assert_allclose(float(result[0]), 6.0, atol=1e-5)

    def test_forecast_contribute(self):
        comp = ExternalRegression()
        X = jnp.ones((6, 2))
        params = {"betas": jnp.array([0.5, -0.5])}
        result = comp.forecast_contribute(params, X)
        assert result.shape == (6,)
        np.testing.assert_allclose(np.asarray(result), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# resolve_components tests
# ---------------------------------------------------------------------------


class TestResolveComponents:
    def test_default_config(self):
        cfg = NodeConfig()
        comps = resolve_components(cfg)
        assert len(comps) == 1
        assert isinstance(comps[0], LocalLinearTrend)

    def test_explicit_components(self):
        ou = OUMeanReversion()
        monthly = MonthlySeasonality()
        cfg = NodeConfig(components=(ou, monthly))
        comps = resolve_components(cfg)
        assert len(comps) == 2
        assert comps[0] is ou
        assert comps[1] is monthly

    def test_explicit_trend_and_seasonality(self):
        cfg = NodeConfig(
            components=(DampedLocalLinearTrend(), FourierSeasonality(n_harmonics=2)),
        )
        comps = resolve_components(cfg)
        assert len(comps) == 2
        assert isinstance(comps[0], DampedLocalLinearTrend)
        assert isinstance(comps[1], FourierSeasonality)
        assert comps[1].n_harmonics == 2


# ---------------------------------------------------------------------------
# Component Registry tests
# ---------------------------------------------------------------------------


from ergodicts.components import (
    Aggregator,
    ComponentLibrary,
    RegressionComponent,
    SeasonalityComponent,
    TrendComponent,
)


class TestRegistry:
    def test_trend_registry(self):
        reg = TrendComponent._registry
        assert "local_linear_trend" in reg
        assert "damped_local_linear_trend" in reg
        assert "ou_mean_reversion" in reg
        assert reg["local_linear_trend"] is LocalLinearTrend

    def test_seasonality_registry(self):
        reg = SeasonalityComponent._registry
        assert "fourier_seasonality" in reg
        assert "multiplicative_fourier_seasonality" in reg
        assert "monthly_seasonality" in reg

    def test_regression_registry(self):
        assert "external_regression" in RegressionComponent._registry

    def test_aggregator_registry(self):
        reg = Aggregator._registry
        assert "additive" in reg
        assert "multiplicative" in reg
        assert "log_additive" in reg
        assert "multiplicative_seasonality" in reg

    def test_user_defined_component(self):
        """Custom components auto-register by subclassing."""
        class _TestTrend(TrendComponent, name="_test_custom"):
            def sample_params(self, node_key): return {}
            def sample_innovations(self, node_key, T): return {}
            def init_state(self, params): return jnp.zeros(2)
            def transition_fn(self, carry, innovations, params): return carry, carry[0]
            def extract_posterior(self, node_key, samples): return {}
            def forecast_from_state(self, final_state, params, horizon, rng_key):
                return jnp.zeros(horizon)

        assert "_test_custom" in TrendComponent._registry
        assert TrendComponent._registry["_test_custom"] is _TestTrend
        # Cleanup
        del TrendComponent._registry["_test_custom"]


class TestSerialization:
    def test_trend_no_params_round_trip(self):
        comp = LocalLinearTrend()
        d = comp.to_dict()
        assert d == {"type": "local_linear_trend", "params": {}}
        comp2 = TrendComponent.from_dict(d)
        assert isinstance(comp2, LocalLinearTrend)

    def test_seasonality_with_params_round_trip(self):
        comp = FourierSeasonality(n_harmonics=3, period=24.0)
        d = comp.to_dict()
        assert d["type"] == "fourier_seasonality"
        assert d["params"]["n_harmonics"] == 3
        assert d["params"]["period"] == 24.0
        comp2 = SeasonalityComponent.from_dict(d)
        assert isinstance(comp2, FourierSeasonality)
        assert comp2.n_harmonics == 3
        assert comp2.period == 24.0

    def test_aggregator_round_trip(self):
        agg = MultiplicativeSeasonality()
        d = agg.to_dict()
        assert d == {"type": "multiplicative_seasonality", "params": {}}
        agg2 = Aggregator.from_dict(d)
        assert isinstance(agg2, MultiplicativeSeasonality)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown TrendComponent"):
            TrendComponent.from_dict({"type": "nonexistent"})

    def test_component_library_from_dict(self):
        d = {"type": "ou_mean_reversion", "params": {}}
        comp = ComponentLibrary.from_dict(d)
        assert isinstance(comp, OUMeanReversion)

    def test_component_library_from_dict_aggregator(self):
        d = {"type": "log_additive", "params": {}}
        agg = ComponentLibrary.from_dict(d)
        assert isinstance(agg, LogAdditiveAggregator)


class TestNodeConfigSerialization:
    def test_round_trip_with_components(self):
        cfg = NodeConfig(
            mode="active",
            components=(DampedLocalLinearTrend(), FourierSeasonality(n_harmonics=2)),
            aggregator=MultiplicativeSeasonality(),
        )
        dumped = cfg.model_dump()
        assert isinstance(dumped["components"], list)
        assert dumped["components"][0]["type"] == "damped_local_linear_trend"
        assert dumped["aggregator"]["type"] == "multiplicative_seasonality"

        cfg2 = NodeConfig(**dumped)
        assert len(cfg2.components) == 2
        assert isinstance(cfg2.components[0], DampedLocalLinearTrend)
        assert isinstance(cfg2.components[1], FourierSeasonality)
        assert isinstance(cfg2.aggregator, MultiplicativeSeasonality)

    def test_round_trip_none_components(self):
        """Default config with components=None round-trips correctly."""
        cfg = NodeConfig()
        dumped = cfg.model_dump()
        assert dumped["components"] is None
        assert dumped["aggregator"] is None
        cfg2 = NodeConfig(**dumped)
        assert cfg2.components is None
        assert cfg2.aggregator is None

    def test_round_trip_preserves_params(self):
        cfg = NodeConfig(
            components=(FourierSeasonality(n_harmonics=5, period=6.0),),
        )
        dumped = cfg.model_dump()
        cfg2 = NodeConfig(**dumped)
        fs = cfg2.components[0]
        assert isinstance(fs, FourierSeasonality)
        assert fs.n_harmonics == 5
        assert fs.period == 6.0


class TestComponentLibrary:
    def test_all_components_returns_all(self):
        lib = ComponentLibrary.all_components()
        assert len(lib) >= 11  # all built-in components
        for name, info in lib.items():
            assert "role" in info
            assert "class_name" in info
            assert "description" in info
            assert "params" in info
            assert info["role"] in ("trend", "seasonality", "regression", "aggregator")

    def test_parameterized_component_has_params(self):
        lib = ComponentLibrary.all_components()
        fs = lib["fourier_seasonality"]
        assert "n_harmonics" in fs["params"]
        assert "period" in fs["params"]
        assert fs["params"]["n_harmonics"]["default"] == 2

    def test_no_param_component_empty_params(self):
        lib = ComponentLibrary.all_components()
        llt = lib["local_linear_trend"]
        assert llt["params"] == {}
