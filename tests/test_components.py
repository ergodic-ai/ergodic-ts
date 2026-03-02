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
    BassTrend,
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
        assert component_role(BassTrend()) == "trend"

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


class TestBassTrend:
    def test_sample_params_keys(self):
        comp = BassTrend()

        def model():
            params = comp.sample_params("test")
            assert "p" in params
            assert "q" in params
            assert "M" in params

        with numpyro.handlers.seed(rng_seed=0):
            model()

    def test_sample_innovations_shape(self):
        comp = BassTrend()

        def model():
            inn = comp.sample_innovations("test", 24)
            assert inn["inn"].shape == (24,)
            assert inn["obs_cumulative"].shape == (24,)

        with numpyro.handlers.seed(rng_seed=0):
            model()

    def test_init_state_shape(self):
        comp = BassTrend()
        params = {"p": jnp.array(0.05), "q": jnp.array(0.3), "M": jnp.array(1.0)}
        state = comp.init_state(params)
        assert state.shape == (2,)
        assert state[0] == 0.0
        assert state[1] == 0.0

    def test_transition_fn(self):
        comp = BassTrend()
        carry = jnp.array([0.0, 0.0])
        p, q, M = 0.05, 0.3, 100.0
        S_prev = 10.0
        inn = {"inn": jnp.array(0.0), "obs_cumulative": jnp.array(S_prev)}
        params = {"p": jnp.array(p), "q": jnp.array(q), "M": jnp.array(M)}
        new_carry, incremental = comp.transition_fn(carry, inn, params)
        # incremental = (M - S_prev) * (p + q * S_prev / M)
        expected = (100.0 - 10.0) * (0.05 + 0.3 * 10.0 / 100.0)
        np.testing.assert_allclose(float(incremental), expected, atol=1e-5)

    def test_transition_zero_before_launch(self):
        comp = BassTrend()
        carry = jnp.array([0.0, 0.0])
        inn = {"inn": jnp.array(0.0), "obs_cumulative": jnp.array(0.0)}
        params = {"p": jnp.array(0.01), "q": jnp.array(0.3), "M": jnp.array(1.0)}
        _, incremental = comp.transition_fn(carry, inn, params)
        # S=0 → incremental = M * p = 0.01 (small)
        np.testing.assert_allclose(float(incremental), 0.01, atol=1e-5)

    def test_transition_saturation(self):
        comp = BassTrend()
        carry = jnp.array([99.9, 0.0])
        inn = {"inn": jnp.array(0.0), "obs_cumulative": jnp.array(99.9)}
        params = {"p": jnp.array(0.05), "q": jnp.array(0.3), "M": jnp.array(100.0)}
        _, incremental = comp.transition_fn(carry, inn, params)
        # remaining ≈ 0.1, so incremental is very small
        assert float(incremental) < 0.1

    def test_forecast_from_state_shape(self):
        comp = BassTrend()
        state = jnp.array([0.5, 0.1])
        params = {"p": jnp.array(0.05), "q": jnp.array(0.3), "M": jnp.array(1.0)}
        levels = comp.forecast_from_state(state, params, 12, jr.PRNGKey(0))
        assert levels.shape == (12,)

    def test_forecast_autoregressive_cumulates(self):
        comp = BassTrend()
        state = jnp.array([0.0, 0.0])
        params = {"p": jnp.array(0.05), "q": jnp.array(0.3), "M": jnp.array(10.0)}
        levels = comp.forecast_from_state(state, params, 30, jr.PRNGKey(42))
        # Cumulative sum of incremental should grow
        cum = jnp.cumsum(levels)
        assert float(cum[-1]) > float(cum[0])
        # Should not exceed M
        assert float(cum[-1]) <= 10.0 + 0.5  # small noise tolerance

    def test_state_dim(self):
        assert BassTrend().state_dim == 2

    def test_serialization_roundtrip(self):
        comp = BassTrend()
        d = comp.to_dict()
        assert d["type"] == "bass_diffusion"
        assert d["params"]["p_prior"] == (1.0, 19.0)
        assert d["params"]["q_prior"] == (2.0, 5.0)
        assert d["params"]["M_prior"] == (0.0, 1.0)
        comp2 = TrendComponent.from_dict(d)
        assert isinstance(comp2, BassTrend)
        assert comp2.p_prior == (1.0, 19.0)

    def test_serialization_roundtrip_custom_priors(self):
        comp = BassTrend(p_prior=(3.0, 57.0), q_prior=(10.0, 20.0), M_prior=(1.5, 0.3))
        d = comp.to_dict()
        comp2 = TrendComponent.from_dict(d)
        assert isinstance(comp2, BassTrend)
        assert comp2.p_prior == (3.0, 57.0)
        assert comp2.q_prior == (10.0, 20.0)
        assert comp2.M_prior == (1.5, 0.3)

    def test_custom_priors_affect_sampling(self):
        # Tight prior on p centered at 0.5 (very different from default 0.05)
        comp = BassTrend(p_prior=(50.0, 50.0))  # Beta(50,50) → mean=0.5, tight

        def model():
            params = comp.sample_params("test")
            return params["p"]

        with numpyro.handlers.seed(rng_seed=0):
            p_val = model()
        # Should be near 0.5, not 0.05
        assert float(p_val) > 0.3

    def test_from_posterior(self):
        # Simulate posterior samples
        rng = np.random.default_rng(42)
        samples = {
            "bass_p_GEN2": rng.beta(2, 38, size=100),   # mean ~0.05
            "bass_q_GEN2": rng.beta(7, 14, size=100),   # mean ~0.33
            "bass_M_GEN2": rng.lognormal(0.5, 0.3, size=100),
        }
        comp = BassTrend.from_posterior(samples, "GEN2")
        assert isinstance(comp, BassTrend)
        # p prior should be centered near 0.05
        p_mean = comp.p_prior[0] / (comp.p_prior[0] + comp.p_prior[1])
        assert 0.02 < p_mean < 0.10
        # q prior should be centered near 0.33
        q_mean = comp.q_prior[0] / (comp.q_prior[0] + comp.q_prior[1])
        assert 0.20 < q_mean < 0.50

    def test_from_posterior_with_M_override(self):
        rng = np.random.default_rng(0)
        samples = {
            "bass_p_old": rng.beta(2, 38, size=50),
            "bass_q_old": rng.beta(7, 14, size=50),
            "bass_M_old": rng.lognormal(0, 0.5, size=50),
        }
        comp = BassTrend.from_posterior(samples, "old", M_prior=(2.0, 0.2))
        assert comp.M_prior == (2.0, 0.2)

    def test_beta_moment_match(self):
        alpha, beta = BassTrend._beta_moment_match(np.array([0.04, 0.05, 0.06, 0.05, 0.04]))
        mean = alpha / (alpha + beta)
        assert 0.04 < mean < 0.06

    def test_lognormal_moment_match(self):
        mu, sigma = BassTrend._lognormal_moment_match(np.array([1.0, 1.1, 0.9, 1.05, 0.95]))
        # mu should be near log(1.0) ≈ 0
        assert -0.5 < mu < 0.5
        assert sigma > 0

    def test_inject_data(self):
        comp = BassTrend()
        y = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = comp.inject_data(y, T=4, horizon=0)
        assert "obs_cumulative" in result
        # obs_cumulative[0] = 0, obs_cumulative[1] = cumsum(y)[0] = 1, etc.
        expected = jnp.array([0.0, 1.0, 3.0, 6.0])
        np.testing.assert_allclose(np.asarray(result["obs_cumulative"]), np.asarray(expected), atol=1e-5)

    def test_inject_data_with_horizon(self):
        comp = BassTrend()
        y = jnp.array([1.0, 2.0])
        result = comp.inject_data(y, T=2, horizon=3)
        assert result["obs_cumulative"].shape == (5,)
        # Last 3 should be zeros (forecast period)
        np.testing.assert_allclose(np.asarray(result["obs_cumulative"][2:]), [0.0, 0.0, 0.0])


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
        assert "bass_diffusion" in reg
        assert reg["local_linear_trend"] is LocalLinearTrend
        assert reg["bass_diffusion"] is BassTrend

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


# ---------------------------------------------------------------------------
# Batched scan tests
# ---------------------------------------------------------------------------


class TestBatchedScan:
    """Test that batched_scan produces results matching sequential scan."""

    def _run_batched_vs_sequential(self, comp_cls, make_params, make_innovations, make_init):
        """Generic helper: compare batched scan output to per-node sequential scan."""
        N = 3
        total_T = 24
        group = []
        for i in range(N):

            def model(i=i):
                params = make_params(f"node_{i}")
                innovations = make_innovations(f"node_{i}", total_T)
                init = make_init(params)
                return params, innovations, init

            with numpyro.handlers.seed(rng_seed=i):
                params, innovations, init = model()
            group.append((i, f"node_{i}", params, innovations, init))

        comp = comp_cls()
        batched_results = comp.batched_scan(group, total_T)

        # Verify each node matches a sequential scan
        for g in group:
            idx, node_key, params, all_innovations, init = g
            b_carry, b_levels = batched_results[idx]

            # Sequential scan
            def scan_fn(carry, t_idx):
                inn_t = {k: v[t_idx] for k, v in all_innovations.items()}
                new_carry, level_t = comp.transition_fn(carry, inn_t, params)
                return new_carry, level_t

            s_carry, s_levels = jax.lax.scan(scan_fn, init, jnp.arange(total_T))

            np.testing.assert_allclose(
                np.asarray(b_carry), np.asarray(s_carry), atol=1e-4,
                err_msg=f"Carry mismatch for node {idx}",
            )
            np.testing.assert_allclose(
                np.asarray(b_levels), np.asarray(s_levels), atol=1e-4,
                err_msg=f"Levels mismatch for node {idx}",
            )

    def test_llt_batched_matches_sequential(self):
        def make_params(key):
            return {
                "level_sigma": jnp.array(0.1),
                "slope_sigma": jnp.array(0.01),
                "level_init": jnp.array(0.0),
                "slope_init": jnp.array(0.05),
            }
        def make_innovations(key, T):
            k = jr.PRNGKey(hash(key) % 2**31)
            k1, k2 = jr.split(k)
            return {"lev_inn": jr.normal(k1, (T,)), "slp_inn": jr.normal(k2, (T,))}
        def make_init(params):
            return jnp.array([params["level_init"], params["slope_init"]])

        self._run_batched_vs_sequential(
            LocalLinearTrend, make_params, make_innovations, make_init,
        )

    def test_dllt_batched_matches_sequential(self):
        def make_params(key):
            return {
                "level_sigma": jnp.array(0.1),
                "slope_sigma": jnp.array(0.01),
                "level_init": jnp.array(0.0),
                "slope_init": jnp.array(0.05),
                "phi": jnp.array(0.9),
            }
        def make_innovations(key, T):
            k = jr.PRNGKey(hash(key) % 2**31)
            k1, k2 = jr.split(k)
            return {"lev_inn": jr.normal(k1, (T,)), "slp_inn": jr.normal(k2, (T,))}
        def make_init(params):
            return jnp.array([params["level_init"], params["slope_init"]])

        self._run_batched_vs_sequential(
            DampedLocalLinearTrend, make_params, make_innovations, make_init,
        )

    def test_ou_batched_matches_sequential(self):
        def make_params(key):
            return {
                "theta": jnp.array(0.3),
                "mu": jnp.array(0.0),
                "sigma": jnp.array(0.1),
                "level_init": jnp.array(1.0),
            }
        def make_innovations(key, T):
            k = jr.PRNGKey(hash(key) % 2**31)
            return {"inn": jr.normal(k, (T,))}
        def make_init(params):
            return jnp.array([params["level_init"]])

        self._run_batched_vs_sequential(
            OUMeanReversion, make_params, make_innovations, make_init,
        )


# ---------------------------------------------------------------------------
# Auto-regression via resolve_components
# ---------------------------------------------------------------------------


class TestResolveComponentsAutoRegression:
    def test_auto_regression_with_edges(self):
        """When predictor_edges are given, ExternalRegression is auto-appended."""
        cfg = NodeConfig()
        comps = resolve_components(cfg, predictor_edges=["edge1"])
        assert any(isinstance(c, ExternalRegression) for c in comps)
        assert len(comps) == 2  # LLT + ExternalRegression

    def test_no_double_regression(self):
        """If RegressionComponent already present, don't add another."""
        cfg = NodeConfig(components=(LocalLinearTrend(), ExternalRegression()))
        comps = resolve_components(cfg, predictor_edges=["edge1"])
        reg_count = sum(1 for c in comps if isinstance(c, ExternalRegression))
        assert reg_count == 1

    def test_backward_compatible(self):
        """No edges → no regression added (backward compatibility)."""
        cfg = NodeConfig()
        comps = resolve_components(cfg)
        assert len(comps) == 1
        assert isinstance(comps[0], LocalLinearTrend)

    def test_no_regression_with_empty_edges(self):
        """Empty list → no regression added."""
        cfg = NodeConfig()
        comps = resolve_components(cfg, predictor_edges=[])
        assert len(comps) == 1
        assert isinstance(comps[0], LocalLinearTrend)
