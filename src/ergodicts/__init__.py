"""Ergodic TS — tools for time-series forecasting."""

__version__ = "0.1.0"

from ergodicts.snowflake_client import SnowflakeClient, TableExistsError
from ergodicts.utils import date_to_quarter_string, quarter_string_to_date
from ergodicts.reducer import (
    ModelKey,
    DependencyGraph,
    ReducerConfig,
    ReducerResult,
    ReducerPipeline,
    PipelineResult,
    apply_reducer,
    check_harmonization,
)
from ergodicts.causal_dag import (
    ExternalNode,
    EdgeSpec,
    CausalDAG,
    NodeConfig,
)

from ergodicts.components import (
    Aggregator,
    AdditiveAggregator,
    MultiplicativeAggregator,
    LogAdditiveAggregator,
    MultiplicativeSeasonality,
    TrendComponent,
    SeasonalityComponent,
    RegressionComponent,
    LocalLinearTrend,
    DampedLocalLinearTrend,
    OUMeanReversion,
    FourierSeasonality,
    MultiplicativeFourierSeasonality,
    MultiplicativeMonthlySeasonality,
    MonthlySeasonality,
    ExternalRegression,
    ComponentLibrary,
    component_role,
    resolve_aggregator,
    resolve_components,
)
from ergodicts.forecaster import HierarchicalForecaster, ForecastData
from ergodicts.backtester import Backtester, BacktestResult, BacktestSummary


def snowflake_client(**kwargs) -> SnowflakeClient:
    """Create a Snowflake client. Params default to env vars / .env file."""
    return SnowflakeClient(**kwargs)
