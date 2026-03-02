# Components

Composable dynamics components for structural time-series models.  Each component
contributes a term to the predicted mean; the aggregation strategy controls how
contributions are combined.

## Aggregators

::: ergodicts.components.Aggregator

::: ergodicts.components.AdditiveAggregator

::: ergodicts.components.MultiplicativeAggregator

::: ergodicts.components.LogAdditiveAggregator

::: ergodicts.components.MultiplicativeSeasonality

## Trend components

::: ergodicts.components.TrendComponent

::: ergodicts.components.LocalLinearTrend

::: ergodicts.components.DampedLocalLinearTrend

::: ergodicts.components.OUMeanReversion

## Seasonality components

::: ergodicts.components.SeasonalityComponent

::: ergodicts.components.FourierSeasonality

::: ergodicts.components.MultiplicativeFourierSeasonality

::: ergodicts.components.MonthlySeasonality

::: ergodicts.components.MultiplicativeMonthlySeasonality

## Regression components

::: ergodicts.components.RegressionComponent

::: ergodicts.components.ExternalRegression

## Component library

::: ergodicts.components.ComponentLibrary

## Helpers

::: ergodicts.components.resolve_components

::: ergodicts.components.resolve_aggregator

::: ergodicts.components.component_role
