"""Tests for ergodicts.reducer."""

import numpy as np
import pandas as pd
import pytest

from ergodicts.reducer import (
    DependencyGraph,
    HarmonizationError,
    ModelKey,
    PipelineResult,
    ReducerConfig,
    ReducerPipeline,
    ReducerResult,
    apply_reducer,
    check_harmonization,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small deterministic dataset: 3 dates x 2 cities x 2 SKUs x 2 SL1s."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    records = []
    for date in dates:
        for city in ["NYC", "LA"]:
            for sku in ["A100", "B200"]:
                for sl1 in ["ClassX", "ClassY"]:
                    records.append({
                        "month_end_date": date,
                        "CITY": city,
                        "SKU": sku,
                        "SL1": sl1,
                        "QTY": int(rng.integers(10, 100)),
                    })
    return pd.DataFrame(records)


@pytest.fixture()
def city_sku_config() -> ReducerConfig:
    return ReducerConfig(
        parent_dimensions=["CITY"],
        child_dimensions=["CITY", "SKU"],
    )


# ---------------------------------------------------------------------------
# ModelKey
# ---------------------------------------------------------------------------


class TestModelKey:

    def test_basic(self):
        key = ModelKey(dimensions=("CITY",), values=("NYC",))
        assert str(key) == "NYC"
        assert key.label == "CITY=NYC"

    def test_multi_dim(self):
        key = ModelKey(dimensions=("CITY", "SKU"), values=("NYC", "A100"))
        assert str(key) == "NYC@A100"
        assert key.label == "CITY=NYC, SKU=A100"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            ModelKey(dimensions=("CITY",), values=("NYC", "extra"))

    def test_frozen(self):
        key = ModelKey(dimensions=("CITY",), values=("NYC",))
        with pytest.raises(AttributeError):
            key.dimensions = ("OTHER",)  # type: ignore[misc]

    def test_hashable(self):
        k1 = ModelKey(dimensions=("CITY",), values=("NYC",))
        k2 = ModelKey(dimensions=("CITY",), values=("NYC",))
        assert k1 == k2
        assert len({k1, k2}) == 1

    def test_project(self):
        key = ModelKey(dimensions=("CITY", "SKU"), values=("NYC", "A100"))
        projected = key.project(("CITY",))
        assert projected == ModelKey(dimensions=("CITY",), values=("NYC",))

    def test_project_missing_dimension(self):
        key = ModelKey(dimensions=("CITY",), values=("NYC",))
        with pytest.raises(KeyError):
            key.project(("COUNTRY",))


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------


class TestDependencyGraph:

    def _make_key(self, dims, vals):
        return ModelKey(dimensions=tuple(dims), values=tuple(vals))

    def test_add_and_query(self):
        g = DependencyGraph()
        parent = self._make_key(["CITY"], ["NYC"])
        child1 = self._make_key(["CITY", "SKU"], ["NYC", "A100"])
        child2 = self._make_key(["CITY", "SKU"], ["NYC", "B200"])

        g.add(parent, child1)
        g.add(parent, child2)

        assert g.children_of(parent) == {child1, child2}
        assert g.parents_of(child1) == {parent}
        assert len(g) == 2

    def test_merge(self):
        g1 = DependencyGraph()
        g2 = DependencyGraph()
        p = self._make_key(["CITY"], ["NYC"])
        c1 = self._make_key(["CITY", "SKU"], ["NYC", "A100"])
        c2 = self._make_key(["CITY", "SL1"], ["NYC", "ClassX"])

        g1.add(p, c1)
        g2.add(p, c2)

        merged = g1.merge(g2)
        assert merged.children_of(p) == {c1, c2}
        # originals are untouched
        assert g1.children_of(p) == {c1}
        assert g2.children_of(p) == {c2}

    def test_all_parents_and_children(self):
        g = DependencyGraph()
        p = self._make_key(["CITY"], ["NYC"])
        c = self._make_key(["CITY", "SKU"], ["NYC", "A100"])
        g.add(p, c)
        assert p in g.all_parents
        assert c in g.all_children

    def test_orphans(self):
        g = DependencyGraph()
        p = self._make_key(["CITY"], ["NYC"])
        c = self._make_key(["CITY", "SKU"], ["NYC", "A100"])
        orphan = self._make_key(["CITY", "SKU"], ["LA", "Z999"])
        g.add(p, c)
        assert g.orphans({p, c, orphan}) == {orphan}

    def test_empty_graph(self):
        g = DependencyGraph()
        assert len(g) == 0
        assert g.all_parents == set()
        assert g.all_children == set()


# ---------------------------------------------------------------------------
# ReducerConfig
# ---------------------------------------------------------------------------


class TestReducerConfig:

    def test_valid(self):
        cfg = ReducerConfig(
            parent_dimensions=["CITY"],
            child_dimensions=["CITY", "SKU"],
        )
        assert cfg.parent_dimensions == ["CITY"]

    def test_parent_not_subset_raises(self):
        with pytest.raises(ValueError, match="strict subset"):
            ReducerConfig(
                parent_dimensions=["COUNTRY"],
                child_dimensions=["CITY", "SKU"],
            )

    def test_equal_dimensions_raises(self):
        with pytest.raises(ValueError, match="strict subset"):
            ReducerConfig(
                parent_dimensions=["CITY"],
                child_dimensions=["CITY"],
            )

    def test_custom_columns(self):
        cfg = ReducerConfig(
            parent_dimensions=["CITY"],
            child_dimensions=["CITY", "SKU"],
            date_col="ds",
            target_col="y",
        )
        assert cfg.date_col == "ds"
        assert cfg.target_col == "y"


# ---------------------------------------------------------------------------
# apply_reducer
# ---------------------------------------------------------------------------


class TestApplyReducer:

    def test_returns_reducer_result(self, sample_df, city_sku_config):
        result = apply_reducer(sample_df, city_sku_config)
        assert isinstance(result, ReducerResult)

    def test_parent_data_shape(self, sample_df, city_sku_config):
        result = apply_reducer(sample_df, city_sku_config)
        # 3 dates x 2 cities = 6 parent rows
        assert len(result.parent_data) == 6
        assert set(result.parent_data.columns) == {"date", "value", "__model_key"}

    def test_child_data_shape(self, sample_df, city_sku_config):
        result = apply_reducer(sample_df, city_sku_config)
        # 3 dates x 2 cities x 2 SKUs = 12 child rows
        assert len(result.child_data) == 12
        assert "__parent_key" in result.child_data.columns

    def test_level_names(self, sample_df, city_sku_config):
        result = apply_reducer(sample_df, city_sku_config)
        assert result.parent_level_name == "CITY"
        assert result.child_level_name == "CITY@SKU"

    def test_dependency_graph_populated(self, sample_df, city_sku_config):
        result = apply_reducer(sample_df, city_sku_config)
        graph = result.dependencies
        # 2 parents (NYC, LA), each with 2 children
        assert len(graph.all_parents) == 2
        for parent in graph.all_parents:
            assert len(graph.children_of(parent)) == 2

    def test_values_are_float(self, sample_df, city_sku_config):
        result = apply_reducer(sample_df, city_sku_config)
        assert result.parent_data["value"].dtype == float
        assert result.child_data["value"].dtype == float

    def test_missing_column_raises(self, sample_df):
        config = ReducerConfig(
            parent_dimensions=["CITY"],
            child_dimensions=["CITY", "MISSING_COL"],
        )
        with pytest.raises(ValueError, match="missing required columns"):
            apply_reducer(sample_df, config)

    def test_parent_equals_sum_of_children(self, sample_df, city_sku_config):
        """Core invariant: parent value == sum of child values for each date."""
        result = apply_reducer(sample_df, city_sku_config)
        child_sums = (
            result.child_data
            .groupby(["__parent_key", "date"])["value"]
            .sum()
            .reset_index()
        )
        parent = result.parent_data.rename(columns={"__model_key": "__parent_key"})
        merged = child_sums.merge(parent, on=["__parent_key", "date"], suffixes=("_child", "_parent"))
        assert (merged["value_child"] == merged["value_parent"]).all()

    def test_config_preserved(self, sample_df, city_sku_config):
        result = apply_reducer(sample_df, city_sku_config)
        assert result.config is city_sku_config

    def test_custom_columns(self, sample_df):
        renamed = sample_df.rename(columns={"month_end_date": "ds", "QTY": "y"})
        config = ReducerConfig(
            parent_dimensions=["CITY"],
            child_dimensions=["CITY", "SKU"],
            date_col="ds",
            target_col="y",
        )
        result = apply_reducer(renamed, config)
        assert len(result.parent_data) == 6


# ---------------------------------------------------------------------------
# check_harmonization
# ---------------------------------------------------------------------------


class TestHarmonization:

    def test_clean_data_no_errors(self, sample_df, city_sku_config):
        result = apply_reducer(sample_df, city_sku_config)
        errors = check_harmonization(result)
        assert errors == []

    def test_detects_mismatch(self, sample_df, city_sku_config):
        result = apply_reducer(sample_df, city_sku_config)
        # Corrupt one parent value
        result.parent_data.iloc[0, result.parent_data.columns.get_loc("value")] += 999.0
        errors = check_harmonization(result)
        assert len(errors) == 1
        assert isinstance(errors[0], HarmonizationError)
        assert errors[0].diff == pytest.approx(-999.0)

    def test_atol_parameter(self, sample_df, city_sku_config):
        result = apply_reducer(sample_df, city_sku_config)
        result.parent_data.iloc[0, result.parent_data.columns.get_loc("value")] += 0.001
        # With tight tolerance, should catch it
        assert len(check_harmonization(result, atol=1e-8)) == 1
        # With loose tolerance, should pass
        assert len(check_harmonization(result, atol=0.01)) == 0


# ---------------------------------------------------------------------------
# ReducerPipeline
# ---------------------------------------------------------------------------


class TestReducerPipeline:

    def test_pipeline_runs(self, sample_df):
        pipeline = ReducerPipeline([
            ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SKU"]),
            ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SL1"]),
        ])
        result = pipeline.run(sample_df)
        assert isinstance(result, PipelineResult)

    def test_datasets_produced(self, sample_df):
        pipeline = ReducerPipeline([
            ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SKU"]),
            ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SL1"]),
        ])
        result = pipeline.run(sample_df)
        assert set(result.datasets.keys()) == {"CITY", "CITY@SKU", "CITY@SL1"}

    def test_shared_parent_deduped(self, sample_df):
        """When two reducers share a parent level, rows should not be duplicated."""
        pipeline = ReducerPipeline([
            ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SKU"]),
            ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SL1"]),
        ])
        result = pipeline.run(sample_df)
        # CITY level: 3 dates x 2 cities = 6, no duplicates
        assert len(result.datasets["CITY"]) == 6

    def test_merged_dependency_graph(self, sample_df):
        pipeline = ReducerPipeline([
            ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SKU"]),
            ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SL1"]),
        ])
        result = pipeline.run(sample_df)
        graph = result.dependencies
        # Each city parent should have children from both reducers
        for parent in graph.all_parents:
            children = graph.children_of(parent)
            child_dims = {c.dimensions for c in children}
            assert ("CITY", "SKU") in child_dims
            assert ("CITY", "SL1") in child_dims

    def test_pipeline_harmonization(self, sample_df):
        pipeline = ReducerPipeline([
            ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SKU"]),
            ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SL1"]),
        ])
        result = pipeline.run(sample_df)
        errors = result.check_harmonization()
        assert errors == []

    def test_single_config_pipeline(self, sample_df):
        pipeline = ReducerPipeline([
            ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SKU"]),
        ])
        result = pipeline.run(sample_df)
        assert len(result.results) == 1
        assert set(result.datasets.keys()) == {"CITY", "CITY@SKU"}
