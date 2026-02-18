"""Reducer — aggregate hierarchical time-series across arbitrary dimension combinations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import graphviz

import pandas as pd
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# ModelKey — hashable identifier for a model in the dimension space
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class ModelKey:
    """Identifies a single model by its dimension names and values.

    Examples
    --------
    ```python
    key = ModelKey(dimensions=("CITY", "SKU"), values=("NYC", "A100"))
    str(key)            # 'NYC@A100'
    key.dimensions      # ('CITY', 'SKU')
    key.label           # 'CITY=NYC, SKU=A100'
    ```
    """

    dimensions: tuple[str, ...]
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.dimensions) != len(self.values):
            raise ValueError(
                f"dimensions and values must have the same length: "
                f"{len(self.dimensions)} != {len(self.values)}"
            )

    def project(self, target_dimensions: tuple[str, ...]) -> ModelKey:
        """Return a new ModelKey keeping only *target_dimensions*.

        Raises ``KeyError`` if a target dimension is not present.
        """
        dim_to_val = dict(zip(self.dimensions, self.values))
        try:
            projected_values = tuple(dim_to_val[d] for d in target_dimensions)
        except KeyError as e:
            raise KeyError(
                f"Dimension {e} not found in {self.dimensions}"
            ) from None
        return ModelKey(dimensions=target_dimensions, values=projected_values)

    @property
    def label(self) -> str:
        """Human-readable label: ``dim1=val1, dim2=val2``."""
        return ", ".join(f"{d}={v}" for d, v in zip(self.dimensions, self.values))

    def matches(self, pattern: ModelKey) -> bool:
        """Return ``True`` if this key matches *pattern*.

        A pattern matches when dimensions are identical and every value
        matches, where ``"*"`` in the pattern acts as a wildcard that
        matches any value.
        """
        if self.dimensions != pattern.dimensions:
            return False
        return all(
            pv == "*" or pv == sv
            for sv, pv in zip(self.values, pattern.values)
        )

    @property
    def has_wildcard(self) -> bool:
        """Return ``True`` if any value is ``"*"``."""
        return "*" in self.values

    def __str__(self) -> str:
        return "@".join(str(v) for v in self.values)

    def __repr__(self) -> str:
        return f"ModelKey({self.label})"


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------


class DependencyGraph:
    """Tracks parent → children relationships across reducer outputs.

    Each entry maps a parent :class:`ModelKey` to a set of child
    :class:`ModelKey` objects.
    """

    def __init__(self) -> None:
        self._parent_to_children: dict[ModelKey, set[ModelKey]] = {}
        self._child_to_parents: dict[ModelKey, set[ModelKey]] = {}

    # -- mutation --------------------------------------------------------------

    def add(self, parent: ModelKey, child: ModelKey) -> None:
        self._parent_to_children.setdefault(parent, set()).add(child)
        self._child_to_parents.setdefault(child, set()).add(parent)

    def merge(self, other: DependencyGraph) -> DependencyGraph:
        """Return a **new** graph containing edges from both graphs."""
        merged = DependencyGraph()
        for graph in (self, other):
            for parent, children in graph._parent_to_children.items():
                for child in children:
                    merged.add(parent, child)
        return merged

    # -- queries ---------------------------------------------------------------

    def children_of(self, parent: ModelKey) -> set[ModelKey]:
        return set(self._parent_to_children.get(parent, set()))

    def parents_of(self, child: ModelKey) -> set[ModelKey]:
        return set(self._child_to_parents.get(child, set()))

    @property
    def all_parents(self) -> set[ModelKey]:
        return set(self._parent_to_children.keys())

    @property
    def all_children(self) -> set[ModelKey]:
        return set(self._child_to_parents.keys())

    def orphans(self, all_keys: set[ModelKey]) -> set[ModelKey]:
        """Return keys in *all_keys* that appear neither as parent nor child."""
        known = self.all_parents | self.all_children
        return all_keys - known

    # -- visualization ---------------------------------------------------------

    def show(self) -> graphviz.Digraph:
        """Render the dependency graph as a Graphviz digraph.

        Returns the ``graphviz.Digraph`` object, which displays inline
        in Jupyter notebooks. Call ``.render()`` on it to save to a file.

        Requires the ``graphviz`` Python package (``pip install graphviz``)
        and the Graphviz system binaries.
        """
        try:
            import graphviz as gv
        except ImportError:
            raise ImportError(
                "graphviz is required for .show(). "
                "Install it with: pip install graphviz"
            ) from None

        dot = gv.Digraph(
            "DependencyGraph",
            graph_attr={"rankdir": "TB", "fontsize": "12"},
            node_attr={"shape": "box", "style": "rounded,filled", "fontsize": "10"},
        )

        # Group nodes by their dimension level for visual clustering
        levels: dict[tuple[str, ...], set[ModelKey]] = {}
        for parent in self.all_parents:
            levels.setdefault(parent.dimensions, set()).add(parent)
        for child in self.all_children:
            levels.setdefault(child.dimensions, set()).add(child)

        palette = ["#e8f4f8", "#d4e8d4", "#f5e6cc", "#e8d4e8", "#f5d4d4", "#d4d4e8"]
        for i, (dims, keys) in enumerate(sorted(levels.items(), key=lambda x: len(x[0]))):
            level_name = "@".join(dims)
            color = palette[i % len(palette)]
            with dot.subgraph(name=f"cluster_{level_name}") as sub:
                sub.attr(label=level_name, style="dashed", color="grey")
                for key in sorted(keys):
                    sub.node(str(id(key)), label=key.label, fillcolor=color)

        for parent, children in self._parent_to_children.items():
            for child in sorted(children):
                dot.edge(str(id(parent)), str(id(child)))

        return dot

    # -- dunder ----------------------------------------------------------------

    def __len__(self) -> int:
        return sum(len(ch) for ch in self._parent_to_children.values())

    def __repr__(self) -> str:
        return f"DependencyGraph(parents={len(self._parent_to_children)}, edges={len(self)})"


# ---------------------------------------------------------------------------
# ReducerConfig
# ---------------------------------------------------------------------------


class ReducerConfig(BaseModel):
    """Configuration for a single reduction step.

    ``parent_dimensions`` must be a strict subset of ``child_dimensions``.
    """

    parent_dimensions: list[str]
    child_dimensions: list[str]
    date_col: str = Field("month_end_date", alias="date_column")
    target_col: str = Field("QTY", alias="value_column")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _validate_dimensions(self) -> ReducerConfig:
        parent_set = set(self.parent_dimensions)
        child_set = set(self.child_dimensions)
        if not parent_set < child_set:
            raise ValueError(
                f"parent_dimensions must be a strict subset of child_dimensions. "
                f"parent={self.parent_dimensions}, child={self.child_dimensions}"
            )
        return self


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_series_from_frames(frames: list[pd.DataFrame], key: ModelKey) -> pd.DataFrame:
    """Extract time-series rows matching *key* from a list of DataFrames.

    Returns a DataFrame with ``model_name``, ``date``, ``value`` columns.
    Raises ``KeyError`` if nothing matches.
    """
    parts: list[pd.DataFrame] = []
    for data in frames:
        if "__model_key" not in data.columns:
            continue
        if key.has_wildcard:
            mask = data["__model_key"].apply(lambda k: k.matches(key))
        else:
            mask = data["__model_key"] == key
        if mask.any():
            matched = data.loc[mask].copy()
            matched["model_name"] = matched["__model_key"].apply(str)
            parts.append(matched[["model_name", "date", "value"]])

    if not parts:
        raise KeyError(f"ModelKey not found: {key!r}")

    result = pd.concat(parts, ignore_index=True)
    return result.sort_values(["model_name", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# ReducerResult
# ---------------------------------------------------------------------------


@dataclass
class ReducerResult:
    """Output of a single :func:`apply_reducer` call."""

    parent_data: pd.DataFrame
    child_data: pd.DataFrame
    dependencies: DependencyGraph
    config: ReducerConfig

    @property
    def parent_level_name(self) -> str:
        return "@".join(self.config.parent_dimensions)

    @property
    def child_level_name(self) -> str:
        return "@".join(self.config.child_dimensions)

    def get_series(self, key: ModelKey) -> pd.DataFrame:
        """Return the time-series rows matching *key*.

        Supports wildcards: use ``"*"`` in any value position to match
        all values for that dimension.

        Searches both parent and child data. Returns a DataFrame with
        ``model_name``, ``date`` and ``value`` columns.

        Raises ``KeyError`` if no matching keys are found.

        Examples
        --------
        ```python
        # Exact match
        result.get_series(ModelKey(("CITY",), ("NYC",)))

        # Wildcard — all CITY×SL1 models for NYC
        result.get_series(ModelKey(("CITY", "SL1"), ("NYC", "*")))
        ```
        """
        return _get_series_from_frames(
            [self.parent_data, self.child_data], key
        )


# ---------------------------------------------------------------------------
# apply_reducer
# ---------------------------------------------------------------------------


def apply_reducer(df: pd.DataFrame, config: ReducerConfig) -> ReducerResult:
    """Aggregate *df* from child dimensions to parent dimensions.

    Parameters
    ----------
    df : DataFrame
        Must contain columns for ``config.date_col``, ``config.target_col``,
        and every dimension in ``config.child_dimensions``.
    config : ReducerConfig
        Specifies parent/child dimensions and column names.

    Returns
    -------
    ReducerResult
        Contains aggregated parent/child DataFrames, a DependencyGraph,
        and a reference to the config used.
    """
    date_col = config.date_col
    target_col = config.target_col
    parent_dims = config.parent_dimensions
    child_dims = config.child_dimensions

    # --- validate columns exist -----------------------------------------------
    required_cols = {date_col, target_col} | set(child_dims)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # --- aggregate ------------------------------------------------------------
    parent_data = (
        df.groupby([date_col] + parent_dims, sort=False)[[target_col]]
        .sum()
        .astype(float)
        .reset_index()
    )
    child_data = (
        df.groupby([date_col] + child_dims, sort=False)[[target_col]]
        .sum()
        .astype(float)
        .reset_index()
    )

    # --- build model keys -----------------------------------------------------
    parent_dim_tuple = tuple(parent_dims)
    child_dim_tuple = tuple(child_dims)

    parent_data["__model_key"] = [
        ModelKey(parent_dim_tuple, tuple(str(row[d]) for d in parent_dims))
        for _, row in parent_data.iterrows()
    ]
    child_data["__model_key"] = [
        ModelKey(child_dim_tuple, tuple(str(row[d]) for d in child_dims))
        for _, row in child_data.iterrows()
    ]
    child_data["__parent_key"] = [
        key.project(parent_dim_tuple) for key in child_data["__model_key"]
    ]

    # --- dependency graph -----------------------------------------------------
    deps = DependencyGraph()
    seen: set[tuple[ModelKey, ModelKey]] = set()
    for _, row in child_data.iterrows():
        edge = (row["__parent_key"], row["__model_key"])
        if edge not in seen:
            deps.add(*edge)
            seen.add(edge)

    # --- clean up output frames -----------------------------------------------
    parent_out = parent_data[[date_col, target_col, "__model_key"]].copy()
    parent_out.columns = ["date", "value", "__model_key"]

    child_out = child_data[[date_col, target_col, "__model_key", "__parent_key"]].copy()
    child_out.columns = ["date", "value", "__model_key", "__parent_key"]

    return ReducerResult(
        parent_data=parent_out,
        child_data=child_out,
        dependencies=deps,
        config=config,
    )


# ---------------------------------------------------------------------------
# Harmonization check
# ---------------------------------------------------------------------------


@dataclass
class HarmonizationError:
    """A single date/parent where children don't sum to parent."""

    parent_key: ModelKey
    date: Any
    parent_value: float
    children_sum: float

    @property
    def diff(self) -> float:
        return self.children_sum - self.parent_value

    def __repr__(self) -> str:
        return (
            f"HarmonizationError({self.parent_key.label}, date={self.date}, "
            f"parent={self.parent_value:.4f}, children_sum={self.children_sum:.4f}, "
            f"diff={self.diff:+.4f})"
        )


def check_harmonization(
    result: ReducerResult,
    *,
    atol: float = 1e-8,
) -> list[HarmonizationError]:
    """Verify that children sum to parent for every date.

    Returns an empty list if everything is consistent.
    """
    errors: list[HarmonizationError] = []

    parent_indexed = result.parent_data.set_index(["__model_key", "date"])["value"]

    child_sums = (
        result.child_data
        .groupby(["__parent_key", "date"])["value"]
        .sum()
    )

    for (parent_key, date), child_total in child_sums.items():
        parent_val = parent_indexed.get((parent_key, date))
        if parent_val is None:
            continue
        if abs(child_total - parent_val) > atol:
            errors.append(
                HarmonizationError(
                    parent_key=parent_key,
                    date=date,
                    parent_value=float(parent_val),
                    children_sum=float(child_total),
                )
            )

    return errors


# ---------------------------------------------------------------------------
# ReducerPipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Aggregated output of a :class:`ReducerPipeline`."""

    results: list[ReducerResult] = field(default_factory=list)
    dependencies: DependencyGraph = field(default_factory=DependencyGraph)
    datasets: dict[str, pd.DataFrame] = field(default_factory=dict)

    def get_series(self, key: ModelKey) -> pd.DataFrame:
        """Return the time-series rows matching *key*.

        Supports wildcards: use ``"*"`` in any value position to match
        all values for that dimension.

        Searches all datasets in the pipeline result. Returns a DataFrame
        with ``model_name``, ``date`` and ``value`` columns.

        Raises ``KeyError`` if no matching keys are found.
        """
        frames = [
            data for data in self.datasets.values()
            if "__model_key" in data.columns
        ]
        return _get_series_from_frames(frames, key)

    def check_harmonization(self, *, atol: float = 1e-8) -> list[HarmonizationError]:
        """Run harmonization checks across all reducer results."""
        all_errors: list[HarmonizationError] = []
        for r in self.results:
            all_errors.extend(check_harmonization(r, atol=atol))
        return all_errors


class ReducerPipeline:
    """Run multiple reducers on the same DataFrame and merge results.

    Examples
    --------
    ```python
    pipeline = ReducerPipeline([
        ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SKU"]),
        ReducerConfig(parent_dimensions=["CITY"], child_dimensions=["CITY", "SL1"]),
    ])
    result = pipeline.run(df)
    result.datasets.keys()  # dict_keys(['CITY', 'CITY@SKU', 'CITY@SL1'])
    ```
    """

    def __init__(self, configs: list[ReducerConfig]) -> None:
        self.configs = configs

    def run(self, df: pd.DataFrame) -> PipelineResult:
        pipeline_result = PipelineResult()

        for config in self.configs:
            result = apply_reducer(df, config)
            pipeline_result.results.append(result)
            pipeline_result.dependencies = pipeline_result.dependencies.merge(
                result.dependencies
            )

            # Store datasets by level name; parent may already exist from
            # a previous reducer — concat and deduplicate.
            parent_key = result.parent_level_name
            child_key = result.child_level_name

            for key, data in [(parent_key, result.parent_data), (child_key, result.child_data)]:
                if key in pipeline_result.datasets:
                    pipeline_result.datasets[key] = (
                        pd.concat([pipeline_result.datasets[key], data])
                        .drop_duplicates(subset=["date", "__model_key"])
                        .reset_index(drop=True)
                    )
                else:
                    pipeline_result.datasets[key] = data.copy()

        return pipeline_result
