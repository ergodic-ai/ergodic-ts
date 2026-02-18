"""CausalDAG — directed acyclic graph of predictive/causal relationships.

This module is **independent** of the hierarchy graph
(:class:`~ergodicts.reducer.DependencyGraph`).  The hierarchy describes
aggregation (what sums to what); the causal DAG describes prediction
(what predicts what).  A forecasting model consumes both.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, field_serializer, field_validator

from ergodicts.reducer import ModelKey

if TYPE_CHECKING:
    import graphviz

# ---------------------------------------------------------------------------
# Node type alias
# ---------------------------------------------------------------------------

type Node = ModelKey | ExternalNode


# ---------------------------------------------------------------------------
# ExternalNode
# ---------------------------------------------------------------------------

_DYNAMICS_CHOICES = frozenset({"rw", "ar1", "ar2", "stationary", "known_future"})


@dataclass(frozen=True)
class ExternalNode:
    """An external predictor time-series (e.g. GDP, weather).

    Parameters
    ----------
    name : str
        Unique identifier for this external series.
    dynamics : str
        How this series should be evolved during the forecast horizon.
        One of ``"rw"`` (random walk), ``"ar1"``, ``"ar2"``,
        ``"stationary"``, or ``"known_future"`` (user provides future
        values).
    integrated : bool
        If ``True`` the series is I(1) and should be modelled on
        differences.

    Examples
    --------
    ```python
    gdp = ExternalNode("GDP", dynamics="rw", integrated=True)
    weather = ExternalNode("Weather_LA", dynamics="stationary")
    ```
    """

    name: str
    dynamics: Literal["rw", "ar1", "ar2", "stationary", "known_future"] = "ar1"
    integrated: bool = False

    def __post_init__(self) -> None:
        if self.dynamics not in _DYNAMICS_CHOICES:
            raise ValueError(
                f"dynamics must be one of {sorted(_DYNAMICS_CHOICES)}, "
                f"got {self.dynamics!r}"
            )

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return (
            f"ExternalNode({self.name!r}, dynamics={self.dynamics!r}, "
            f"integrated={self.integrated})"
        )

    def __lt__(self, other: object) -> bool:
        if isinstance(other, ExternalNode):
            return self.name < other.name
        if isinstance(other, ModelKey):
            return str(self) < str(other)
        return NotImplemented


# ---------------------------------------------------------------------------
# EdgeSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeSpec:
    """A directed causal/predictive edge with temporal metadata.

    Parameters
    ----------
    source : Node
        The predictor node.
    target : Node
        The predicted node.
    lag : int
        How many periods back the source value is used.
        ``lag=0`` with ``contemporaneous=False`` means the source at
        *t-0* is used but in the lagged sense (available at prediction
        time).  ``lag=0`` with ``contemporaneous=True`` means a true
        same-timestep effect.
    contemporaneous : bool
        If ``True`` the source and target interact within the same time
        step.  Requires ``lag=0``.
    """

    source: Node
    target: Node
    lag: int = 0
    contemporaneous: bool = False

    def __post_init__(self) -> None:
        if self.lag < 0:
            raise ValueError(f"lag must be >= 0, got {self.lag}")
        if self.contemporaneous and self.lag != 0:
            raise ValueError("contemporaneous=True requires lag=0")

    def __repr__(self) -> str:
        lag_str = "contemp" if self.contemporaneous else f"lag={self.lag}"
        return f"EdgeSpec({self.source} -> {self.target}, {lag_str})"


# ---------------------------------------------------------------------------
# CausalDAG
# ---------------------------------------------------------------------------


class CausalDAG:
    """Directed acyclic graph of causal/predictive relationships.

    Nodes are either :class:`ModelKey` (internal sales series) or
    :class:`ExternalNode` (external predictors).  Edges carry temporal
    metadata (lag, contemporaneous flag).

    This graph is **independent** of the hierarchy
    (:class:`~ergodicts.reducer.DependencyGraph`).

    Examples
    --------
    ```python
    from ergodicts import CausalDAG, ExternalNode, ModelKey

    gdp = ExternalNode("GDP", dynamics="rw", integrated=True)
    usa = ModelKey(("COUNTRY",), ("USA",))
    ca  = ModelKey(("STATE",), ("CA",))

    dag = CausalDAG()
    dag.add_edge(gdp, usa, lag=1)
    dag.add_edge(gdp, ca,  lag=1)
    dag.add_edge(usa, ca,  lag=0, contemporaneous=True)

    dag.topological_order()  # [ExternalNode('GDP', ...), ModelKey(COUNTRY=USA), ModelKey(STATE=CA)]
    ```
    """

    def __init__(self) -> None:
        self._outgoing: dict[Node, list[EdgeSpec]] = {}
        self._incoming: dict[Node, list[EdgeSpec]] = {}
        self._nodes: set[Node] = set()

    # -- mutation -----------------------------------------------------------

    def add_edge(
        self,
        source: Node,
        target: Node,
        *,
        lag: int = 0,
        contemporaneous: bool = False,
    ) -> EdgeSpec:
        """Add a causal edge from *source* to *target*.

        Returns the created :class:`EdgeSpec`.  Duplicate edges (same
        source, target, lag, and contemporaneous flag) are silently
        ignored.
        """
        edge = EdgeSpec(
            source=source,
            target=target,
            lag=lag,
            contemporaneous=contemporaneous,
        )
        # Deduplicate
        if edge in self._outgoing.get(source, []):
            return edge

        self._outgoing.setdefault(source, []).append(edge)
        self._incoming.setdefault(target, []).append(edge)
        self._nodes.add(source)
        self._nodes.add(target)
        return edge

    def merge(self, other: CausalDAG) -> CausalDAG:
        """Return a **new** DAG containing edges from both graphs."""
        merged = CausalDAG()
        for dag in (self, other):
            for edges in dag._outgoing.values():
                for e in edges:
                    merged.add_edge(
                        e.source, e.target,
                        lag=e.lag,
                        contemporaneous=e.contemporaneous,
                    )
        return merged

    # -- queries ------------------------------------------------------------

    def parents_of(self, node: Node) -> list[EdgeSpec]:
        """Return all incoming edges (predictors of *node*)."""
        return list(self._incoming.get(node, []))

    def children_of(self, node: Node) -> list[EdgeSpec]:
        """Return all outgoing edges (nodes predicted by *node*)."""
        return list(self._outgoing.get(node, []))

    @property
    def all_nodes(self) -> frozenset[Node]:
        """All nodes in the graph."""
        return frozenset(self._nodes)

    @property
    def external_nodes(self) -> frozenset[ExternalNode]:
        """Only :class:`ExternalNode` instances."""
        return frozenset(n for n in self._nodes if isinstance(n, ExternalNode))

    @property
    def internal_nodes(self) -> frozenset[ModelKey]:
        """Only :class:`ModelKey` instances."""
        return frozenset(n for n in self._nodes if isinstance(n, ModelKey))

    # -- topological sort ---------------------------------------------------

    def topological_order(self) -> list[Node]:
        """Return nodes in topological order (parents before children).

        Raises :class:`ValueError` if the graph contains a cycle.
        """
        # Kahn's algorithm
        in_degree: dict[Node, int] = {n: 0 for n in self._nodes}
        for edges in self._outgoing.values():
            for e in edges:
                in_degree[e.target] += 1

        queue: deque[Node] = deque(
            sorted(
                (n for n, d in in_degree.items() if d == 0),
                key=str,
            )
        )
        result: list[Node] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for e in self._outgoing.get(node, []):
                in_degree[e.target] -= 1
                if in_degree[e.target] == 0:
                    queue.append(e.target)
            # Re-sort queue for deterministic output when new nodes are added
            if queue:
                queue = deque(sorted(queue, key=str))

        if len(result) != len(self._nodes):
            raise ValueError(
                "CausalDAG contains a cycle — topological ordering is impossible"
            )
        return result

    def validate(self) -> None:
        """Check that the graph is a valid DAG.

        Raises :class:`ValueError` if cycles exist.
        """
        self.topological_order()

    # -- visualization ------------------------------------------------------

    def show(self) -> graphviz.Digraph:
        """Render the causal DAG as a Graphviz digraph.

        External nodes are shown as ellipses (orange); internal
        :class:`ModelKey` nodes are shown as rounded boxes (green).
        Edge labels indicate lag information.

        Requires the ``graphviz`` Python package.
        """
        try:
            import graphviz as gv
        except ImportError:
            raise ImportError(
                "graphviz is required for .show(). "
                "Install it with: pip install graphviz"
            ) from None

        dot = gv.Digraph(
            "CausalDAG",
            graph_attr={"rankdir": "LR", "fontsize": "12"},
            node_attr={"fontsize": "10"},
        )

        # Add nodes with type-specific styling
        for node in sorted(self._nodes, key=str):
            nid = self._node_id(node)
            if isinstance(node, ExternalNode):
                dot.node(
                    nid,
                    label=str(node),
                    shape="ellipse",
                    style="filled",
                    fillcolor="#f5e6cc",
                )
            else:
                dot.node(
                    nid,
                    label=node.label,
                    shape="box",
                    style="rounded,filled",
                    fillcolor="#d4e8d4",
                )

        # Add edges with lag labels
        for edges in self._outgoing.values():
            for e in edges:
                label = "contemp" if e.contemporaneous else f"lag={e.lag}"
                dot.edge(
                    self._node_id(e.source),
                    self._node_id(e.target),
                    label=label,
                )

        return dot

    @staticmethod
    def _node_id(node: Node) -> str:
        """Unique graphviz node ID that avoids cross-type collisions."""
        return f"{type(node).__name__}:{node}"

    # -- dunder -------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of edges."""
        return sum(len(edges) for edges in self._outgoing.values())

    def __repr__(self) -> str:
        return f"CausalDAG(nodes={len(self._nodes)}, edges={len(self)})"


# ---------------------------------------------------------------------------
# NodeConfig
# ---------------------------------------------------------------------------


class NodeConfig(BaseModel):
    """Per-node configuration for the hierarchical forecaster.

    Controls how a node behaves in the factor-graph model:
    whether it runs its own dynamics (``"active"``), purely inherits
    from its hierarchy parent (``"passive"``), or lets the model
    decide (``"auto"``).

    Parameters
    ----------
    mode : str
        ``"auto"`` estimates mixing weight from data, ``"active"``
        forces full own dynamics, ``"passive"`` forces proportional
        allocation from hierarchy parent.
    prior_alpha : tuple of float
        Beta distribution parameters ``(a, b)`` for the mixing weight
        when ``mode="auto"``.
    reconciliation_sigma : float
        Standard deviation of the reconciliation potential (smaller =
        harder constraint).
    components : tuple or None
        Explicit tuple of component instances.  When ``None`` (default),
        a single ``LocalLinearTrend`` is used.
    aggregator : Aggregator or None
        How component contributions are combined into the predicted
        mean.  ``None`` (default) uses additive aggregation.
        Pass ``MultiplicativeAggregator()`` or
        ``LogAdditiveAggregator()`` for alternative decompositions.

    Examples
    --------
    ```python
    # Simple trend-only (default)
    NodeConfig()

    # Trend + Fourier seasonality (additive)
    NodeConfig(components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=2)))

    # (trend + exogenous) * seasonality
    NodeConfig(
        components=(LocalLinearTrend(), MultiplicativeFourierSeasonality(n_harmonics=2)),
        aggregator=MultiplicativeSeasonality(),
    )
    ```
    """

    mode: Literal["auto", "active", "passive"] = "auto"
    prior_alpha: tuple[float, float] = (1.0, 1.0)
    reconciliation_sigma: float = 1.0
    components: tuple | None = None
    aggregator: object | None = None

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @field_validator("prior_alpha")
    @classmethod
    def _prior_alpha_positive(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("prior_alpha values must be > 0")
        return v

    @field_validator("reconciliation_sigma")
    @classmethod
    def _reconciliation_sigma_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("reconciliation_sigma must be > 0")
        return v

    @field_serializer("components")
    def _serialize_components(self, value: tuple | None, _info: Any) -> list[dict] | None:
        if value is None:
            return None
        return [comp.to_dict() for comp in value]

    @field_serializer("aggregator")
    def _serialize_aggregator(self, value: object | None, _info: Any) -> dict | None:
        if value is None:
            return None
        return value.to_dict()  # type: ignore[union-attr]

    @field_validator("components", mode="before")
    @classmethod
    def _deserialize_components(cls, v: Any) -> tuple | None:
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], dict):
            from ergodicts.components import ComponentLibrary
            return tuple(ComponentLibrary.from_dict(d) for d in v)
        return v

    @field_validator("aggregator", mode="before")
    @classmethod
    def _deserialize_aggregator(cls, v: Any) -> object | None:
        if v is None or not isinstance(v, dict):
            return v
        from ergodicts.components import Aggregator
        return Aggregator.from_dict(v)
