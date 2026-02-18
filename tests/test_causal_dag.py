"""Tests for the causal_dag module."""

from __future__ import annotations

import pytest

from ergodicts.causal_dag import CausalDAG, EdgeSpec, ExternalNode, NodeConfig
from ergodicts.reducer import ModelKey


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def usa():
    return ModelKey(("COUNTRY",), ("USA",))


@pytest.fixture
def ca():
    return ModelKey(("STATE",), ("CA",))


@pytest.fixture
def ny():
    return ModelKey(("STATE",), ("NY",))


@pytest.fixture
def la():
    return ModelKey(("CITY",), ("LA",))


@pytest.fixture
def sf():
    return ModelKey(("CITY",), ("SF",))


@pytest.fixture
def gdp():
    return ExternalNode("GDP", dynamics="rw", integrated=True)


@pytest.fixture
def weather():
    return ExternalNode("Weather_LA", dynamics="stationary")


@pytest.fixture
def simple_dag(gdp, usa, ca):
    """GDP -> USA -> CA."""
    dag = CausalDAG()
    dag.add_edge(gdp, usa, lag=1)
    dag.add_edge(usa, ca, lag=0, contemporaneous=True)
    return dag


# ---------------------------------------------------------------------------
# TestExternalNode
# ---------------------------------------------------------------------------


class TestExternalNode:
    def test_basic(self, gdp):
        assert gdp.name == "GDP"
        assert gdp.dynamics == "rw"
        assert gdp.integrated is True

    def test_defaults(self):
        node = ExternalNode("CPI")
        assert node.dynamics == "ar1"
        assert node.integrated is False

    def test_str(self, gdp):
        assert str(gdp) == "GDP"

    def test_repr(self, gdp):
        r = repr(gdp)
        assert "GDP" in r
        assert "rw" in r
        assert "integrated=True" in r

    def test_frozen(self, gdp):
        with pytest.raises(AttributeError):
            gdp.name = "other"  # type: ignore[misc]

    def test_hashable(self, gdp):
        same = ExternalNode("GDP", dynamics="rw", integrated=True)
        assert gdp == same
        assert hash(gdp) == hash(same)
        assert len({gdp, same}) == 1

    def test_not_equal_different_name(self, gdp):
        other = ExternalNode("CPI", dynamics="rw", integrated=True)
        assert gdp != other

    def test_ordering(self, gdp, weather):
        nodes = sorted([weather, gdp])
        assert nodes[0].name == "GDP"
        assert nodes[1].name == "Weather_LA"

    def test_cross_type_ordering(self, gdp, usa):
        # ExternalNode and ModelKey should be sortable via __lt__
        result = sorted([usa, gdp], key=str)
        assert isinstance(result, list)

    def test_invalid_dynamics(self):
        with pytest.raises(ValueError, match="dynamics must be one of"):
            ExternalNode("X", dynamics="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestEdgeSpec
# ---------------------------------------------------------------------------


class TestEdgeSpec:
    def test_basic(self, gdp, usa):
        edge = EdgeSpec(source=gdp, target=usa, lag=1)
        assert edge.source is gdp
        assert edge.target is usa
        assert edge.lag == 1
        assert edge.contemporaneous is False

    def test_contemporaneous(self, usa, ca):
        edge = EdgeSpec(source=usa, target=ca, lag=0, contemporaneous=True)
        assert edge.contemporaneous is True
        assert "contemp" in repr(edge)

    def test_repr_lag(self, gdp, usa):
        edge = EdgeSpec(source=gdp, target=usa, lag=2)
        assert "lag=2" in repr(edge)

    def test_frozen(self, gdp, usa):
        edge = EdgeSpec(source=gdp, target=usa)
        with pytest.raises(AttributeError):
            edge.lag = 5  # type: ignore[misc]

    def test_negative_lag_raises(self, gdp, usa):
        with pytest.raises(ValueError, match="lag must be >= 0"):
            EdgeSpec(source=gdp, target=usa, lag=-1)

    def test_contemporaneous_with_nonzero_lag_raises(self, gdp, usa):
        with pytest.raises(ValueError, match="contemporaneous=True requires lag=0"):
            EdgeSpec(source=gdp, target=usa, lag=2, contemporaneous=True)

    def test_equality(self, gdp, usa):
        e1 = EdgeSpec(source=gdp, target=usa, lag=1)
        e2 = EdgeSpec(source=gdp, target=usa, lag=1)
        assert e1 == e2

    def test_different_lags_not_equal(self, gdp, usa):
        e1 = EdgeSpec(source=gdp, target=usa, lag=1)
        e2 = EdgeSpec(source=gdp, target=usa, lag=2)
        assert e1 != e2


# ---------------------------------------------------------------------------
# TestCausalDAG
# ---------------------------------------------------------------------------


class TestCausalDAG:
    def test_empty(self):
        dag = CausalDAG()
        assert len(dag) == 0
        assert dag.all_nodes == frozenset()

    def test_add_edge(self, gdp, usa):
        dag = CausalDAG()
        edge = dag.add_edge(gdp, usa, lag=1)
        assert isinstance(edge, EdgeSpec)
        assert len(dag) == 1
        assert gdp in dag.all_nodes
        assert usa in dag.all_nodes

    def test_parents_of(self, simple_dag, gdp, usa, ca):
        parents = simple_dag.parents_of(usa)
        assert len(parents) == 1
        assert parents[0].source is gdp

        parents_ca = simple_dag.parents_of(ca)
        assert len(parents_ca) == 1
        assert parents_ca[0].source is usa

    def test_children_of(self, simple_dag, gdp, usa):
        children = simple_dag.children_of(gdp)
        assert len(children) == 1
        assert children[0].target is usa

    def test_parents_of_unknown_node(self):
        dag = CausalDAG()
        unknown = ExternalNode("unknown")
        assert dag.parents_of(unknown) == []

    def test_children_of_unknown_node(self):
        dag = CausalDAG()
        unknown = ExternalNode("unknown")
        assert dag.children_of(unknown) == []

    def test_duplicate_edge_ignored(self, gdp, usa):
        dag = CausalDAG()
        dag.add_edge(gdp, usa, lag=1)
        dag.add_edge(gdp, usa, lag=1)
        assert len(dag) == 1

    def test_same_pair_different_lags_kept(self, gdp, usa):
        dag = CausalDAG()
        dag.add_edge(gdp, usa, lag=1)
        dag.add_edge(gdp, usa, lag=2)
        assert len(dag) == 2

    def test_external_nodes(self, simple_dag, gdp):
        ext = simple_dag.external_nodes
        assert ext == frozenset({gdp})

    def test_internal_nodes(self, simple_dag, usa, ca):
        internal = simple_dag.internal_nodes
        assert internal == frozenset({usa, ca})

    def test_all_nodes(self, simple_dag, gdp, usa, ca):
        assert simple_dag.all_nodes == frozenset({gdp, usa, ca})

    def test_repr(self, simple_dag):
        r = repr(simple_dag)
        assert "nodes=3" in r
        assert "edges=2" in r

    # -- topological order -------------------------------------------------

    def test_topological_order_linear(self, gdp, usa, ca):
        """GDP -> USA -> CA should yield [GDP, USA, CA]."""
        dag = CausalDAG()
        dag.add_edge(gdp, usa, lag=1)
        dag.add_edge(usa, ca, lag=0, contemporaneous=True)
        order = dag.topological_order()
        assert order.index(gdp) < order.index(usa)
        assert order.index(usa) < order.index(ca)

    def test_topological_order_diamond(self, gdp, usa, ca, ny):
        """GDP -> USA, GDP -> CA, USA -> NY, CA -> NY."""
        dag = CausalDAG()
        dag.add_edge(gdp, usa, lag=1)
        dag.add_edge(gdp, ca, lag=1)
        dag.add_edge(usa, ny, lag=0, contemporaneous=True)
        dag.add_edge(ca, ny, lag=0, contemporaneous=True)
        order = dag.topological_order()
        assert order[0] is gdp
        assert order.index(usa) < order.index(ny)
        assert order.index(ca) < order.index(ny)

    def test_topological_order_mixed_types(self, gdp, weather, usa, la):
        dag = CausalDAG()
        dag.add_edge(gdp, usa, lag=1)
        dag.add_edge(weather, la, lag=0, contemporaneous=True)
        dag.add_edge(usa, la, lag=1)
        order = dag.topological_order()
        assert order.index(gdp) < order.index(usa)
        assert order.index(usa) < order.index(la)
        assert order.index(weather) < order.index(la)

    def test_cycle_detection(self, usa, ca):
        dag = CausalDAG()
        dag.add_edge(usa, ca, lag=1)
        dag.add_edge(ca, usa, lag=1)
        with pytest.raises(ValueError, match="cycle"):
            dag.topological_order()

    def test_self_loop_detection(self, usa):
        dag = CausalDAG()
        dag.add_edge(usa, usa, lag=1)
        with pytest.raises(ValueError, match="cycle"):
            dag.topological_order()

    def test_validate_passes(self, simple_dag):
        simple_dag.validate()  # should not raise

    def test_validate_cycle_raises(self, usa, ca):
        dag = CausalDAG()
        dag.add_edge(usa, ca, lag=1)
        dag.add_edge(ca, usa, lag=1)
        with pytest.raises(ValueError, match="cycle"):
            dag.validate()

    # -- merge -------------------------------------------------------------

    def test_merge(self, gdp, usa, ca, weather, la):
        dag1 = CausalDAG()
        dag1.add_edge(gdp, usa, lag=1)

        dag2 = CausalDAG()
        dag2.add_edge(weather, la, lag=0, contemporaneous=True)

        merged = dag1.merge(dag2)
        assert len(merged) == 2
        assert gdp in merged.all_nodes
        assert weather in merged.all_nodes
        # Originals unchanged
        assert len(dag1) == 1
        assert len(dag2) == 1

    def test_merge_deduplicates(self, gdp, usa):
        dag1 = CausalDAG()
        dag1.add_edge(gdp, usa, lag=1)

        dag2 = CausalDAG()
        dag2.add_edge(gdp, usa, lag=1)

        merged = dag1.merge(dag2)
        assert len(merged) == 1

    # -- defensive copies --------------------------------------------------

    def test_parents_of_returns_copy(self, simple_dag, usa):
        parents = simple_dag.parents_of(usa)
        parents.clear()
        assert len(simple_dag.parents_of(usa)) == 1

    def test_children_of_returns_copy(self, simple_dag, gdp):
        children = simple_dag.children_of(gdp)
        children.clear()
        assert len(simple_dag.children_of(gdp)) == 1


# ---------------------------------------------------------------------------
# TestNodeConfig
# ---------------------------------------------------------------------------


class TestNodeConfig:
    def test_defaults(self):
        cfg = NodeConfig()
        assert cfg.mode == "auto"
        assert cfg.components is None
        assert cfg.aggregator is None
        assert cfg.prior_alpha == (1.0, 1.0)
        assert cfg.reconciliation_sigma == 1.0

    def test_custom(self):
        from ergodicts.components import FourierSeasonality, LocalLinearTrend, MultiplicativeSeasonality

        cfg = NodeConfig(
            mode="active",
            components=(LocalLinearTrend(), FourierSeasonality(n_harmonics=3)),
            aggregator=MultiplicativeSeasonality(),
            prior_alpha=(2.0, 5.0),
            reconciliation_sigma=0.01,
        )
        assert cfg.mode == "active"
        assert len(cfg.components) == 2

    def test_frozen(self):
        cfg = NodeConfig()
        with pytest.raises(Exception):  # Pydantic ValidationError
            cfg.mode = "passive"  # type: ignore[misc]

    def test_invalid_prior_alpha_raises(self):
        with pytest.raises(Exception, match="prior_alpha values must be > 0"):
            NodeConfig(prior_alpha=(0.0, 1.0))

    def test_invalid_prior_alpha_second_raises(self):
        with pytest.raises(Exception, match="prior_alpha values must be > 0"):
            NodeConfig(prior_alpha=(1.0, -0.5))

    def test_invalid_reconciliation_sigma_raises(self):
        with pytest.raises(Exception, match="reconciliation_sigma must be > 0"):
            NodeConfig(reconciliation_sigma=0.0)
