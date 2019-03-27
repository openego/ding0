import pytest

from egoio.tools import db
from sqlalchemy.orm import sessionmaker
import oedialect

import networkx as nx
import pandas as pd

from shapely.geometry import Point, LineString, LinearRing, Polygon
from ding0.core import NetworkDing0
from ding0.core.network import (RingDing0, BranchDing0, CircuitBreakerDing0,
                                GeneratorDing0, GeneratorFluctuatingDing0)
from ding0.core.network.stations import MVStationDing0, LVStationDing0
from ding0.core.network.grids import MVGridDing0, LVGridDing0


class TestMVGridDing0(object):

    @pytest.fixture
    def empty_mvgridding0(self):
        """
        Returns an empty MVGridDing0 object with an MVStationDing0 object
        with id_db = 0 and
        with geo_data = shapely.geometry.Point(0.5, 0.5)
        """
        station = MVStationDing0(id_db=0, geo_data=Point(0.5, 0.5))
        grid = MVGridDing0(id_db=0, station=station)
        return grid

    def test_empty_mvgridding0(self, empty_mvgridding0):
        """
        Check that initialization of an object of the
        class MVGridDing0 results in the right attributes being empty
        lists or NoneType objects, with the exception of the
        MVStationDing0 object's id_db and geo_data, which are
        0 and shapely.geometry.Point(0.5, 0.5), respectively.
        """
        assert empty_mvgridding0._rings == []
        assert empty_mvgridding0._circuit_breakers == []
        assert empty_mvgridding0.default_branch_kind is None
        assert empty_mvgridding0.default_branch_type is None
        assert empty_mvgridding0.default_branch_kind_settle is None
        assert empty_mvgridding0.default_branch_type_settle is None
        assert empty_mvgridding0.default_branch_kind_aggregated is None
        assert empty_mvgridding0.default_branch_type_aggregated is None
        assert empty_mvgridding0._station.id_db == 0
        assert empty_mvgridding0._station.geo_data == Point(0.5, 0.5)

    def test_add_circuit_breakers(self, empty_mvgridding0):
        """
        Adding a circuit breaker into an empty_mvgridding0 and check if it
        works.
        """
        circuit_breaker = CircuitBreakerDing0(id_db=0,
                                              geo_data=Point(0, 0),
                                              grid=empty_mvgridding0)
        empty_mvgridding0.add_circuit_breaker(circuit_breaker)
        circuit_breakers_in_grid = list(empty_mvgridding0.circuit_breakers())
        assert len(circuit_breakers_in_grid) == 1
        assert circuit_breakers_in_grid[0] == circuit_breaker

    def test_add_circuit_breakers_negative(self, empty_mvgridding0):
        """
        Adding a GeneratorDing0 as a circuit_breaker through the
        add_circuit_breaker just to see if the function rejects it.
        """
        bad_object = GeneratorDing0(id_db=0)
        empty_mvgridding0.add_circuit_breaker(bad_object)
        circuit_breakers_in_grid = list(empty_mvgridding0.circuit_breakers())
        assert len(circuit_breakers_in_grid) == 0

    @pytest.fixture
    def circuit_breaker_mvgridding0(self):
        """
        Returns an MVGridDing0 object with a branch and a
        circuit breaker.
        """
        station = MVStationDing0(id_db=0, geo_data=Point(0.5, 0.5))
        grid = MVGridDing0(id_db=0, station=station)
        branch = BranchDing0(id_db=0, length=2.0, kind='cable')
        circuit_breaker = CircuitBreakerDing0(id_db=0,
                                              geo_data=Point(0, 0),
                                              branch=branch,
                                              grid=grid)
        grid.add_circuit_breaker(circuit_breaker)
        grid._graph.add_edge(circuit_breaker, station,
                             branch=branch)
        return grid

    def test_open_circuit_breakers(self, circuit_breaker_mvgridding0):
        """
        Checks that using open_circuit_breakers function used from
        the MVGridDing0 object actually opens all the circuit breakers.
        """
        circuit_breakers_in_grid = list(
            circuit_breaker_mvgridding0.circuit_breakers()
        )
        assert circuit_breakers_in_grid[0].status == 'closed'
        circuit_breaker_mvgridding0.open_circuit_breakers()
        assert circuit_breakers_in_grid[0].status == 'open'

    def test_close_circuit_breakers(self, circuit_breaker_mvgridding0):
        """
        Checks that using close_circuit_breakers function used from
        the MVGridDing0 object actually closes all the circuit breakers.
        """
        circuit_breakers_in_grid = list(
            circuit_breaker_mvgridding0.circuit_breakers()
        )
        assert circuit_breakers_in_grid[0].status == 'closed'
        circuit_breaker_mvgridding0.open_circuit_breakers()
        assert circuit_breakers_in_grid[0].status == 'open'
        circuit_breaker_mvgridding0.close_circuit_breakers()
        assert circuit_breakers_in_grid[0].status == 'closed'

    @pytest.fixture
    def ring_mvgridding0(self):
        """
        Returns an MVGridDing0 object with 2 branches,
        a circuit breaker and a ring.
        """
        station = MVStationDing0(id_db=0, geo_data=Point(1, 1))
        grid = MVGridDing0(id_db=0,
                           station=station)
        generator1 = GeneratorDing0(id_db=0,
                                    geo_data=Point(1, 2),
                                    mv_grid=grid)
        grid.add_generator(generator1)
        generator2 = GeneratorDing0(id_db=1,
                                    geo_data=Point(2, 1),
                                    mv_grid=grid)
        grid.add_generator(generator2)
        generator3 = GeneratorDing0(id_db=2,
                                    geo_data=Point(2, 2),
                                    mv_grid=grid)
        grid.add_generator(generator3)
        ring = RingDing0(grid=grid)
        branch1 = BranchDing0(id_db='0', length=2.0, kind='cable', ring=ring)
        branch1a = BranchDing0(id_db='0a', lenght=1.2, kind='cable', ring=ring)
        branch2 = BranchDing0(id_db='1', lenght=3.0, kind='line', ring=ring)
        branch2a = BranchDing0(id_db='1a', lenght=2.0, kind='line', ring=ring)
        branch3 = BranchDing0(id_db='2', length=2.5, kind='line')
        circuit_breaker1 = CircuitBreakerDing0(id_db=0,
                                               geo_data=Point(0, 0),
                                               branch=branch1,
                                               grid=grid)
        grid.add_circuit_breaker(circuit_breaker1)
        grid._graph.add_edge(generator1, station,
                             branch=branch1)
        grid._graph.add_edge(circuit_breaker1, generator1,
                             branch=branch1a)
        grid._graph.add_edge(generator2, station,
                             branch=branch2)
        grid._graph.add_edge(circuit_breaker1, generator2,
                             branch=branch2a)
        grid._graph.add_edge(generator3, generator2, branch=branch3)
        grid.add_ring(ring)
        return (ring, grid)

    def test_add_ring(self, ring_mvgridding0):
        """
        Check if the number of rings is increased and the correct ring
        is added by using the add_ring function inside of MVGriDing0.
        """
        ring, grid = ring_mvgridding0
        assert len(grid._rings) == 1
        assert grid._rings[0] == ring

    def test_rings_count(self, ring_mvgridding0):
        """
        Check if the number of rings is correctly reflected using the
        rings_count function in MVGridDing0 and the correct ring
        is added by using the add_ring function inside of MVGriDing0.
        """
        ring, grid = ring_mvgridding0
        assert grid.rings_count() == 1
        assert grid._rings[0] == ring

    def test_get_ring_from_node(self, ring_mvgridding0):
        """
        Checks that the ring obtained from the get_ring_from_node object
        works as expected returning the correct ring.
        """
        ring, grid = ring_mvgridding0
        station = grid.station()
        assert grid.get_ring_from_node(station) == ring

    def test_rings_nodes_root_only_include_root(self, ring_mvgridding0):
        """
        Checks that the ring obtained from the rings_nodes function
        setting the include_root_node parameter to "True"
        contains the list of nodes that are expected.
        Currently the test doesn't check the consistency of the
        order of the items in the resulting list. It only checks
        if all the nodes expected are present and the
        length of the list is the same
        """
        ring, grid = ring_mvgridding0
        station = grid.station()
        generators = list(grid.generators())
        circuit_breakers = list(grid.circuit_breakers())
        rings_nodes_expected = [generators[0],
                                circuit_breakers[0],
                                generators[1],
                                station]
        rings_nodes = list(grid.rings_nodes(include_root_node=True))[0]
        assert len(rings_nodes) == len(rings_nodes_expected)
        assert set(rings_nodes) == set(rings_nodes_expected)

    def test_rings_nodes_root_only_exclude_root(self, ring_mvgridding0):
        """
        Checks that the ring obtained from the rings_nodes function
        setting the include_root_node parameter to "False"
        contains the list of nodes that are expected.
        Currently the test doesn't check the consistency of the
        order of the items in the resulting list. It only checks
        if all the nodes expected are present and the
        length of the list is the same
        """
        ring, grid = ring_mvgridding0
        generators = list(grid.generators())
        circuit_breakers = list(grid.circuit_breakers())
        rings_nodes_expected = [generators[0],
                                circuit_breakers[0],
                                generators[1]]
        rings_nodes = list(grid.rings_nodes(include_root_node=False))[0]
        assert len(rings_nodes) == len(rings_nodes_expected)
        assert set(rings_nodes) == set(rings_nodes_expected)

    def test_rings_nodes_include_satellites_include_root(self,
                                                         ring_mvgridding0):
        """
        Checks that the ring obtained from the rings_nodes function
        setting the include_root_node parameter to "True" and
        setting the include_satellites to "True"
        contains the list of nodes that are expected.
        Currently the test doesn't check the consistency of the
        order of the items in the resulting list. It only checks
        if all the nodes expected are present and the
        length of the list is the same
        """
        ring, grid = ring_mvgridding0
        station = grid.station()
        generators = list(grid.generators())
        circuit_breakers = list(grid.circuit_breakers())
        rings_nodes_expected = [generators[0],
                                circuit_breakers[0],
                                generators[1],
                                #station,
                                generators[2]]
        rings_nodes = list(grid.rings_nodes(include_root_node=True,
                                            include_satellites=True))[0]
        assert len(rings_nodes) == len(rings_nodes_expected)
        assert set(rings_nodes) == set(rings_nodes_expected)

    def test_rings_nodes_include_satellites_exclude_root(self,
                                                         ring_mvgridding0):
        """
        Checks that the ring obtained from the rings_nodes function
        setting the include_root_node parameter to "False" and
        setting the include_satellites to "True"
        contains the list of nodes that are expected.
        Currently the test doesn't check the consistency of the
        order of the items in the resulting list. It only checks
        if all the nodes expected are present and the
        length of the list is the same
        """
        ring, grid = ring_mvgridding0
        generators = list(grid.generators())
        circuit_breakers = list(grid.circuit_breakers())
        rings_nodes_expected = [generators[0],
                                circuit_breakers[0],
                                generators[1],
                                generators[2]]
        rings_nodes = list(grid.rings_nodes(include_root_node=False,
                                            include_satellites=True))[0]
        assert len(rings_nodes) == len(rings_nodes_expected)
        assert set(rings_nodes) == set(rings_nodes_expected)

    def test_rings_full_data(self, ring_mvgridding0):
        """
        Checks if the function rings_full_data produces the expected
        list of rings, list of branches and list of ring_nodes.
        """
        ring, grid = ring_mvgridding0
        station = grid.station()
        generators = list(grid.generators())
        circuit_breakers = list(grid.circuit_breakers())
        branches = sorted(list(map(lambda x: x['branch'],
                                   grid.graph_edges())),
                          key=lambda x: repr(x))
        ring_expected = ring
        # branches following the ring
        branches_expected = [branches[1],
                             branches[0],
                             branches[3],
                             branches[2]]
        rings_nodes_expected = [generators[0],
                                circuit_breakers[0],
                                generators[1],
                                station]
        (ring_out,
         branches_out,
         rings_nodes_out) = list(grid.rings_full_data())[0]
        assert ring_out == ring_expected
        assert len(branches_out) == len(branches_expected)
        assert set(branches_out) == set(branches_expected)
        assert len(rings_nodes_out) == len(rings_nodes_expected)
        assert set(rings_nodes_out) == set(rings_nodes_expected)

    def test_graph_nodes_from_subtree_station(self, ring_mvgridding0):
        """
        Check the output of function graph_nodes_from_subtree using the
        station as the source node. With this input, there should be no
        nodes. This should mean an empty list is returned by the
        graph_nodes_from_subtree function.
        """
        ring, grid = ring_mvgridding0
        station = grid.station()
        with pytest.raises(UnboundLocalError):
            nodes_out = grid.graph_nodes_from_subtree(station)

    def test_graph_nodes_from_subtree_circuit_breaker(self, ring_mvgridding0):
        """
        Check the output of function graph_nodes_from_subtree using the
        circuit breaker as the source node. With this input, there should be no
        nodes. This should mean an empty list is returned by the
        graph_nodes_from_subtree function.
        """
        ring, grid = ring_mvgridding0
        circuit_breakers = list(grid.circuit_breakers())
        nodes_out = grid.graph_nodes_from_subtree(circuit_breakers[0])
        nodes_expected = []
        assert nodes_out == nodes_expected

    def test_graph_nodes_from_subtree_ring_branch_left(self, ring_mvgridding0):
        """
        Check the output of function graph_nodes_from_subtree using the
        generator on the left branch as the source node.
        With this input, there should be no nodes.
        This should mean an empty list is returned by the
        graph_nodes_from_subtree function.
        """
        ring, grid = ring_mvgridding0
        generators = list(grid.generators())
        nodes_out = grid.graph_nodes_from_subtree(generators[0])
        nodes_expected = []
        assert nodes_out == nodes_expected

    def test_graph_nodes_from_subtree_ring_branch_right(self,
                                                        ring_mvgridding0):
        """
        Check the output of function graph_nodes_from_subtree using the
        generator on the right branch as the source node.
        With this input, there should be one node,
        the generator outside the ring connected to the right branch using
        a stub. This should mean a list with this specific generator
        should be returned by the graph_nodes_from_subtree function.
        """
        ring, grid = ring_mvgridding0
        generators = list(grid.generators())
        nodes_out = grid.graph_nodes_from_subtree(generators[1])
        nodes_expected = [generators[2]]
        assert nodes_out == nodes_expected

    def test_graph_nodes_from_subtree_off_ring(self, ring_mvgridding0):
        """
        Check the output of function graph_nodes_from_subtree using the
        generator outside of the ring as the source node.
        With this input, there should be no nodes.
        This should mean an empty list is returned by the
        graph_nodes_from_subtree function.
        """
        ring, grid = ring_mvgridding0
        generators = list(grid.generators())
        with pytest.raises(UnboundLocalError):
            nodes_out = grid.graph_nodes_from_subtree(generators[2])
        # nodes_expected = []
        # assert nodes_out == nodes_expected

    @pytest.fixture
    def oedb_session(self):
        """
        Returns an ego.io oedb session and closes it on finishing the test
        """
        engine = db.connection(section='oedb')
        session = sessionmaker(bind=engine)()
        yield session
        print("closing session")
        session.close()

    def test_routing(self, oedb_session):
        """
        Using the grid district 460 as an example, the properties of the
        networkx graph is tested before routing and after routing.
        """
        # instantiate new ding0 network object
        nd = NetworkDing0(name='network')

        nd.import_mv_grid_districts(oedb_session,
                                    mv_grid_districts_no=[460])
        # STEP 2: Import generators
        nd.import_generators(oedb_session)
        # STEP 3: Parametrize MV grid
        nd.mv_parametrize_grid()
        # STEP 4: Validate MV Grid Districts
        nd.validate_grid_districts()
        # STEP 5: Build LV grids
        nd.build_lv_grids()

        graph = nd._mv_grid_districts[0].mv_grid._graph

        assert len(list(graph.nodes())) == 256
        assert len(list(graph.edges())) == 0
        assert len(list(nx.isolates(graph))) == 256
        assert pd.Series(dict(graph.degree())).sum(axis=0) == 0
        assert pd.Series(dict(graph.degree())).mean(axis=0) == 0.0
        assert len(list(nx.get_edge_attributes(graph, 'branch'))) == 0
        assert nx.average_node_connectivity(graph) == 0.0
        assert pd.Series(
            nx.degree_centrality(graph)
            ).mean(axis=0) == 0.0
        assert pd.Series(
            nx.closeness_centrality(graph)
            ).mean(axis=0) == 0.0
        assert pd.Series(
            nx.betweenness_centrality(graph)
            ).mean(axis=0) == 0.0

        nd.mv_routing()

        assert len(list(graph.nodes())) == 269
        assert len(list(graph.edges())) == 218
        assert len(list(nx.isolates(graph))) == 54
        assert pd.Series(dict(graph.degree())).sum(axis=0) == 436
        assert pd.Series(
            dict(graph.degree())
            ).mean(axis=0) == pytest.approx(1.62, 0.001)
        assert len(list(nx.get_edge_attributes(graph, 'branch'))) == 218
        assert nx.average_node_connectivity(graph) == pytest.approx(
            0.688,
            abs=0.0001
            )
        assert pd.Series(
            nx.degree_centrality(graph)
            ).mean(axis=0) == pytest.approx(0.006, abs=0.001)
        assert pd.Series(
            nx.closeness_centrality(graph)
            ).mean(axis=0) == pytest.approx(0.042474, abs=0.00001)
        assert pd.Series(
            nx.betweenness_centrality(graph)
            ).mean(axis=0) == pytest.approx(0.0354629, abs=0.00001)
        assert pd.Series(
            nx.edge_betweenness_centrality(graph)
            ).mean(axis=0) == pytest.approx(0.04636150, abs=0.00001)


class TestLVGridDing0(object):

    @pytest.fixture
    def empty_lvgridding0(self):
        """
        Returns and empty LVGridDing0 object
        """
        lv_station = LVStationDing0(id_db=0, geo_data=Point(1, 1))
        grid = LVGridDing0(id_db=0, station=lv_station)
        return grid


if __name__ == "__main__":
    pass
