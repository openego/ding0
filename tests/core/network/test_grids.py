import pytest



from math import isnan

import networkx as nx
import pandas as pd

from shapely.geometry import Point, LineString, LinearRing, Polygon

from ding0.core import NetworkDing0
from ding0.core.network import (GridDing0, StationDing0,
                                TransformerDing0,
                                RingDing0, BranchDing0,
                                CableDistributorDing0, CircuitBreakerDing0,
                                GeneratorDing0, GeneratorFluctuatingDing0,
                                LoadDing0)
from ding0.core.network.grids import MVGridDing0, LVGridDing0
from ding0.core.network.stations import MVStationDing0, LVStationDing0
from ding0.core.structure.regions import (MVGridDistrictDing0,
                                          LVLoadAreaDing0,
                                          LVLoadAreaCentreDing0,
                                          LVGridDistrictDing0)
from ding0.tools.tools import (get_dest_point,
                               get_cart_dest_point,
                               create_poly_from_source)
from ding0.core import NetworkDing0

from egoio.tools import db
from sqlalchemy.orm import sessionmaker
import oedialect

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
        contains all nodes that are expected and that the length of the list
        is as expected.
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
        contains all nodes that are expected and that the length of the list
        is as expected.
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
        contains all nodes that are expected and that the length of the list
        is as expected.
        """
        ring, grid = ring_mvgridding0
        station = grid.station()
        generators = list(grid.generators())
        circuit_breakers = list(grid.circuit_breakers())
        rings_nodes_expected = [generators[0],
                                circuit_breakers[0],
                                generators[1],
                                station,
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
        contains all nodes that are expected and that the length of the list
        is as expected.
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
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
            nodes_out = grid.graph_nodes_from_subtree(generators[2])

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

    @pytest.fixture
    def minimal_unrouted_grid(self):
        """
        Returns an MVGridDing0 object with a few artificially
        generated information about a fictious set of load
        areas and mv grid district.
        All the data created here is typically a replacement
        for the data that ding0 draws from the openenergy-platform.
        The flow of this function follows the :meth:`~.core.NetworkDing0.run_ding0` function
        in :class:`~.core.NetworkDing0`.

        Returns
        -------
        """
        # define source Coordinates EPSG 4326
        source = Point(8.638204, 49.867307)

        # distance_scaling factor
        distance_scaling_factor = 3

        # Create NetworkDing0 object
        network = NetworkDing0(name='network')

        # Create MVStationDing0 and the HV-MV Transformers
        mv_station = MVStationDing0(
            id_db=0,
            geo_data=source,
            peak_load=10000.0,
            v_level_operation=20.0
        )

        hvmv_transformers = [
            TransformerDing0(
                id_db=0,
                s_max_longterm=63000.0,
                v_level=20.0
            ),
            TransformerDing0(
                id_db=1,
                s_max_longterm=63000.0,
                v_level=20.0
            )
        ]

        # Add the transformers to the station

        for hvmv_transfromer in hvmv_transformers:
            mv_station.add_transformer(hvmv_transfromer)

        # Create the MV Grid
        mv_grid = MVGridDing0(
            id_db=0,
            network=network,
            v_level=20.0,
            station=mv_station,
            default_branch_kind='line',
            default_branch_kind_aggregated='line',
            default_branch_kind_settle='cable',
            default_branch_type=pd.Series(
                dict(name='48-AL1/8-ST1A',
                     U_n=20.0,
                     I_max_th=210.0,
                     R=0.37,
                     L=1.18,
                     C=0.0098,
                     reinforce_only=0)
            ),
            default_branch_type_aggregated=pd.Series(
                dict(name='122-AL1/20-ST1A',
                     U_n=20.0,
                     I_max_th=410.0,
                     R=0.34,
                     L=1.08,
                     C=0.0106,
                     reinforce_only=0)
            ),
            default_branch_type_settle=pd.Series(
                dict(name='NA2XS2Y 3x1x150 RE/25',
                     U_n=20.0,
                     I_max_th=319.0,
                     R=0.206,
                     L=0.4011,
                     C=0.24,
                     reinforce_only=0)
            )
        )
        # Add some MV Generators that are directly connected at the station

        mv_generators = [
            GeneratorDing0(
                id_db=0,
                capacity=200.0,
                mv_grid=mv_grid,
                type='biomass',
                subtype='biogas_from_grid',
                v_level=5,
                geo_data=source
            ),
            GeneratorDing0(
                id_db=1,
                capacity=200.0,
                mv_grid=mv_grid,
                type='biomass',
                subtype='biogas_from_grid',
                v_level=5,
                geo_data=source
            ),
            GeneratorFluctuatingDing0(
                id_db=2,
                weather_cell_id=0,
                capacity=1000.0,
                mv_grid=mv_grid,
                type='wind',
                subtype='wind_onshore',
                v_level=5,
                geo_data=source
            ),
            GeneratorFluctuatingDing0(
                id_db=3,
                weather_cell_id=1,
                capacity=1000.0,
                mv_grid=mv_grid,
                type='wind',
                subtype='wind_onshore',
                v_level=5,
                geo_data=source
            )
        ]
        for mv_gen in mv_generators:
            mv_grid.add_generator(mv_gen)

        # Create the MV Grid District

        mv_grid_district_geo_data = create_poly_from_source(
            source,
            distance_scaling_factor * 5000,
            distance_scaling_factor * 10000,
            distance_scaling_factor * 5000,
            distance_scaling_factor * 8000
        )

        mv_grid_district = MVGridDistrictDing0(id_db=10000,
                                               mv_grid=mv_grid,
                                               geo_data=mv_grid_district_geo_data)
        mv_grid.grid_district = mv_grid_district
        mv_station.grid = mv_grid

        # Put the MV Grid District into Network

        network.add_mv_grid_district(mv_grid_district)

        # Create the LV Grid Districts
        lv_grid_districts_data = pd.DataFrame(
            dict(
                la_id=list(range(19)),
                population=[
                    223, 333, 399, 342,
                    429, 493, 431, 459,
                    221, 120, 111, 140, 70, 156, 83, 4, 10,
                    679,
                    72
                ],
                peak_load_residential=[
                    141.81529461443094,
                    226.7797238255373,
                    162.89215815390679,
                    114.27072719604516,
                    102.15005942005169,
                    198.71310283772826,
                    147.79521215488836,
                    159.08248928315558,
                    35.977009995358891,
                    84.770575023989707,
                    29.61665986331883,
                    48.544967225998533,
                    41.253931841299796,
                    50.2394059644368,
                    14.69162136059512,
                    88.790542939734,
                    98.03,
                    500.73283157785517,
                    137.54926972703203
                ],
                peak_load_retail=[
                    34.0,
                    0.0,
                    0.0,
                    82.0,
                    63.0,
                    20.0,
                    0.0,
                    0.0,
                    28.0,
                    0.0,
                    0.0,
                    0.0,
                    37.0,
                    0.0,
                    19.0,
                    0.0,
                    0.0,
                    120.0,
                    0.0
                ],
                peak_load_industrial=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    300.0,
                    0.0,
                    0.0,
                    158.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    100.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                peak_load_agricultural=[
                    0.0,
                    0.0,
                    120.0,
                    0.0,
                    0.0,
                    30.0,
                    0.0,
                    140.0,
                    0.0,
                    0.0,
                    60.0,
                    40.0,
                    0.0,
                    0.0,
                    0.0,
                    20.0,
                    80.0,
                    0.0,
                    0.0
                ],
                geom=[
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -1000 * distance_scaling_factor,
                            1000 * distance_scaling_factor
                        ),
                        100,
                        100,
                        100,
                        100
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -1000 * distance_scaling_factor,
                            4000 * distance_scaling_factor
                        ),
                        100,
                        100,
                        100,
                        100
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -4000 * distance_scaling_factor,
                            4000 * distance_scaling_factor
                        ),
                        100,
                        100,
                        100,
                        100
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -4000 * distance_scaling_factor,
                            1000 * distance_scaling_factor
                        ),
                        100,
                        100,
                        100,
                        100
                    ),  # ----
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -1000 * distance_scaling_factor,
                            -1000 * distance_scaling_factor
                        ),
                        100,
                        100,
                        100,
                        100
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -1000 * distance_scaling_factor,
                            -4000 * distance_scaling_factor
                        ),
                        100,
                        100,
                        100,
                        100
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -4000 * distance_scaling_factor,
                            -4000 * distance_scaling_factor
                        ),
                        100,
                        100,
                        100,
                        100
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -4000 * distance_scaling_factor,
                            -1000 * distance_scaling_factor
                        ),
                        100,
                        100,
                        100,
                        100
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -2500 * distance_scaling_factor,
                            -5500 * distance_scaling_factor
                        ),
                        50,
                        50,
                        50,
                        50
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -2500 * distance_scaling_factor,
                            -6500 * distance_scaling_factor
                        ),
                        50,
                        50,
                        50,
                        50
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -3500 * distance_scaling_factor,
                            -7500 * distance_scaling_factor
                        ),
                        50,
                        50,
                        50,
                        50
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -1500 * distance_scaling_factor,
                            -7500 * distance_scaling_factor
                        ),
                        50,
                        50,
                        50,
                        50
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -500 * distance_scaling_factor,
                            -6500 * distance_scaling_factor
                        ),
                        50,
                        50,
                        50,
                        50
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -500 * distance_scaling_factor,
                            -5500 * distance_scaling_factor
                        ),
                        50,
                        50,
                        50,
                        50
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -500 * distance_scaling_factor,
                            -7500 * distance_scaling_factor
                        ),
                        50,
                        50,
                        50,
                        50
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            -1500 * distance_scaling_factor,
                            -6500 * distance_scaling_factor
                        ),
                        50,
                        50,
                        50,
                        50
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            2500 * distance_scaling_factor,
                            2500 * distance_scaling_factor
                        ),
                        150,
                        150,
                        150,
                        150
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            9000 * distance_scaling_factor,
                            4000 * distance_scaling_factor
                        ),
                        100,
                        100,
                        100,
                        100
                    ),
                    create_poly_from_source(
                        get_cart_dest_point(
                            source,
                            9000 * distance_scaling_factor,
                            -7000 * distance_scaling_factor
                        ),
                        100,
                        100,
                        100,
                        100
                    ),
                ]
            )
        )

        lv_grid_districts_data.loc[:, 'peak_load'] = (
            lv_grid_districts_data.loc[:, ['peak_load_residential',
                                           'peak_load_retail',
                                           'peak_load_industrial',
                                           'peak_load_agricultural']].sum(axis=1)
        )

        # Create the LVGridDing0 objects and all its constituents
        # from the LVGridDistrictDing0 data

        lv_nominal_voltage = 400.0
        # assuming that the lv_grid districts and lv_load_areas are the same!
        # or 1 lv_griddistrict is 1 lv_loadarea
        lv_stations = []
        for id_db, row in lv_grid_districts_data.iterrows():
            lv_load_area = LVLoadAreaDing0(id_db=id_db,
                                           db_data=row,
                                           mv_grid_district=mv_grid_district,
                                           peak_load=row['peak_load'])
            lv_load_area.geo_area = row['geom']
            lv_load_area.geo_centre = row['geom'].centroid
            lv_grid_district = LVGridDistrictDing0(
                id_db=id,
                lv_load_area=lv_load_area,
                geo_data=row['geom'],
                population=(0
                            if isnan(row['population'])
                            else int(row['population'])),
                peak_load_residential=row['peak_load_residential'],
                peak_load_retail=row['peak_load_retail'],
                peak_load_industrial=row['peak_load_industrial'],
                peak_load_agricultural=row['peak_load_agricultural'],
                peak_load=(row['peak_load_residential']
                           + row['peak_load_retail']
                           + row['peak_load_industrial']
                           + row['peak_load_agricultural']),
                sector_count_residential=(0
                                          if row['peak_load_residential'] == 0.0
                                          else 1),
                sector_count_retail=(0
                                     if row['peak_load_retail'] == 0.0
                                     else 1),
                sector_count_agricultural=(0
                                           if row['peak_load_agricultural'] == 0.0
                                           else 1),
                sector_count_industrial=(0
                                         if row['peak_load_industrial'] == 0.0
                                         else 1),
                sector_consumption_residential=row['peak_load_residential'],
                sector_consumption_retail=row['peak_load_retail'],
                sector_consumption_industrial=row['peak_load_industrial'],
                sector_consumption_agricultural=row['peak_load_agricultural']
            )

            # be aware, lv_grid takes grid district's geom!
            lv_grid = LVGridDing0(network=network,
                                  grid_district=lv_grid_district,
                                  id_db=id_db,
                                  geo_data=row['geom'],
                                  v_level=lv_nominal_voltage)

            # create LV station
            lv_station = LVStationDing0(
                id_db=id_db,
                grid=lv_grid,
                lv_load_area=lv_load_area,
                geo_data=row['geom'].centroid,
                peak_load=lv_grid_district.peak_load
            )

            # assign created objects
            # note: creation of LV grid is done separately,
            # see NetworkDing0.build_lv_grids()
            lv_grid.add_station(lv_station)
            lv_grid_district.lv_grid = lv_grid
            lv_load_area.add_lv_grid_district(lv_grid_district)

            lv_load_area_centre = LVLoadAreaCentreDing0(id_db=id_db,
                                                        geo_data=row['geom'].centroid,
                                                        lv_load_area=lv_load_area,
                                                        grid=mv_grid_district.mv_grid)
            lv_load_area.lv_load_area_centre = lv_load_area_centre
            mv_grid_district.add_lv_load_area(lv_load_area)
            lv_stations.append(lv_station)

        lv_stations = sorted(lv_stations, key=lambda x: repr(x))
        mv_grid_district.add_peak_demand()
        mv_grid.set_voltage_level()

        # set MV station's voltage level
        mv_grid._station.set_operation_voltage_level()

        # set default branch types (normal, aggregated areas and within settlements)
        (mv_grid.default_branch_type,
         mv_grid.default_branch_type_aggregated,
         mv_grid.default_branch_type_settle) = mv_grid.set_default_branch_type(
            debug=True
        )

        # set default branch kinds
        mv_grid.default_branch_kind_aggregated = mv_grid.default_branch_kind
        mv_grid.default_branch_kind_settle = 'cable'

        # choose appropriate transformers for each HV/MV sub-station
        mv_grid._station.select_transformers()

        # parameterize the network
        mv_grid.network.mv_parametrize_grid(debug=True)

        # build the lv_grids
        mv_grid.network.build_lv_grids()

        return network, mv_grid, lv_stations

    def test_local_routing(self, minimal_unrouted_grid):
        """
        Rigorous test to the function :meth:`~.core.network.grids.MVGridDing0.routing`
        """
        nd, mv_grid, lv_stations = minimal_unrouted_grid

        graph = mv_grid._graph

        # pre-routing asserts
        # check the grid_district
        assert mv_grid.grid_district.id_db == 1000
        assert len(list(mv_grid.grid_district.lv_load_areas())) == 19
        assert len(list(mv_grid.grid_district.lv_load_area_groups())) == 0
        assert mv_grid.grid_district.peak_load == pytest.approx(
            3834.695583,
            abs=0.001
        )

        # check mv_grid graph attributes
        assert len(list(graph.nodes())) == 43
        assert len(list(graph.edges())) == 0
        assert len(list(nx.isolates(graph))) == 43
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

        # do the routing
        nd.mv_routing()

        # post-routing asserts
        # check that the connections are between the expected
        # load areas
        mv_station = mv_grid.station()
        mv_cable_distributors = sorted(list(mv_grid.cable_distributors()), key=lambda x: repr(x))
        expected_edges_list = [
            (mv_station, mv_cable_distributors[0]),
            (mv_station, mv_cable_distributors[1]),
            (mv_station, mv_cable_distributors[2]),
            (mv_station, lv_stations[0]),
            (mv_station, lv_stations[1]),
            (mv_station, lv_stations[5]),
            (mv_station, lv_stations[8]),
            (mv_station, lv_stations[9]),
            (mv_station, lv_stations[10]),
            (mv_station, lv_stations[13]),
            (mv_station, lv_stations[14]),
            (mv_station, lv_stations[16]),
            (lv_stations[0], lv_stations[12]),
            (lv_stations[1], lv_stations[11]),
            (lv_stations[11], lv_stations[12]),
            (lv_stations[13], lv_stations[17]),
            (lv_stations[14], lv_stations[15]),
            (lv_stations[15], lv_stations[16]),
            (lv_stations[17], mv_cable_distributors[3]),
            (lv_stations[18], lv_stations[2]),
            (lv_stations[18], mv_cable_distributors[3]),
            (lv_stations[3], lv_stations[7]),
            (lv_stations[4], mv_cable_distributors[4]),
            (lv_stations[5], mv_cable_distributors[4]),
            (lv_stations[6], lv_stations[7]),
            (lv_stations[7], mv_cable_distributors[3]),
            (lv_stations[7], mv_cable_distributors[4]),
            (lv_stations[8], mv_cable_distributors[0]),
            (lv_stations[9], mv_cable_distributors[1]),
            (lv_stations[10], mv_cable_distributors[2]),
        ]

        assert set(expected_edges_list) == set(graph.edges)

        # check graph attributes
        assert len(list(graph.nodes())) == 35
        assert len(list(graph.edges())) == 30
        assert len(list(nx.isolates(graph))) == 10
        assert pd.Series(dict(graph.degree())).sum(axis=0) == 60
        assert pd.Series(
            dict(graph.degree())
        ).mean(axis=0) == pytest.approx(1.714, 0.001)
        assert len(list(nx.get_edge_attributes(graph, 'branch'))) == 30
        assert nx.average_node_connectivity(graph) == pytest.approx(
            0.5815,
            abs=0.0001
        )
        assert pd.Series(
            nx.degree_centrality(graph)
        ).mean(axis=0) == pytest.approx(0.0504, abs=0.001)
        assert pd.Series(
            nx.closeness_centrality(graph)
        ).mean(axis=0) == pytest.approx(0.16379069, abs=0.00001)
        assert pd.Series(
            nx.betweenness_centrality(graph)
        ).mean(axis=0) == pytest.approx(0.033613445, abs=0.00001)
        assert pd.Series(
            nx.edge_betweenness_centrality(graph)
        ).mean(axis=0) == pytest.approx(0.05378151, abs=0.00001)

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
