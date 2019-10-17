import pytest

from ding0.tools.tools import (get_cart_dest_point, create_poly_from_source)
#from ding0.flexopt.check_tech_constraints import get_critical_line_loading

from ding0.core.network.loads import LVLoadDing0
from ding0.core.network.cable_distributors import LVCableDistributorDing0

from shapely.geometry import Point
import pandas as pd
import networkx as nx
from ding0.core.network import (TransformerDing0,
                                RingDing0, BranchDing0,
                                CircuitBreakerDing0,
                                GeneratorDing0, GeneratorFluctuatingDing0)
from ding0.core.network.grids import MVGridDing0, LVGridDing0
from ding0.core.network.stations import MVStationDing0, LVStationDing0
from ding0.core.structure.regions import (MVGridDistrictDing0,
                                          LVLoadAreaDing0,
                                          LVLoadAreaCentreDing0,
                                          LVGridDistrictDing0)
import numpy as np
from math import isnan


from ding0.core import NetworkDing0

from egoio.tools import db
from sqlalchemy.orm import sessionmaker


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
        engine = db.connection(readonly=True)
        session = sessionmaker(bind=engine)()
        yield session
        print("closing session")
        session.close()

    def minimal_unrouted_testgrid(self):
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
                id_db=1,
                s_max_longterm=63000.0,
                v_level=20.0
            ),
            TransformerDing0(
                id_db=2,
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
                     R_per_km=0.37,
                     L_per_km=1.18,
                     C_per_km=0.0098,
                     reinforce_only=0)
            ),
            default_branch_type_aggregated=pd.Series(
                dict(name='122-AL1/20-ST1A',
                     U_n=20.0,
                     I_max_th=410.0,
                     R_per_km=0.34,
                     L_per_km=1.08,
                     C_per_km=0.0106,
                     reinforce_only=0)
            ),
            default_branch_type_settle=pd.Series(
                dict(name='NA2XS2Y 3x1x150 RE/25',
                     U_n=20.0,
                     I_max_th=319.0,
                     R_per_km=0.206,
                     L_per_km=0.4011,
                     C_per_km=0.24,
                     reinforce_only=0)
            )
        )

        for hvmv_transfromer in hvmv_transformers:
            hvmv_transfromer.grid = mv_grid

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
                id_db=id_db,
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

        lv_stations = sorted(lv_stations, key=lambda x: x.id_db)
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

    @pytest.fixture
    def minimal_unrouted_grid(self):
        return self.minimal_unrouted_testgrid()

    def test_local_routing(self, minimal_unrouted_grid):
        """
        Rigorous test to the function :meth:`~.core.network.grids.MVGridDing0.routing`
        """
        nd, mv_grid, lv_stations = minimal_unrouted_grid

        graph = mv_grid._graph

        # pre-routing asserts
        # check the grid_district
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
        expected_edges_list = [
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
            (lv_stations[18], lv_stations[2]),
            (lv_stations[3], lv_stations[7]),
            (lv_stations[6], lv_stations[7]),
        ]


        #real edges sorted
        real_edges_sort = []
        for i in range(0,len(list(graph.edges()))):
            real_edges_sort.append(
                tuple(sorted(list(graph.edges())[i], key=lambda x: repr(x))))

        #sort the tuples and compare them
        correct_edges = []
        for i in range(0, len(expected_edges_list)):
            if tuple(sorted(expected_edges_list[i], key=lambda x:repr(x))) in\
                    real_edges_sort:
                correct_edges.append(True)
            else:
                correct_edges.append((False))
                print('Tuple', expected_edges_list[i], 'not in set')


        #check if every tuple of the expected edges is in the list, disregarding order
        assert len(list(filter(lambda x: x == True, correct_edges))) == 18

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

    def test_construct(self, minimal_unrouted_grid):

        """Creates a synthetic mv grid containing enough nodes
        for the routing algorithm to work and verifies:
        -Number of mv grid connections
        -Number of rings
        -Right connections for the cable distr.
        -Right type of cable for the circuit breakers"""

        network, mv_grid, lv_stations =minimal_unrouted_grid
        lv_stations = sorted(lv_stations, key=lambda x: x.id_db)

        network.mv_routing(debug=True)

        #Check mv_stations connections (generators should not be connected)
        assert len(list(mv_grid._graph.adj[mv_grid._station])) == 12
        assert any(j in list(mv_grid._graph.adj[mv_grid._station])
                   for j in mv_grid._generators[0:4]) == False

        b = [0,1,13,16,17,18,4,5,7]
        lv_stations_mvstat = [lv_stations[i] for i in b]

        assert all(j in list(mv_grid._graph.adj[mv_grid._station])
                   for j in lv_stations_mvstat) == True

        assert len(mv_grid._rings) == 6

        """Check cable Distributors have right connections
        (Single Connections)"""

        assert len(mv_grid._cable_distributors) == 5
        assert any(y in mv_grid._graph.adj[lv_stations[8]]
                   for y in mv_grid._cable_distributors)

        assert any(y in mv_grid._graph.adj[lv_stations[9]]
                   for y in mv_grid._cable_distributors)

        assert any(y in mv_grid._graph.adj[lv_stations[12]]
                   for y in mv_grid._cable_distributors)

        assert any(y in mv_grid._graph.adj[lv_stations[13]]
                   for y in mv_grid._cable_distributors)

        assert any(y in mv_grid._graph.adj[lv_stations[16]]
                   for y in mv_grid._cable_distributors)

        assert any(y in mv_grid._graph.adj[lv_stations[17]]
                   for y in mv_grid._cable_distributors)

        assert any(y in mv_grid._graph.adj[lv_stations[18]]
                   for y in mv_grid._cable_distributors)

        """Check cable Distributors have right connections
        (Double Connections disregarding order)
        """
        mv_stat_cdt = list(y in mv_grid._graph.adj[mv_grid._station]
                           for y in mv_grid._cable_distributors)

        lv_15_cdt = list(y in mv_grid._graph.adj[lv_stations[15]]
                         for y in mv_grid._cable_distributors)

        assert len(list(filter(lambda x: x == True, mv_stat_cdt))) == 3
        assert len(list(filter(lambda x: x == True, lv_15_cdt))) == 2


        #Thermal Capacity of circuit breaker's branch
        mv_grid._circuit_breakers[0].branch.type['I_max_th'] == 210
        mv_grid._circuit_breakers[1].branch.type['I_max_th'] == 210
        mv_grid._circuit_breakers[2].branch.type['I_max_th'] == 210
        mv_grid._circuit_breakers[3].branch.type['I_max_th'] == 210
        mv_grid._circuit_breakers[4].branch.type['I_max_th'] == 210
        mv_grid._circuit_breakers[5].branch.type['I_max_th'] == 210

    @pytest.fixture
    def lv_grid_district_data(self):
        """
        Creates a pandas DataFrame containing the data
        needed for the creation of an LVGridDisctrict
        """
        distance_scaling_factor = 3
        source = Point(8.638204, 49.867307)

        lv_grid_district_data = pd.DataFrame(
            dict(
                la_id=1,
                population=0,
                peak_load_residential=0.0,
                peak_load_retail=0.0,
                peak_load_industrial=0.0,
                peak_load_agricultural=0.0,
                geom=create_poly_from_source(
                        get_cart_dest_point(source,
                                            -1000*distance_scaling_factor,
                                            1000*distance_scaling_factor),
                        100,
                        100,
                        100,
                        100)

            ), index=[0]
        )

        lv_grid_district_data.loc[:, 'peak_load'] = (
                lv_grid_district_data.loc[:, ['peak_load_residential',
                                              'peak_load_retail',
                                              'peak_load_industrial',
                                              'peak_load_agricultural']].sum(axis=1))
        return lv_grid_district_data

    @pytest.fixture
    def basic_lv_grid(self, lv_grid_district_data):
        """
        Minimal needed LVGridDing0 to run LVGridDing0.build_grid
        """
        #ToDo: Check if definition of own function makes sense so both MVTest and LVTest can use it. Makes changes easier and reduces code.
        source = Point(8.638204, 49.867307)
        distance_scaling_factor = 3

        network = NetworkDing0(name='network')
        mv_station = MVStationDing0(
            id_db=0,
            geo_data=source,
            peak_load=10000.0,
            v_level_operation=20.0
        )
        hvmv_transformer = TransformerDing0(
                id_db=0,
                s_max_longterm=63000.0,
                v_level=20.0)

        mv_station.add_transformer(hvmv_transformer)

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
                     R_per_km=0.37,
                     L_per_km=1.18,
                     C_per_km=0.0098,
                     reinforce_only=0)
            ),
            default_branch_type_aggregated=pd.Series(
                dict(name='122-AL1/20-ST1A',
                     U_n=20.0,
                     I_max_th=410.0,
                     R_per_km=0.34,
                     L_per_km=1.08,
                     C_per_km=0.0106,
                     reinforce_only=0)
            ),
            default_branch_type_settle=pd.Series(
                dict(name='NA2XS2Y 3x1x150 RE/25',
                     U_n=20.0,
                     I_max_th=319.0,
                     R_per_km=0.206,
                     L_per_km=0.4011,
                     C_per_km=0.24,
                     reinforce_only=0)
            )
        )
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

            GeneratorFluctuatingDing0(
                id_db=2,
                weather_cell_id=1,
                capacity=1000.0,
                mv_grid=mv_grid,
                type='wind',
                subtype='wind_onshore',
                v_level=5,
                geo_data=source
            )
        ]
        # 2 Generator: 1 Constant,1 Fluctating
        for mv_gen in mv_generators:
            mv_grid.add_generator(mv_gen)

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
        network.add_mv_grid_district(mv_grid_district)

        lv_grid_district_data.loc[:, 'peak_load'] = (
            lv_grid_district_data.loc[:, ['peak_load_residential',
                                           'peak_load_retail',
                                           'peak_load_industrial',
                                           'peak_load_agricultural']].sum(axis=1)
        )
        lv_nominal_voltage = 400.0
        # assuming that the lv_grid districts and lv_load_areas are the same!
        # or 1 lv_griddistrict is 1 lv_loadarea
        lv_load_area = LVLoadAreaDing0(id_db=0,
                                       db_data=lv_grid_district_data.iloc[0],
                                       mv_grid_district=mv_grid_district,
                                       peak_load=lv_grid_district_data['peak_load'])

        lv_load_area.geo_area = lv_grid_district_data['geom'][0]
        lv_load_area.geo_centre = lv_grid_district_data['geom'][0].centroid
        lv_grid_district = LVGridDistrictDing0(
            id_db=27,
            lv_load_area=lv_load_area,
            geo_data=lv_grid_district_data['geom'][0],
            population=(0
                        if isnan(lv_grid_district_data['population'][0])
                        else int(lv_grid_district_data['population'][0])),
            peak_load_residential=lv_grid_district_data['peak_load_residential'][0],
            peak_load_retail=lv_grid_district_data['peak_load_retail'][0],
            peak_load_industrial=lv_grid_district_data['peak_load_industrial'][0],
            peak_load_agricultural=lv_grid_district_data['peak_load_agricultural'][0],
            peak_load=(lv_grid_district_data['peak_load_residential'][0]
                       + lv_grid_district_data['peak_load_retail'][0]
                       + lv_grid_district_data['peak_load_industrial'][0]
                       + lv_grid_district_data['peak_load_agricultural'][0]),
            sector_count_residential=(0
                                      if lv_grid_district_data['peak_load_residential'][0] == 0.0
                                      else 1),
            sector_count_retail=(0
                                 if lv_grid_district_data['peak_load_retail'][0] == 0.0
                                 else 1),
            sector_count_agricultural=(0
                                       if lv_grid_district_data['peak_load_agricultural'][0] == 0.0
                                       else 1),
            sector_count_industrial=(0
                                     if lv_grid_district_data['peak_load_industrial'][0] == 0.0
                                     else 1),
            sector_consumption_residential=lv_grid_district_data['peak_load_residential'][0],
            sector_consumption_retail=lv_grid_district_data['peak_load_retail'][0],
            sector_consumption_industrial=lv_grid_district_data['peak_load_industrial'][0],
            sector_consumption_agricultural=lv_grid_district_data['peak_load_agricultural'][0]
        )

        #be aware, lv_grid takes grid district's geom!
        lv_grid = LVGridDing0(network=network,
                              grid_district=lv_grid_district,
                              id_db=1,
                              geo_data=lv_grid_district_data['geom'][0],
                              v_level=lv_nominal_voltage)

        # create LV station
        lv_station = LVStationDing0(
            id_db=1,
            grid=lv_grid,
            lv_load_area=lv_load_area,
            geo_data=lv_grid_district_data['geom'][0].centroid,
            peak_load=lv_grid_district.peak_load
        )

        # assign created objects
        # note: creation of LV grid is done separately,
        # see NetworkDing0.build_lv_grids()
        lv_grid.add_station(lv_station)
        lv_grid_district.lv_grid = lv_grid
        lv_load_area.add_lv_grid_district(lv_grid_district)

        lv_load_area_centre = LVLoadAreaCentreDing0(id_db=1,
                                                    geo_data=lv_grid_district_data['geom'][0].centroid,
                                                    lv_load_area=lv_load_area,
                                                    grid=mv_grid_district.mv_grid)

        lv_load_area.lv_load_area_centre = lv_load_area_centre
        mv_grid_district.add_lv_load_area(lv_load_area)

        return lv_grid



class TestLVGridDing0(object):

    @pytest.fixture
    def lv_grid_district_data(self):
        """
        Creates a pandas DataFrame containing the data
        needed for the creation of an LVGridDisctrict
        """
        distance_scaling_factor = 3
        source = Point(8.638204, 49.867307)

        lv_grid_district_data = pd.DataFrame(
            dict(
                la_id=1,
                population=0,
                peak_load_residential=0.0,
                peak_load_retail=0.0,
                peak_load_industrial=0.0,
                peak_load_agricultural=0.0,
                geom=create_poly_from_source(
                        get_cart_dest_point(source,
                                            -1000*distance_scaling_factor,
                                            1000*distance_scaling_factor),
                        100,
                        100,
                        100,
                        100)

            ), index=[0]
        )

        lv_grid_district_data.loc[:, 'peak_load'] = (
                lv_grid_district_data.loc[:, ['peak_load_residential',
                                              'peak_load_retail',
                                              'peak_load_industrial',
                                              'peak_load_agricultural']].sum(axis=1))
        return lv_grid_district_data


    @pytest.fixture
    def basic_lv_grid(self, lv_grid_district_data):
        """
        Minimal needed LVGridDing0 to run LVGridDing0.build_grid
        """
        # ToDo: Check if definition of own function makes sense so both MVTest and LVTest can use it. Makes changes easier and reduces code.

        source = Point(8.638204, 49.867307)
        distance_scaling_factor = 3

        network = NetworkDing0(name='network')
        mv_station = MVStationDing0(
            id_db=0,
            geo_data=source,
            peak_load=10000.0,
            v_level_operation=20.0
        )
        hvmv_transformer = TransformerDing0(
                id_db=0,
                s_max_longterm=63000.0,
                v_level=20.0)

        mv_station.add_transformer(hvmv_transformer)

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
                     R_per_km=0.37,
                     L_per_km=1.18,
                     C_per_km=0.0098,
                     reinforce_only=0)
            ),
            default_branch_type_aggregated=pd.Series(
                dict(name='122-AL1/20-ST1A',
                     U_n=20.0,
                     I_max_th=410.0,
                     R_per_km=0.34,
                     L_per_km=1.08,
                     C_per_km=0.0106,
                     reinforce_only=0)
            ),
            default_branch_type_settle=pd.Series(
                dict(name='NA2XS2Y 3x1x150 RE/25',
                     U_n=20.0,
                     I_max_th=319.0,
                     R_per_km=0.206,
                     L_per_km=0.4011,
                     C_per_km=0.24,
                     reinforce_only=0)
            )
        )
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

            GeneratorFluctuatingDing0(
                id_db=2,
                weather_cell_id=1,
                capacity=1000.0,
                mv_grid=mv_grid,
                type='wind',
                subtype='wind_onshore',
                v_level=5,
                geo_data=source
            )
        ]
        # 2 Generator: 1 Constant,1 Fluctating
        for mv_gen in mv_generators:
            mv_grid.add_generator(mv_gen)

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
        network.add_mv_grid_district(mv_grid_district)

        lv_grid_district_data.loc[:, 'peak_load'] = (
            lv_grid_district_data.loc[:, ['peak_load_residential',
                                           'peak_load_retail',
                                           'peak_load_industrial',
                                           'peak_load_agricultural']].sum(axis=1)
        )
        lv_nominal_voltage = 400.0
        # assuming that the lv_grid districts and lv_load_areas are the same!
        # or 1 lv_griddistrict is 1 lv_loadarea
        lv_load_area = LVLoadAreaDing0(id_db=0,
                                       db_data=lv_grid_district_data.iloc[0],
                                       mv_grid_district=mv_grid_district,
                                       peak_load=lv_grid_district_data['peak_load'])

        lv_load_area.geo_area = lv_grid_district_data['geom'][0]
        lv_load_area.geo_centre = lv_grid_district_data['geom'][0].centroid
        lv_grid_district = LVGridDistrictDing0(
            id_db=27,
            lv_load_area=lv_load_area,
            geo_data=lv_grid_district_data['geom'][0],
            population=(0
                        if isnan(lv_grid_district_data['population'][0])
                        else int(lv_grid_district_data['population'][0])),
            peak_load_residential=lv_grid_district_data['peak_load_residential'][0],
            peak_load_retail=lv_grid_district_data['peak_load_retail'][0],
            peak_load_industrial=lv_grid_district_data['peak_load_industrial'][0],
            peak_load_agricultural=lv_grid_district_data['peak_load_agricultural'][0],
            peak_load=(lv_grid_district_data['peak_load_residential'][0]
                       + lv_grid_district_data['peak_load_retail'][0]
                       + lv_grid_district_data['peak_load_industrial'][0]
                       + lv_grid_district_data['peak_load_agricultural'][0]),
            sector_count_residential=(0
                                      if lv_grid_district_data['peak_load_residential'][0] == 0.0
                                      else 1),
            sector_count_retail=(0
                                 if lv_grid_district_data['peak_load_retail'][0] == 0.0
                                 else 1),
            sector_count_agricultural=(0
                                       if lv_grid_district_data['peak_load_agricultural'][0] == 0.0
                                       else 1),
            sector_count_industrial=(0
                                     if lv_grid_district_data['peak_load_industrial'][0] == 0.0
                                     else 1),
            sector_consumption_residential=lv_grid_district_data['peak_load_residential'][0],
            sector_consumption_retail=lv_grid_district_data['peak_load_retail'][0],
            sector_consumption_industrial=lv_grid_district_data['peak_load_industrial'][0],
            sector_consumption_agricultural=lv_grid_district_data['peak_load_agricultural'][0]
        )

        #be aware, lv_grid takes grid district's geom!
        lv_grid = LVGridDing0(network=network,
                              grid_district=lv_grid_district,
                              id_db=1,
                              geo_data=lv_grid_district_data['geom'][0],
                              v_level=lv_nominal_voltage)

        # create LV station
        lv_station = LVStationDing0(
            id_db=1,
            grid=lv_grid,
            lv_load_area=lv_load_area,
            geo_data=lv_grid_district_data['geom'][0].centroid,
            peak_load=lv_grid_district.peak_load
        )

        # assign created objects
        # note: creation of LV grid is done separately,
        # see NetworkDing0.build_lv_grids()
        lv_grid.add_station(lv_station)
        lv_grid_district.lv_grid = lv_grid
        lv_load_area.add_lv_grid_district(lv_grid_district)

        lv_load_area_centre = LVLoadAreaCentreDing0(id_db=1,
                                                    geo_data=lv_grid_district_data['geom'][0].centroid,
                                                    lv_load_area=lv_load_area,
                                                    grid=mv_grid_district.mv_grid)

        lv_load_area.lv_load_area_centre = lv_load_area_centre
        mv_grid_district.add_lv_load_area(lv_load_area)
        mv_grid.routing()

        return lv_grid

    @pytest.fixture
    def empty_lvgridding0(self,lv_grid_district_data):
        """
        Returns and empty LVGridDing0 object, belonging
        to an LVGridDisctrict with the corresponding
        fixture data
        """

        network = NetworkDing0(name='network')
        mv_station = MVStationDing0(network=network)
        mv_grid = MVGridDing0(station=mv_station, network=network)
        mv_grid_district = MVGridDistrictDing0(mv_grid=mv_grid)

        lv_load_area = LVLoadAreaDing0(id_db=0,
                                       db_data=lv_grid_district_data.iloc[0],
                                       mv_grid_district=mv_grid_district,
                                       peak_load=lv_grid_district_data['peak_load'])

        lv_load_area.geo_area = lv_grid_district_data['geom'][0]
        lv_load_area.geo_centre = lv_grid_district_data['geom'][0].centroid
        lv_grid_district = LVGridDistrictDing0(
            id_db=id,
            lv_load_area=lv_load_area,
            geo_data=lv_grid_district_data['geom'],
            population=(0
                        if isnan(lv_grid_district_data['population'])
                        else int(lv_grid_district_data['population'])),
            peak_load_residential=lv_grid_district_data['peak_load_residential'],
            peak_load_retail=lv_grid_district_data['peak_load_retail'],
            peak_load_industrial=lv_grid_district_data['peak_load_industrial'],
            peak_load_agricultural=lv_grid_district_data['peak_load_agricultural'],
            peak_load=(lv_grid_district_data['peak_load_residential']
                       + lv_grid_district_data['peak_load_retail']
                       + lv_grid_district_data['peak_load_industrial']
                       + lv_grid_district_data['peak_load_agricultural']),
            sector_count_residential=(0
                                      if lv_grid_district_data['peak_load_residential'][0] == 0.0
                                      else 1),
            sector_count_retail=(0
                                 if lv_grid_district_data['peak_load_retail'][0] == 0.0
                                 else 1),
            sector_count_agricultural=(0
                                       if lv_grid_district_data['peak_load_agricultural'][0] == 0.0
                                       else 1),
            sector_count_industrial=(0
                                     if lv_grid_district_data['peak_load_industrial'][0] == 0.0
                                     else 1),
            sector_consumption_residential=lv_grid_district_data['peak_load_residential'][0],
            sector_consumption_retail=lv_grid_district_data['peak_load_retail'][0],
            sector_consumption_industrial=lv_grid_district_data['peak_load_industrial'][0],
            sector_consumption_agricultural=lv_grid_district_data['peak_load_agricultural'][0]
        )
        grid = LVGridDing0(id_db=0,
                           grid_district=lv_grid_district,
                           population=lv_grid_district.population,
                           network=network)

        return grid, lv_load_area

    def test_empty_grid(self, empty_lvgridding0):
        """
        Check that the initialization of an object of the
        class LVGridDing0 results in the right attributes being empty
        lists or NoneType objects, with the exception of the
        LVStationDing0 object's id_db and geo_data, which are
        0 and shapely.geometry.Point(1, 1) respectively.
        Check if the type of branch is set to cable by default
        """
        empty_lvgridding0,lv_load_area = empty_lvgridding0
        assert empty_lvgridding0.default_branch_kind == 'cable'
        assert empty_lvgridding0._station is None
        assert empty_lvgridding0.population == 0

    def test_station(self, empty_lvgridding0):
        """
        Check if station function returns the current
        given station
        """
        empty_lvgridding0,lv_load_area = empty_lvgridding0
        assert empty_lvgridding0.station() == empty_lvgridding0._station

    def test_add_station(self, empty_lvgridding0):
        """
        Check if a new station is added correctly
        """
        lv_grid, lv_load_area = empty_lvgridding0
        new_lv = LVStationDing0(id_db=0,
                                geo_data=(1,2),
                                lv_grid=lv_grid,
                                lv_load_area=lv_load_area)

        graph = lv_grid._graph
        n = nx.number_of_nodes(graph)
        lv_grid.add_station(new_lv)
        m = nx.number_of_nodes(graph)
        assert m == n+1

    def test_add_station_negative(self, empty_lvgridding0):
        """
        Check if adding a non LVStation object is added as a station
        raises the proper error
        """
        lv_grid, lv_load_area = empty_lvgridding0
        bad_station = GeneratorDing0(id_db=0)
        with pytest.raises(Exception, match='not'):
            lv_grid.add_station(bad_station)

    def test_add_load(self, empty_lvgridding0):
        """
        Check if load gets added correctly
        """
        empty_lvgridding0,lv_load_area = empty_lvgridding0
        new_lv_load = LVLoadDing0(grid=empty_lvgridding0)
        empty_lvgridding0.add_load(new_lv_load)
        assert empty_lvgridding0.loads_count() == 1

    def test_add_cable_dist(self, empty_lvgridding0):
        """
        Check if cable distributor is added properly
        """
        empty_lvgridding0,lv_load_area = empty_lvgridding0
        new_cable_dist = LVCableDistributorDing0(grid=empty_lvgridding0)
        empty_lvgridding0.add_cable_dist(new_cable_dist)
        assert empty_lvgridding0.cable_distributors_count() == 1

    def test_build_grid_transformers(self, basic_lv_grid):
        """
        Check if transformers are added correctly to
        the grid. Transformers are added according to
        s_max in load case.
        """
        #s_max / cosphi < 1000
        basic_lv_grid.grid_district.peak_load = 969
        basic_lv_grid.build_grid()
        assert len(basic_lv_grid.station()._transformers) == 1
        basic_lv_grid.station()._transformers = [] #reset transformers

        #s_max / cosphi = 1000
        basic_lv_grid.grid_district.peak_load = 970
        basic_lv_grid.build_grid()
        assert len(basic_lv_grid.station()._transformers) == 2
        basic_lv_grid.station()._transformers = [] #reset transformers

        #s_max / cosphi > 1000
        basic_lv_grid.grid_district.peak_load = 971
        basic_lv_grid.build_grid()
        assert len(basic_lv_grid.station()._transformers) == 2

    def test_build_grid_ria_branches(self, basic_lv_grid):
        """
        Check if the correct number of branches and nodes
        is created. As the peak load for retail/industrial
        areas is surpassed, the number of loads created
        doubles by redistributing the load.
        """
        basic_lv_grid.grid_district.peak_load_retail = 564
        basic_lv_grid.grid_district.peak_load_industrial = 140
        basic_lv_grid.grid_district.peak_load_agricultural = 280

        basic_lv_grid.grid_district.sector_count_retail = 1
        basic_lv_grid.grid_district.sector_count_industrial = 1
        basic_lv_grid.grid_district.sector_count_agricultural = 5

        basic_lv_grid.build_grid()
        assert len(basic_lv_grid._loads) == 9
        assert len(list(basic_lv_grid._graph.nodes)) == 28
        assert (basic_lv_grid._loads[n].peak_load == 176 for n in range(0, 4))
        assert (basic_lv_grid._loads[n].peak_load == 56 for n in range(4, 9))

    def test_build_grid_residential_branches(self, basic_lv_grid):
        """
        Verifies that the number of loads and nodes
        created correspond to the peak_load and population given for
        the residential area. Additionally checks
        if the load value of every node corresponds
        to the predicted one and if the strings
        used are the ones defined by the heuristic.
        """

        basic_lv_grid.grid_district.population = 100
        basic_lv_grid.grid_district.peak_load_residential = 300
        basic_lv_grid.build_grid()

        assert len(basic_lv_grid._loads) == 29
        assert len(basic_lv_grid._graph.nodes) == 29 + 2*29 + 1
        assert (round(basic_lv_grid._loads[n].peak_load) == 10.0
                for n in range(0, 29))

        #2 Branches from LV_station
        assert len(np.nonzero(nx.adjacency_matrix(basic_lv_grid._graph)[0, :]
                              .toarray())) == 2

    def test_connect_generators(self, basic_lv_grid):
        """
        Check if generator is added to the graph
        """
        new_gen = GeneratorDing0()
        basic_lv_grid.add_generator(new_gen)
        assert len(basic_lv_grid._generators) == 1

    def test_reinforce_grid(self, basic_lv_grid):
        """Verifies the correct creation of an LV_Grid Graph,
        regarding number of nodes, as well as type of cables and
        then checks if the grid reinforcement is done correctly
        by inspecting this attributes for the new graph
        """

        basic_lv_grid.grid_district.population = 20
        basic_lv_grid.grid_district.peak_load_residential = 200
        basic_lv_grid.build_grid()
        #crit_branches,crit_stations = get_critical_line_loading(basic_lv_grid)

        #from ding0.flexopt.check_tech_constraints import get_critical_line_loading

        # Verify the LV_Grid._graph gets constructed correctly and that it only has one
        #Transformer
        assert len(list(basic_lv_grid._graph.edges(basic_lv_grid._station)))\
               == 1

        assert len(basic_lv_grid._station._transformers) == 1

        # Branches HH_1,HH_2,HH_3 should have as default the corresponding cable
        assert basic_lv_grid._graph._adj[basic_lv_grid._station]\
                   [basic_lv_grid._cable_distributors[0]]['branch'].type.name == "NAYY 4x1x150"

        assert basic_lv_grid._graph._adj[basic_lv_grid._cable_distributors[2]]\
                   [basic_lv_grid._cable_distributors[0]]['branch'].type.name == "NAYY 4x1x150"

        assert basic_lv_grid._graph._adj[basic_lv_grid._cable_distributors[4]]\
                   [basic_lv_grid._cable_distributors[2]]['branch'].type.name == "NAYY 4x1x150"

        basic_lv_grid.reinforce_grid()
        basic_lv_grid.build_grid()


        # Verify that the station is reinforced with another transformer
        assert len(list(basic_lv_grid._graph.edges(basic_lv_grid._station))) \
               == 2

        assert len(basic_lv_grid._station._transformers) == 2

        #Verify that the critical branches get the cables changed with the correct ones
        assert basic_lv_grid._graph._adj[basic_lv_grid._station] \
                   [basic_lv_grid._cable_distributors[0]]['branch'].type.name == "NAYY 4x1x300"

        assert basic_lv_grid._graph._adj[basic_lv_grid._cable_distributors[2]]\
                   [basic_lv_grid._cable_distributors[0]]['branch'].type.name == "NAYY 4x1x240"

        assert basic_lv_grid._graph._adj[basic_lv_grid._cable_distributors[4]]\
                   [basic_lv_grid._cable_distributors[2]]['branch'].type.name == "NAYY 4x1x240"


if __name__ == "__main__":
   pass