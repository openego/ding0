import pytest
from shapely.geometry import Point, LineString, LinearRing, Polygon
from ding0.core import NetworkDing0
from ding0.core.network import (GridDing0,
                                StationDing0, TransformerDing0,
                                RingDing0, BranchDing0,
                                CableDistributorDing0, CircuitBreakerDing0,
                                GeneratorDing0, GeneratorFluctuatingDing0,
                                LoadDing0)
from ding0.core.structure.regions import LVLoadAreaCentreDing0
from ding0.tools.results import (calculate_lvgd_stats, calculate_lvgd_voltage_current_stats, calculate_mvgd_stats,
                                 calculate_mvgd_voltage_current_stats)
import pandas as pd
from pandas.util.testing import assert_frame_equal
import os
import numpy as np
from egoio.tools import db
from sqlalchemy.orm import sessionmaker


class TestGridDing0(object):

    @pytest.fixture
    def empty_grid(self):
        """
        Returns an empty GridDing0 object.
        """
        return GridDing0()

    @pytest.fixture
    def cable_distributor_grid(self):
        """
        Returns a GridDing0 object with 2 CableDistributorDing0
        objects at Point(0, 0) and id_db=0,
        and at Point(0, 1) and id_db=1
        """
        cable_distributor_grid = GridDing0()
        geo_data1 = Point(0, 0)
        cable_distributor1 = CableDistributorDing0(id_db=0,
                                                   geo_data=geo_data1,
                                                   grid=cable_distributor_grid)
        geo_data2 = Point(0, 1)
        cable_distributor2 = CableDistributorDing0(id_db=1,
                                                   geo_data=geo_data2,
                                                   grid=cable_distributor_grid)
        # There is no setter function for providing a list
        # of cable_distributors to the empty_grid
        cable_distributor_grid._cable_distributors = [cable_distributor1,
                                                      cable_distributor2]
        return cable_distributor_grid

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

    def test_get_cable_distributors_list(self, cable_distributor_grid):
        cable_distributor_list = list(
            cable_distributor_grid.cable_distributors()
            )
        assert len(cable_distributor_list) == 2
        assert cable_distributor_list[0].id_db == 0
        assert cable_distributor_list[0].geo_data == Point(0, 0)
        assert cable_distributor_list[1].id_db == 1
        assert cable_distributor_list[1].geo_data == Point(0, 1)

    def test_cable_distributors_count(self, cable_distributor_grid):
        assert cable_distributor_grid.cable_distributors_count() == 2

    @pytest.fixture
    def load_grid(self):
        """
        Returns a GridDing0 object with 2 LoadDing0
        objects at Point(0, 0) and id_db=0, and
        another at Point(0, 1) and id_db=1,
        Note: this function causes the id_db to be increased by 1.
        Thus changing the id_db to 1 and 2 respectively
        """
        load_grid = GridDing0()
        geo_data1 = Point(0, 0)
        load1 = LoadDing0(id_db=0,
                          geo_data=geo_data1,
                          grid=load_grid)
        # There is no setter function for providing a list
        # of loads to the empty_grid
        load_grid._loads = [load1]
        geo_data2 = Point(0, 1)
        load2 = LoadDing0(id_db=0,
                          geo_data=geo_data2,
                          grid=load_grid)
        load_grid._loads.append(load2)
        return load_grid

    def test_loads_list(self, load_grid):
        load_list = list(load_grid.loads())
        assert len(load_list) == 2
        assert load_list[0].id_db == 1
        assert load_list[0].geo_data == Point(0, 0)
        assert load_list[1].id_db == 2
        assert load_list[1].geo_data == Point(0, 1)

    def test_loads_count(self, load_grid):
        assert load_grid.loads_count() == 2

    @pytest.fixture
    def generator_grid(self):
        """
        Returns a GridDing0 object with
        2 GeneratorDing0 objects
        at Point(0, 0) and id_db=0 and
        at Point(0, 1) and id_db=1, and
        2 GeneratorFluctuatingDing0 objects
        at Point(1, 0), id_db=2, weather_cell_id=0 and
        at Point(1, 1), id_db=3, weather_cell_id=1
        """
        generator_grid = GridDing0()
        geo_data1 = Point(0, 0)
        generator1 = GeneratorDing0(id_db=0,
                                    geo_data=geo_data1)
        geo_data2 = Point(0, 1)
        generator2 = GeneratorDing0(id_db=1,
                                    geo_data=geo_data2)
        geo_data3 = Point(1, 0)
        generator3 = GeneratorFluctuatingDing0(id_db=2,
                                               weather_cell_id=0,
                                               geo_data=geo_data3)
        geo_data4 = Point(1, 1)
        generator4 = GeneratorFluctuatingDing0(id_db=3,
                                               weather_cell_id=1,
                                               geo_data=geo_data4)
        generator_grid._generators = [generator1,
                                      generator2,
                                      generator3,
                                      generator4]
        return generator_grid

    def test_add_generator(self, empty_grid):
        geo_data1 = Point(0, 0)
        generator1 = GeneratorDing0(id_db=0,
                                    geo_data=geo_data1)
        empty_grid.add_generator(generator1)
        assert len(list(empty_grid.generators())) == 1
        geo_data2 = Point(0, 1)
        generator2 = GeneratorDing0(id_db=1,
                                    geo_data=geo_data2)
        empty_grid.add_generator(generator2)
        assert len(list(empty_grid.generators())) == 2
        geo_data3 = Point(1, 0)
        generator3 = GeneratorFluctuatingDing0(id_db=2,
                                               weather_cell_id=0,
                                               geo_data=geo_data3)
        empty_grid.add_generator(generator3)
        assert len(list(empty_grid.generators())) == 3
        geo_data4 = Point(1, 1)
        generator4 = GeneratorFluctuatingDing0(id_db=3,
                                               weather_cell_id=1,
                                               geo_data=geo_data4)
        empty_grid.add_generator(generator4)
        assert len(list(empty_grid.generators())) == 4
        generator_list = list(empty_grid.generators())
        assert generator_list[0].id_db == 0
        assert generator_list[0].geo_data == geo_data1
        assert generator_list[1].id_db == 1
        assert generator_list[1].geo_data == geo_data2
        assert generator_list[2].id_db == 2
        assert generator_list[2].weather_cell_id == 0
        assert generator_list[2].geo_data == geo_data3
        assert generator_list[3].id_db == 3
        assert generator_list[3].weather_cell_id == 1
        assert generator_list[3].geo_data == geo_data4

    # check for all positive cases of graph_add_node
    def test_graph_add_node_station(self, empty_grid):
        station1 = StationDing0()
        empty_grid.graph_add_node(station1)
        assert station1 in empty_grid._graph.nodes()
        assert len(list(empty_grid._graph.nodes())) == 1

    def test_graph_add_node_cable_distributor(self, empty_grid):
        cable_distributor1 = CableDistributorDing0()
        empty_grid.graph_add_node(cable_distributor1)
        assert cable_distributor1 in empty_grid._graph.nodes()
        assert len(list(empty_grid._graph.nodes())) == 1

    def test_graph_add_node_lv_load_area_centre(self, empty_grid):
        lv_load_area_centre1 = LVLoadAreaCentreDing0()
        empty_grid.graph_add_node(lv_load_area_centre1)
        assert lv_load_area_centre1 in empty_grid._graph.nodes()
        assert len(list(empty_grid._graph.nodes())) == 1

    def test_graph_add_node_generator(self, empty_grid):
        # an add_node is called within add_generator
        geo_data1 = Point(0, 0)
        generator1 = GeneratorDing0(id_db=0,
                                    geo_data=geo_data1)
        empty_grid.graph_add_node(generator1)
        assert generator1 in empty_grid._graph.nodes()
        assert len(list(empty_grid._graph.nodes())) == 1

    def test_graph_add_node_add_generator(self, empty_grid):
        # an add_node is called within add_generator
        geo_data1 = Point(0, 0)
        generator1 = GeneratorDing0(id_db=0,
                                    geo_data=geo_data1)
        empty_grid.add_generator(generator1)
        assert generator1 in empty_grid._graph.nodes()
        # make sure that another call of add_nodes
        # does nothing
        len_nodes_before = len(list(empty_grid._graph.nodes()))
        empty_grid.graph_add_node(generator1)
        len_nodes_after = len(list(empty_grid._graph.nodes()))
        assert len_nodes_before == len_nodes_after

    def test_graph_add_node_generator_fluctuating(self, empty_grid):
        # an add_node is called within add_generator
        geo_data1 = Point(0, 0)
        generator1 = GeneratorFluctuatingDing0(id_db=0,
                                               geo_data=geo_data1)
        empty_grid.add_generator(generator1)
        assert generator1 in empty_grid._graph.nodes()
        # make sure that another call of add_nodes
        # does nothing
        len_nodes_before = len(list(empty_grid._graph.nodes()))
        empty_grid.graph_add_node(generator1)
        len_nodes_after = len(list(empty_grid._graph.nodes()))
        assert len_nodes_before == len_nodes_after

    # negative tests for graph_add_node
    def test_graph_add_node_load(self, empty_grid):
        load1 = LoadDing0(grid=empty_grid)
        empty_grid._loads = [load1]
        # make sure that call of add_nodes
        # does nothing
        len_nodes_before = len(list(empty_grid._graph.nodes()))
        empty_grid.graph_add_node(load1)
        len_nodes_after = len(list(empty_grid._graph.nodes()))
        assert len_nodes_before == len_nodes_after

    def test_graph_add_node_branch(self, empty_grid):
        branch1 = BranchDing0()
        # make sure that call of add_nodes
        # does nothing
        len_nodes_before = len(list(empty_grid._graph.nodes()))
        empty_grid.graph_add_node(branch1)
        len_nodes_after = len(list(empty_grid._graph.nodes()))
        assert len_nodes_before == len_nodes_after

    def test_graph_add_node_grid(self, empty_grid):
        grid1 = GridDing0()
        # make sure that call of add_nodes
        # does nothing
        len_nodes_before = len(list(empty_grid._graph.nodes()))
        empty_grid.graph_add_node(grid1)
        len_nodes_after = len(list(empty_grid._graph.nodes()))
        assert len_nodes_before == len_nodes_after

    @pytest.fixture
    def simple_graph_grid(self):
        grid = GridDing0(id_db=0)
        station = StationDing0(id_db=0, geo_data=Point(0, 0))
        generator = GeneratorDing0(id_db=0,
                                   geo_data=Point(0, 1),
                                   mv_grid=grid)
        grid.graph_add_node(station)
        grid.add_generator(generator)
        branch = BranchDing0(id_db=0,
                             length=2.0,
                             kind='cable')
        grid._graph.add_edge(generator, station, branch=branch)
        return (grid, station, generator, branch)

    def test_graph_nodes_from_branch(self, simple_graph_grid):
        grid, station, generator, branch = simple_graph_grid
        nodes_from_branch = grid.graph_nodes_from_branch(branch)
        assert type(nodes_from_branch) == tuple
        assert len(nodes_from_branch) == 2
        assert set(nodes_from_branch) == {station, generator}

    def test_graph_branches_from_node(self, simple_graph_grid):
        grid, station, generator, branch = simple_graph_grid
        branches_from_node = grid.graph_branches_from_node(station)
        assert branches_from_node == [(generator, dict(branch=branch))]

    def test_graph_edges(self, simple_graph_grid):
        grid, station, generator, branch = simple_graph_grid
        graph_edges = list(grid.graph_edges())
        assert type(graph_edges[0]) == dict
        assert len(list(graph_edges[0].keys())) == 2
        assert set(list(graph_edges[0].keys())) == {'adj_nodes', 'branch'}
        adj_nodes_out = graph_edges[0]['adj_nodes']
        assert type(adj_nodes_out) == tuple
        assert len(adj_nodes_out) == 2
        assert set(adj_nodes_out) == {station, generator}
        branch_out = graph_edges[0]['branch']
        assert branch_out == branch

    def test_find_path(self, simple_graph_grid):
        grid, station, generator, branch = simple_graph_grid
        path = grid.find_path(generator, station)
        assert path == [generator, station]

    def test_graph_path_length(self, simple_graph_grid):
        grid, station, generator, branch = simple_graph_grid
        path_length = grid.graph_path_length(generator, station)
        assert path_length == 2.0

    def test_graph_isolated_nodes(self, simple_graph_grid):
        grid, station, generator, branch = simple_graph_grid
        isolates = grid.graph_isolated_nodes()
        assert isolates == []

    def test_grid_stats(self, oedb_session):
        '''
        Using grid district 460 to check if statistical data stay the same
        :param oedb_session:
        :return:
        '''
        # instantiate new ding0 network object
        path = os.path.dirname(os.path.abspath(__file__))

        nd = NetworkDing0(name='network')

        mv_grid_districts = [460]

        nd.run_ding0(session=oedb_session,
             mv_grid_districts_no=mv_grid_districts)

        # check mv grid statistics
        mvgd_stats = calculate_mvgd_stats(nd)
        mvgd_stats_comparison = pd.DataFrame.from_csv(os.path.join(path, 'testdata/mvgd_stats.csv'))
        assert_frame_equal(mvgd_stats, mvgd_stats_comparison,check_dtype=False)

        # check mv grid statistics voltages and currents
        mvgd_voltage_current_stats = calculate_mvgd_voltage_current_stats(nd)
        mvgd_current_branches = mvgd_voltage_current_stats[1]
        mvgd_voltage_nodes = mvgd_voltage_current_stats[0]
        mvgd_current_branches_comparison = pd.DataFrame.from_csv(
            os.path.join(path, 'testdata/mvgd_current_branches.csv'))
        mvgd_current_branches_comparison = mvgd_current_branches_comparison.replace(np.NaN, 'NA')
        mvgd_voltage_nodes_comparison = pd.DataFrame.from_csv(
            os.path.join(path, 'testdata/mvgd_voltage_nodes.csv'))
        mvgd_voltage_nodes_comparison = mvgd_voltage_nodes_comparison.replace(np.NaN, 'NA')
        assert_frame_equal(mvgd_current_branches, mvgd_current_branches_comparison,check_dtype=False)
        assert_frame_equal(mvgd_voltage_nodes, mvgd_voltage_nodes_comparison,check_dtype=False)

        # check lv grid statistics
        lvgd_stats = calculate_lvgd_stats(nd)
        lvgd_stats_comparison = pd.DataFrame.from_csv(os.path.join(path,'testdata/lvgd_stats.csv'))
        assert_frame_equal(lvgd_stats, lvgd_stats_comparison,check_dtype=False)

        # check lv grid statistics voltages and currents
        lvgd_voltage_current_stats = calculate_lvgd_voltage_current_stats(nd)
        lvgd_current_branches = lvgd_voltage_current_stats[1]
        lvgd_voltage_nodes = lvgd_voltage_current_stats[0]
        lvgd_current_branches_comparison = pd.DataFrame.from_csv(
            os.path.join(path, 'testdata/lvgd_current_branches.csv'))
        lvgd_current_branches_comparison = lvgd_current_branches_comparison.replace(np.NaN, 'NA')
        lvgd_voltage_nodes_comparison = pd.DataFrame.from_csv(
            os.path.join(path, 'testdata/lvgd_voltage_nodes.csv'))
        lvgd_voltage_nodes_comparison = lvgd_voltage_nodes_comparison.replace(np.NaN, 'NA')
        assert_frame_equal(lvgd_current_branches, lvgd_current_branches_comparison,check_dtype=False)
        assert_frame_equal(lvgd_voltage_nodes, lvgd_voltage_nodes_comparison,check_dtype=False)




class TestStationDing0(object):

    @pytest.fixture
    def empty_stationding0(self):
        """
        Returns an empty StationDing0 object
        """
        return StationDing0()

    @pytest.fixture
    def test_empty_stationding0(self, empty_stationding0):
        assert empty_stationding0.id_db is None
        assert empty_stationding0.geo_data is None
        assert empty_stationding0.grid is None
        assert empty_stationding0.v_level_operation is None
        assert list(empty_stationding0.transformers()) == []

    def test_add_transformer(self, empty_stationding0):
        transformer1 = TransformerDing0(id_db=0,
                                        v_level=4,
                                        s_max_longterm=400.0,
                                        s_max_shortterm=600.0,
                                        s_max_emergency=800.0,
                                        phase_angle=0.0,
                                        tap_ratio=1.02,
                                        r=0.02,
                                        x=0.002)
        transformer2 = TransformerDing0(id_db=1,
                                        v_level=4,
                                        s_max_longterm=600.0,
                                        s_max_shortterm=900.0,
                                        s_max_emergency=1100.0,
                                        phase_angle=0.01,
                                        tap_ratio=1.00,
                                        r=0.01,
                                        x=0.001)
        empty_stationding0.add_transformer(transformer1)  # added 1
        empty_stationding0.add_transformer(transformer2)  # added 2
        transformer_list = list(empty_stationding0.transformers())
        assert len(transformer_list) == 2
        transformer1_in_empty_stationding0 = transformer_list[0]
        assert transformer1_in_empty_stationding0 == transformer1
        assert transformer1_in_empty_stationding0.id_db == 0
        assert transformer1_in_empty_stationding0.v_level == 4
        assert transformer1_in_empty_stationding0.s_max_a == 400.0
        assert transformer1_in_empty_stationding0.s_max_b == 600.0
        assert transformer1_in_empty_stationding0.s_max_c == 800.0
        assert transformer1_in_empty_stationding0.phase_angle == 0.0
        assert transformer1_in_empty_stationding0.tap_ratio == 1.02
        assert transformer1_in_empty_stationding0.r == 0.02
        assert transformer1_in_empty_stationding0.x == 0.002
        transformer2_in_empty_stationding0 = transformer_list[1]
        assert transformer2_in_empty_stationding0 == transformer2
        assert transformer2_in_empty_stationding0.id_db == 1
        assert transformer2_in_empty_stationding0.v_level == 4
        assert transformer2_in_empty_stationding0.s_max_a == 600.0
        assert transformer2_in_empty_stationding0.s_max_b == 900.0
        assert transformer2_in_empty_stationding0.s_max_c == 1100.0
        assert transformer2_in_empty_stationding0.phase_angle == 0.01
        assert transformer2_in_empty_stationding0.tap_ratio == 1.00
        assert transformer2_in_empty_stationding0.r == 0.01
        assert transformer2_in_empty_stationding0.x == 0.001


if __name__ == "__main__":
    pass
