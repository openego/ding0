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


class TestGridDing0(object):

    @pytest.fixture
    def empty_grid(self):
        """
        Returns and empty GridDing0 object
        """
        return GridDing0()

    @pytest.fixture
    def simple_grid(self):
        """
        Returns a basic GridDing0 object
        """
        network = NetworkDing0(name='TestNetwork',
                               run_id='test_run')
        grid_district = Polygon(((0, 0),
                                 (0, 1),
                                 (1, 1),
                                 (1, 0),
                                 (0, 0)))
        grid = GridDing0(network=network,
                         id_db=0,
                         grid_district=grid_district)
        return grid

    # There is no setter function for providing a list
    # of cable_distributors to the empty_grid
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
        cable_distributor_grid._cable_distributors = [cable_distributor1,
                                                      cable_distributor2]
        return cable_distributor_grid

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

    # There is no setter function for providing a list
    # of loads to the empty_grid
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
        load_grid._loads = [load1]
        geo_data2 = Point(0, 1)
        load2 = LoadDing0(id_db=0,
                          geo_data=geo_data2,
                          grid=load_grid)
        load_grid._loads.append(load2)
        return load_grid
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


class TestRingDing0(object):

    @pytest.fixture
    def empty_ringding0(self):
        """
        Returns an empty RingDing0 object
        """
        return RingDing0()

    @pytest.fixture
    def simple_ringding0(self):
        """
        Returns a simple RingDing0 object
        """
        ringding0 = RingDing0(grid=grid,
                              id_db=0)


class TestLoadDing0(object):

    @pytest.fixture
    def empty_loadding0(self):
        """
        Returns an empty LoadDing0 object
        """
        return LoadDing0(grid=GridDing0())

    @pytest.fixture
    def some_loadding0(self):
        """
        Returns a networkless LoadDing0 object with some set parameters
        """
        geo_data = Point(0, 0)
        network = NetworkDing0(name='TestNetwork',
                               run_id='test_run')

        grid_district = Polygon([Point(0, 0),
                                 Point(0, 1),
                                 Point(1, 1),
                                 Point(1, 0),
                                 Point(0, 0)])
        grid = GridDing0(network=network,
                         id_db=0,
                         grid_district=grid_district)
        load = LoadDing0(id_db=0,
                         geo_data=geo_data,
                         grid=grid,
                         peak_load=dict(residential=1.0,
                                        retail=1.0,
                                        industrial=1.0,
                                        agricultural=1.0),
                         consumption=dict(residential=1.0,
                                          retail=1.0,
                                          industrial=1.0,
                                          agricultural=1.0))

        return load

    def test_empty_loadding0(self, empty_loadding0):
        assert empty_loadding0.id_db is 1
        assert empty_loadding0.geo_data is None
        # Once binary equality operators are implemented
        # the following can be tested
        # assert empty_loadding0.grid == GridDing0()
        assert empty_loadding0.peak_load is None
        assert empty_loadding0.consumption is None

    # def test_some_loadding0(self, some_loadding0):
    #     assert some_loadding0.id_db is None
    #     assert some_loadding0.geo_data is None
    #     assert some_loadding0.grid is None
    #     assert some_loadding0.peak_load is None
    #     assert some_loadding0.consumption is None


if __name__ == "__main__":
    pass
