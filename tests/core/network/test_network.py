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
