import pytest
from ding0.core.network import GridDing0, \
    GeneratorDing0, GeneratorFluctuatingDing0


class TestMVStationDing0(object):

    @pytest.fixture
    def empty_gridding0(self):
        """
        Returns an empty GridDing0 object
        """
        return GridDing0()

    def test_mv_grid_districts(self):
        pass

    def test_run_ding0(self):
        pass


if __name__ == "__main__":
    pass
