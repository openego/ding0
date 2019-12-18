import pytest

from egoio.tools import db
from sqlalchemy.orm import sessionmaker
import oedialect

from ding0.core import NetworkDing0


class TestNetworkDing0(object):

    @pytest.fixture
    def emptyNetworkDing0(self):
        """
        Returns an empty NetworkDing0 object for testing
        """
        return NetworkDing0()

    def test_empty_mv_grid_districts(self, emptyNetworkDing0):
        mv_grid_districts = list(emptyNetworkDing0.mv_grid_districts())
        empty_list = []
        assert mv_grid_districts == empty_list

    def test_import_mv_grid_districts(self, oedb_session):
        with pytest.raises(TypeError):
            NetworkDing0.import_mv_grid_districts(
                oedb_session,
                mv_grid_districts_no=['5']
            )

    # def test_run_ding0(self):
    #     pass
