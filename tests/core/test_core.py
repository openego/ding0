import pytest

from egoio.tools import db
from sqlalchemy.orm import sessionmaker
import oedialect
import pandas as pd
import os
from tests.core.network.test_grids import TestMVGridDing0

from ding0.core import NetworkDing0


class TestNetworkDing0(object):

    @pytest.fixture
    def emptyNetworkDing0(self):
        """
        Returns an empty NetworkDing0 object for testing
        """
        return NetworkDing0()

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
    def minimal_grid(self):
        nd, mv_grid, lv_stations = TestMVGridDing0().minimal_unrouted_testgrid()
        nd.mv_routing()
        nd.connect_generators()
        #nd.control_circuit_breakers(mode='open')
        return nd

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

    def test_to_csv(self, minimal_grid):
        """
        Check if export to csv exports network in a way that all exported nodes and elements are connected to each other.
        Checks if lines, transformers, laods and generators can be connected with buses.
        Checks if all exported buses are connected to at least one branch element (lines, transformers).
        """
        nd = minimal_grid

        # export to pypsa csv format
        dir = os.getcwd()
        nd.to_csv(dir)

        # import exported dataset
        id_mvgd = str(nd._mv_grid_districts[0].mv_grid.grid_district.id_db)
        buses = pd.DataFrame.from_csv(os.path.join(dir,id_mvgd,'buses_{}.csv'.format(id_mvgd)))
        lines = pd.DataFrame.from_csv(os.path.join(dir,id_mvgd, 'lines_{}.csv'.format(id_mvgd)))
        loads = pd.DataFrame.from_csv(os.path.join(dir,id_mvgd, 'loads_{}.csv'.format(id_mvgd)))
        generators = pd.DataFrame.from_csv(os.path.join(dir,id_mvgd, 'generators_{}.csv'.format(id_mvgd)))
        transformers = pd.DataFrame.from_csv(os.path.join(dir,id_mvgd, 'transformers_{}.csv'.format(id_mvgd)))

        try:
            # check if buses of lines exist in buses
            for bus in lines['bus0']:
                if bus in buses.T:
                    continue
                else:
                    raise Exception('bus0 {} of line not in buses dataframe.'.format(bus))
            for bus in lines['bus1']:
                if bus in buses.T:
                    continue
                else:
                    raise Exception('bus1 {} of line not in buses dataframe.'.format(bus))

            # check if buses of transformers exist in buses
            for bus in transformers['bus0']:
                if bus in buses.T:
                    continue
                else:
                    raise Exception('bus0 {} of transformer not in buses dataframe.'.format(bus))
            for bus in transformers['bus1']:
                if bus in buses.T:
                    continue
                else:
                    raise Exception('bus1 {} of transformer not in buses dataframe.'.format(bus))

            # check if buses of loads exist in buses
            for bus in loads['bus']:
                if bus in buses.T:
                    continue
                else:
                    raise Exception('bus {} of load not in buses dataframe.'.format(bus))

            # check if buses of generators exist in buses
            for bus in generators['bus']:
                if bus in buses.T:
                    continue
                else:
                    raise Exception('bus {} of generator not in buses dataframe.'.format(bus))

            # check if all buses are connected to either line or transformer
            for bus in buses.T:
                if not bus in lines['bus0'].values and not bus in lines['bus1'].values and \
                    not bus in transformers['bus0'].values and not bus in transformers['bus1'].values:
                    raise Exception('Bus {} is not connected to any branch.'.format(bus))
        finally:
            os.remove(os.path.join(dir,id_mvgd,'buses_{}.csv'.format(id_mvgd)))
            os.remove(os.path.join(dir,id_mvgd,'lines_{}.csv'.format(id_mvgd)))
            os.remove(os.path.join(dir,id_mvgd,'loads_{}.csv'.format(id_mvgd)))
            os.remove(os.path.join(dir,id_mvgd, 'generators_{}.csv'.format(id_mvgd)))
            os.remove(os.path.join(dir,id_mvgd,'transformers_{}.csv'.format(id_mvgd)))
            os.remove(os.path.join(dir, id_mvgd, 'network_{}.csv'.format(id_mvgd)))
            os.rmdir(os.path.join(dir, id_mvgd))


    # def test_run_ding0(self):
    #     pass
