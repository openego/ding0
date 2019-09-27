import pytest

from egoio.tools import db
from sqlalchemy.orm import sessionmaker
import oedialect
import pandas as pd
import os

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

    def test_to_csv(self, oedb_session):
        #Todo: docstring and change grid to testgrid
        #instantiate new ding0 network object
        nd = NetworkDing0(name='network')

        # choose MV Grid Districts to import
        mv_grid_districts = [460]

        # run DING0 on selected MV Grid District
        nd.run_ding0(session=oedb_session,mv_grid_districts_no=mv_grid_districts)

        # export to pypsa csv format
        dir = os.getcwd()
        nd.to_csv(dir)

        # import exported dataset
        buses = pd.DataFrame.from_csv(os.path.join(dir,'460','buses_460.csv'))
        lines = pd.DataFrame.from_csv(os.path.join(dir,'460', 'lines_460.csv'))
        loads = pd.DataFrame.from_csv(os.path.join(dir,'460', 'loads_460.csv'))
        generators = pd.DataFrame.from_csv(os.path.join(dir,'460', 'generators_460.csv'))
        transformers = pd.DataFrame.from_csv(os.path.join(dir,'460', 'transformers_460.csv'))

        try:
            # check if buses of lines exist in buses
            # Todo: get name of line
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
            # Todo: get name of transformer
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
            os.remove(os.path.join(dir,'460','buses_460.csv'))
            os.remove(os.path.join(dir,'460','lines_460.csv'))
            os.remove(os.path.join(dir,'460','loads_460.csv'))
            os.remove(os.path.join(dir,'460', 'generators_460.csv'))
            os.remove(os.path.join(dir, '460','transformers_460.csv'))


    # def test_run_ding0(self):
    #     pass
