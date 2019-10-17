import pytest

from egoio.tools import db
from sqlalchemy.orm import sessionmaker
import shutil
import os
import pandas as pd

from ding0.core import NetworkDing0
from tests.core.network.test_grids import TestMVGridDing0


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
        engine = db.connection(readonly=True)
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

    @pytest.fixture
    def minimal_grid(self):
        nd, mv_grid, lv_stations = \
            TestMVGridDing0().minimal_unrouted_testgrid()
        nd.mv_routing()
        nd.connect_generators()
        nd.set_branch_ids()
        return nd

    def test_run_powerflow(self, minimal_grid):
        """
        Checks if power flow on test grid provides the expected values.
        Both power flow on only mv grid and on combined mv and lv grids
        are tested.
        """
        try:
            def extract_tuple_values_from_string(string):
                tuple = string.replace('[', '')
                tuple = tuple.replace(']', '')
                tuple = str.split(tuple, ',')
                return float(tuple[0]), float(tuple[1])

            def assert_by_absolute_tolerance(x, y, tol=0.0001,
                                             element_name = ''):
                if (x - y > tol):
                    raise Exception('Unequal values for element {}: val1 is {}'
                                    ', val2 is {}.'.format(element_name, x, y))

            def assert_almost_equal(orig_df, comp_df,
                                    element_name, tol=1e-4):
                for key in orig_df.index:
                    load_ref, gen_ref = \
                        extract_tuple_values_from_string(orig_df[key])
                    load_comp, gen_comp = \
                        extract_tuple_values_from_string(
                            comp_df.loc[element_name][key])
                    assert_by_absolute_tolerance(load_ref, load_comp,
                                                 tol, element_name)
                    assert_by_absolute_tolerance(gen_ref, gen_comp,
                                                 tol, element_name)

            # export to pypsa csv format
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            #load network
            nd = minimal_grid
            # save network and components to csv
            path = os.path.join(cur_dir,
                                'test'+ str(nd._mv_grid_districts[0].id_db))
            if not os.path.exists(path):
                os.makedirs(path)

            print('Starting power flow tests for MV grid only')
            nd.run_powerflow(None, export_result_dir=path)
            lines = pd.DataFrame.from_csv(os.path.join(path,'line_data.csv'))
            buses = pd.DataFrame.from_csv(os.path.join(path, 'bus_data.csv'))
            compare_lines = pd.DataFrame.from_csv(
                os.path.join(cur_dir,'testdata','line_data.csv'))
            compare_buses = pd.DataFrame.from_csv(
                os.path.join(cur_dir,'testdata','bus_data.csv'))
            # compare results
            for line_name, line_data in compare_lines.iterrows():
                assert_almost_equal(line_data, lines, line_name)
            for bus_name, bus_data in compare_buses.iterrows():
                assert_almost_equal(bus_data, buses, bus_name)
            print('Finished testing MV grid only')

        finally:
            shutil.rmtree(path, ignore_errors=True)