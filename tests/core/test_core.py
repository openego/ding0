import pytest

from egoio.tools import db
from sqlalchemy.orm import sessionmaker
import oedialect
import pandas as pd
import os
from tests.core.network.test_grids import TestMVGridDing0
from pandas.util.testing import assert_frame_equal, assert_series_equal
from ding0.tools.results import load_nd_from_pickle
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
        switches = pd.DataFrame.from_csv(os.path.join(dir,id_mvgd, 'switches_{}.csv'.format(id_mvgd)))

        try:
            # check if all entries are unique
            if not lines.index.is_unique:
                raise Exception('Line names are not unique. Please check.')
            if not transformers.index.is_unique:
                raise Exception('Transformer names are not unique. Please check.')
            if not loads.index.is_unique:
                raise Exception('Load names are not unique. Please check.')
            if not generators.index.is_unique:
                raise Exception('Generator names are not unique. Please check.')
            if not buses.index.is_unique:
                raise Exception('Bus names are not unique. Please check.')
            if not switches.index.is_unique:
                raise Exception('Switch names are not unique. Please check.')
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

            # check if buses of switches exist in buses
            for bus in switches['bus_open']:
                if bus in buses.T:
                    continue
                else:
                    raise Exception('bus {} of open switches not in buses dataframe.'.format(bus))
            for bus in switches['bus_closed']:
                if bus in buses.T:
                    continue
                else:
                    raise Exception('bus {} of closed switches not in buses dataframe.'.format(bus))

            # check if all buses are connected to either line or transformer
            for bus in buses.T:
                if not bus in lines['bus0'].values and not bus in lines['bus1'].values and \
                    not bus in transformers['bus0'].values and not bus in transformers['bus1'].values\
                        and bus not in switches['bus_open'].values:
                    raise Exception('Bus {} is not connected to any branch.'.format(bus))
        finally:
            os.remove(os.path.join(dir,id_mvgd,'buses_{}.csv'.format(id_mvgd)))
            os.remove(os.path.join(dir,id_mvgd,'lines_{}.csv'.format(id_mvgd)))
            os.remove(os.path.join(dir,id_mvgd,'loads_{}.csv'.format(id_mvgd)))
            os.remove(os.path.join(dir,id_mvgd, 'generators_{}.csv'.format(id_mvgd)))
            os.remove(os.path.join(dir,id_mvgd,'transformers_{}.csv'.format(id_mvgd)))
            os.remove(os.path.join(dir, id_mvgd, 'network_{}.csv'.format(id_mvgd)))
            os.remove(os.path.join(dir, id_mvgd, 'switches_{}.csv'.format(id_mvgd)))
            os.rmdir(os.path.join(dir, id_mvgd))


    def debug_run_powerflow(self):
        try:
            def extract_tuple_values_from_string(string):
                tuple = string.replace('[', '')
                tuple = tuple.replace(']', '')
                tuple = str.split(tuple, ',')
                return float(tuple[0]), float(tuple[1])

            def assert_by_absolute_tolerance(x, y, tol=0.0001, element_name = ''):
                if (x - y > tol):
                    raise Exception('Unequal values for element {}: val1 is {}, val2 is {}.'.format(element_name, x, y))

            def assert_almost_equal(orig_df, comp_df, element_name, tol = 1e-4):
                for key in orig_df.index:
                    load_ref, gen_ref = extract_tuple_values_from_string(orig_df[key])
                    load_comp, gen_comp = extract_tuple_values_from_string(comp_df.loc[element_name][key])
                    assert_by_absolute_tolerance(load_ref, load_comp, tol, element_name)
                    assert_by_absolute_tolerance(gen_ref, gen_comp, tol, element_name)

            # export to pypsa csv format
            dir = os.getcwd()
            #load network, Todo: change to run ding0
            nd = load_nd_from_pickle(filename='ding0_grids_example.pkl',path= 'C:/Users/Anya.Heider/open_BEA/ding0/ding0/examples')
            # save network and components to csv
            path = os.path.join(dir, str(nd._mv_grid_districts[0].id_db))
            if not os.path.exists(path):
                os.makedirs(path)
            nd.run_powerflow(export_result_dir=path)
            lines = pd.DataFrame.from_csv(os.path.join(path,'line_data.csv'))
            buses = pd.DataFrame.from_csv(os.path.join(path, 'bus_data.csv'))
            compare_lines = pd.DataFrame.from_csv('C:/Users/Anya.Heider/.ding0/pf_results_before/line_data.csv') #Todo: move to project directory
            compare_buses = pd.DataFrame.from_csv('C:/Users/Anya.Heider/.ding0/pf_results_before/bus_data.csv')
            #compare results
            for line_name, line_data in compare_lines.iterrows():
                assert_almost_equal(line_data, lines, line_name)
            for bus_name, bus_data in compare_buses.iterrows():
                assert_almost_equal(bus_data, buses, bus_name)

        finally:
            os.remove(os.path.join(path,'line_data.csv'))
            os.remove(os.path.join(path, 'bus_data.csv'))
            # os.rmdir(os.path.join(path))
    # def test_run_ding0(self):
    #     pass
