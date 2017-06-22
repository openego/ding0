import unittest2 as unittest
from test import support

# import DB interface from oemof
import oemof.db as db
# import other dingo stuff
from dingo.core import NetworkDingo
from dingo.tools.logger import setup_logger
from dingo.tools.results import save_nd_to_pickle
from dingo.tools.results import load_nd_from_pickle

from geoalchemy2.shape import to_shape
import logging


logger = setup_logger(loglevel=logging.CRITICAL)

class DingoRunTest(unittest.TestCase):

    def setUp(self):
        print('\n')

    #def tearDown(self):
    #    print("cleanup")

    def test_files(self):
        print('Test File vs File')
        print('  Loading data...')
        nw_1 = load_nd_from_pickle(filename='dingo_tests_grids_1.pkl')
        nw_2 = load_nd_from_pickle(filename='dingo_tests_grids_2.pkl')
        #test equality
        print('  Testing equality...')
        equals_e, msg = self.dataframe_equal(nw_1,nw_1)
        print('  ...'+msg)
        #test difference
        print('  Testing difference...')
        equals_d, msg = self.dataframe_equal(nw_1,nw_2)
        print('  ...'+msg)

        #compare results
        if equals_e and not equals_d:
            msg    = 'No failure'
            passed = True
        elif equals_e and equals_d:
            msg    = 'Only difference failed'
            passed = False
        elif not equals_e and not equals_d:
            msg    = 'Only equality failed'
            passed = False
        elif not equals_e and equals_d:
            msg    = 'Both failed'
            passed = False
        print('    '+msg)
        self.assertTrue(passed,msg=msg)

    #def test_dingo_file(self):
    #    print('Test dingo vs File')
    #    print('  Loading data...')
    #    nw_1 = load_nd_from_pickle(filename='dingo_tests_grids_1.pkl')

    #    print('  Running dingo for the same configuration...')
    #    # database connection
    #    conn = db.connection(section='oedb')
    #    mv_grid_districts = [3545]

    #    nw_2 = NetworkDingo(name='network')
    #    nw_2.run_dingo(conn=conn, mv_grid_districts_no=mv_grid_districts)
    #    nw_2.control_circuit_breakers(mode='close')
    #    nw_2.export_mv_grid(conn, mv_grid_districts)
    #    nw_2.export_mv_grid_new(conn, mv_grid_districts)

    #    conn.close()

    #    #test equality
    #    print('  Testing equality...')
    #    passed, msg = self.dataframe_equal(nw_1,nw_2)
    #    print('    ...'+msg)

    #    self.assertTrue(passed,msg=msg)

    def test_dingo(self):
        print('Test dingo vs dingo')
        conn = db.connection(section='oedb')
        mv_grid_districts = [3545]

        print('  Running dingo once...')
        nw_1 = NetworkDingo(name='network')
        nw_1.run_dingo(conn=conn, mv_grid_districts_no=mv_grid_districts)
        #nw_1.control_circuit_breakers(mode='close')
        #nw_1.export_mv_grid(conn, mv_grid_districts)
        #nw_1.export_mv_grid_new(conn, mv_grid_districts)

        print('  Running dingo twice...')
        nw_2 = NetworkDingo(name='network')
        nw_2.run_dingo(conn=conn, mv_grid_districts_no=mv_grid_districts)
        #nw_2.control_circuit_breakers(mode='close')
        #nw_2.export_mv_grid(conn, mv_grid_districts)
        #nw_2.export_mv_grid_new(conn, mv_grid_districts)

        conn.close()

        #test equality
        print('  Testing equality...')
        passed, msg = self.dataframe_equal(nw_1,nw_2)
        print('    ...'+msg)

        self.assertTrue(passed,msg=msg)

    def dataframe_equal(self, network_one, network_two):
        #initiate dataframes through to_dataframe method
        nodes_one_df, edges_one_df = network_one.to_dataframe()
        nodes_two_df, edges_two_df = network_two.to_dataframe()

        #print(nodes_one_df['node_id'].size)
        #print(nodes_two_df['node_id'].size)
        #First, check if sizes of both are the same
        if nodes_one_df['node_id'].size!=nodes_two_df['node_id'].size:
            msg = 'Different number of nodes'
            return False, msg
        elif edges_one_df['branch_id'].size!=edges_two_df['branch_id'].size:
            msg = 'Different number of edges'
            return False, msg

        #Second, check if IDs of both are the same
        if not nodes_one_df['node_id'].equals(nodes_two_df['node_id']):
            msg = 'Same number of nodes with different IDs'
            return False, msg
        elif not edges_one_df['branch_id'].equals(edges_two_df['branch_id']):
            msg = 'Same number of branches with different IDs'
            return False, msg

        #workaround because 'geom' is strange
        #    shapy geo information is in format WKB, which also includes somehow
        #    the memory slot where the data is stored. So we need to rescue just
        #    the data.
        #For the Nodes:
        nodes_one_df['geom'] = nodes_one_df['geom'].apply(lambda x: x.desc)
        nodes_two_df['geom'] = nodes_two_df['geom'].apply(lambda x: x.desc)

        #Similar for edges, but the extreme nodes of an edge can be switched
        #    first, convert to shape and resque coordinates
        edges_one_geom_xy    = edges_one_df['geom'].apply(lambda x: to_shape(x)).apply(lambda x: x.xy)
        edges_two_geom_xy    = edges_two_df['geom'].apply(lambda x: to_shape(x)).apply(lambda x: x.xy)
        #    rescue data in the dataframe
        edges_one_df['geom'] = edges_one_df['geom'].apply(lambda x: x.desc)
        edges_two_df['geom'] = edges_two_df['geom'].apply(lambda x: x.desc)
        #    then go through the edges and compare crossed coordinates
        for idx, row in edges_one_geom_xy.iteritems():
            if (    (edges_one_geom_xy[idx][0][0] == edges_two_geom_xy[idx][0][1])
                and (edges_one_geom_xy[idx][0][1] == edges_two_geom_xy[idx][0][0])
                and (edges_one_geom_xy[idx][1][0] == edges_two_geom_xy[idx][1][1])
                and (edges_one_geom_xy[idx][1][1] == edges_two_geom_xy[idx][1][0])
            ):
                #if nodes are inverted, force both to be equal
                edges_two_df.loc[idx,'geom'] = edges_one_df.loc[idx,'geom']


        # compare things
        flag_nodes = nodes_one_df.equals(nodes_two_df)
        flag_edges = edges_one_df.equals(edges_two_df)
        passed     = flag_nodes and flag_edges

        #return result of test
        msg = 'Data sets are '
        if passed:
            msg = msg + 'identical.'
        else:
            msg = msg + 'different.'
        return passed, msg

def init_files_for_tests():
    # database connection
    conn = db.connection(section='oedb')

    # instantiate new dingo network object
    nd = NetworkDingo(name='network')

    # choose MV Grid Districts to import
    mv_grid_districts = [3545]

    # run DINGO on selected MV Grid District
    nd.run_dingo(conn=conn,mv_grid_districts_no=mv_grid_districts)

    # export grids to database
    nd.control_circuit_breakers(mode='close')
    nd.export_mv_grid(conn, mv_grid_districts)
    nd.export_mv_grid_new(conn, mv_grid_districts)

    # export grid to file (pickle)
    save_nd_to_pickle(nd, filename='dingo_tests_grids_1.pkl')

    # instantiate new dingo network object
    nd = NetworkDingo(name='network')

    # choose MV Grid Districts to import
    mv_grid_districts = [428]

    # run DINGO on selected MV Grid District
    nd.run_dingo(conn=conn,mv_grid_districts_no=mv_grid_districts)

    # export grid to file (pickle)
    save_nd_to_pickle(nd, filename='dingo_tests_grids_2.pkl')

    conn.close()

if __name__ == "__main__":
    support.run_unittest(DingoRunTest)
    #init_files_for_tests()
