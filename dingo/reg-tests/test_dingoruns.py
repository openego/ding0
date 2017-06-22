import unittest2 as unittest
from test import support

# import DB interface from oemof
import oemof.db as db
# import other dingo stuff
from dingo.core import NetworkDingo
from dingo.tools import results
from dingo.tools.logger import setup_logger
from dingo.tools.results import save_nd_to_pickle

from geoalchemy2.shape import to_shape


logger = setup_logger()

class DingoRunTest(unittest.TestCase):

    def setUp(self):
        print("setup")
        self.file1 = results.load_nd_from_pickle(filename='dingo_tests_grids_1.pkl')
        self.file2 = results.load_nd_from_pickle(filename='dingo_tests_grids_2.pkl')

    def tearDown(self):
        print("cleanup")

    def test_equal(self):
        equals, msg = self.dingo_equal(self.file1,self.file1)
        self.assertTrue(equals,msg=msg)
    def test_different(self):
        equals, msg = self.dingo_equal(self.file1,self.file2)
        self.assertFalse(equals,msg=msg)

    def dingo_equal(self, file_1, file_2):
        #initiate dataframes through to_dataframe method
        nodes_one_df, edges_one_df = file_1.to_dataframe()
        nodes_two_df, edges_two_df = file_2.to_dataframe()
        #future: take the to_dataframe functions out of this function
        #First, check if sizes of both are the same
        if nodes_one_df.size!=nodes_two_df.size:
            msg = 'Different number of nodes!'
            return False, msg
        elif edges_one_df.size!=edges_two_df.size:
            msg = 'Different number of edges!'
            return False, msg

        #Second, check if IDs of both are the same
        if not nodes_one_df['node_id'].equals(nodes_two_df['node_id']):
            msg = 'Same number of nodes with different IDs!'
            return False, msg
        elif not edges_one_df['branch_id'].equals(edges_two_df['branch_id']):
            msg = 'Same number of branches with different IDs!'
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
        msg = 'Data sets are'
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
