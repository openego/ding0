import unittest2 as unittest
from test import support

# import DB interface from oemof
import oemof.db as db

# import other dingo stuff
from dingo.core import NetworkDingo
from dingo.tools import results
from dingo.tools.logger import setup_logger
from dingo.tools.results import save_nd_to_pickle

#from numpy import array_equal
import numpy as np

logger = setup_logger()

class DingoRunTest(unittest.TestCase):

    def setUp(self):
        print("setup")
        filename = 'dingo_grids_example.pkl'
        mv_grid_districts = [3545]
        # database connection
        self.conn = db.connection(section='oedb')

        # load data from file
        self.nd_sav = results.load_nd_from_pickle(filename=filename)


        # Fresh run from dingo
        # instantiate new dingo network object
        #self.nd_sav = NetworkDingo(name='network')

        # run DINGO on selected MV Grid District
        #self.nd_sav.run_dingo(conn=self.conn,
        #                      mv_grid_districts_no=mv_grid_districts)

        #self.nd_sav.control_circuit_breakers(mode='close')
        #self.nd_sav.export_mv_grid(self.conn, mv_grid_districts)
        #self.nd_sav.export_mv_grid_new(self.conn, mv_grid_districts)


        # New run from dingo
        # instantiate new dingo network object
        #self.nd_new = NetworkDingo(name='network')

        # run DINGO on selected MV Grid District
        #self.nd_new.run_dingo(conn=self.conn,
        #                      mv_grid_districts_no=mv_grid_districts)

        #self.nd_new.control_circuit_breakers(mode='close')
        #self.nd_new.export_mv_grid(self.conn, mv_grid_districts)
        #self.nd_new.export_mv_grid_new(self.conn, mv_grid_districts)

        self.nd_new = results.load_nd_from_pickle(filename=filename)

        #self.nd_new = self.nd_sav

    def tearDown(self):
        print("cleanup")

    def test_dingo_todataframe(self):

        #initiate dataframes through to_dataframe method
        nodes_sav_df, edges_sav_df = self.nd_sav.to_dataframe()
        nodes_new_df, edges_new_df = self.nd_new.to_dataframe()

        #workaround because 'geom' is strange
        #    shapy geo information is in format WKB, which also includes somehow
        #    the memory slot where the data is stored. So we need to rescue just
        #    the data.
        nodes_sav_df['geom'] = nodes_sav_df['geom'].apply(lambda x: x.desc)
        nodes_new_df['geom'] = nodes_new_df['geom'].apply(lambda x: x.desc)

        edges_sav_df['geom'] = edges_sav_df['geom'].apply(lambda x: x.desc)
        edges_new_df['geom'] = edges_new_df['geom'].apply(lambda x: x.desc)

        #Prepare dataframe / workaround
        #    The dataframes after to_dataframe do not necessarily have the same
        #    index. That is, the element with 'node_id'='ID_1' with index = 0
        #    in nodes_sav_df, does not necessarily have index = 0 in nodes_new_df
        nodes_sav_df.sort_values('node_id', axis = 0, kind= 'mergesort', inplace=True)
        nodes_new_df.sort_values('node_id', axis = 0, kind= 'mergesort', inplace=True)
        nodes_sav_df = nodes_sav_df.reset_index(drop=True)
        nodes_new_df = nodes_new_df.reset_index(drop=True)

        edges_sav_df.sort_values('branch_id', axis = 0, kind= 'mergesort', inplace=True)
        edges_new_df.sort_values('branch_id', axis = 0, kind= 'mergesort', inplace=True)
        edges_sav_df = edges_sav_df.reset_index(drop=True)
        edges_new_df = edges_new_df.reset_index(drop=True)

        # compare things
        flag_nodes = nodes_sav_df.equals(nodes_new_df)
        flag_edges = edges_sav_df.equals(edges_new_df)
        passed     = flag_nodes and flag_edges

        #do stuff in fail case
        if not passed:
            print('did not work')
            edges_new_matrix = edges_new_df.as_matrix();
            edges_sav_matrix = edges_sav_df.as_matrix();
            #print(np.array_equal(edges_new_matrix, edges_sav_matrix))

            permiso = True
            for i in range(0, edges_new_matrix.size - 1):
                if not np.array_equal(edges_new_matrix.item(i),
                                      edges_sav_matrix.item(i)):
                    #print(i, edges_new_matrix.item(i), 'NEQ',
                    #      edges_sav_matrix.item(i))
                    if permiso:
                        print(edges_new_matrix[0])
                        print(edges_sav_matrix[0])
                        print(edges_new_matrix.item(i))
                        permiso = False
        else:
            print('It worked!')

        #return result of test
        self.assertTrue(passed)

        # node_cols = ['node_id', 'grid_id', 'v_nom', 'geom', 'v_res0', 'v_res1',
        #             'peak_load', 'generation_capacity', 'type']
        # for index, row in nodes_sav_df.iterrows():
        #    for col in node_cols:
        #        print(index, col, nodes_new_df.iloc[index].T.loc(col))
        # print(nodes_new_df.iloc[index])
        # print(nodes_sav_df.iloc[index])
        # print(nodes_sav_df.axes)
        # index = 1;
        # print(index, nodes_new_df.iloc[index])
        # print(index, nodes_sav_df.iloc[index])

def init_file_for_tests():
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

    conn.close()

    # export grid to file (pickle)
    save_nd_to_pickle(nd, filename='dingo_grids_example.pkl')


if __name__ == "__main__":
    support.run_unittest(DingoRunTest)
    #init_file_for_tests()
