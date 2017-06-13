import unittest2 as unittest
from test import support

from dingo.tools import results
from matplotlib import pyplot as plt

class DingoRunTest(unittest.TestCase):

    filename = 'dingo_grids_example.pkl'

    def setUp(self):
        print("setup")

    def tearDown(self):
        print("cleanup")

    def test_exemplary_data(self):
        print(self.filename)

        nd = results.load_nd_from_pickle(filename=self.filename)

        nodes_df, edges_df = nd.to_dataframe()



if __name__ == "__main__":
    support.run_unittest(DingoRunTest)
