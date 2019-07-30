import unittest2 as unittest

from egoio.tools import db
from ding0.core import NetworkDing0
from ding0.tools.logger import setup_logger
from ding0.tools.results import save_nd_to_pickle
from ding0.tools.results import load_nd_from_pickle

from geoalchemy2.shape import to_shape
from sqlalchemy.orm import sessionmaker
from ding0.tools.results import (calculate_lvgd_stats, calculate_lvgd_voltage_current_stats, calculate_mvgd_stats,
                                 calculate_mvgd_voltage_current_stats)

import logging
import os

logger = setup_logger(loglevel=logging.CRITICAL)


class Ding0RunTest(unittest.TestCase):

    def setUp(self):
        print('\n')

    def test_files(self):
        print('\n########################################')
        print('Test File vs File')
        print('\n########################################')
        print('  Loading data...')
        nw_1 = load_nd_from_pickle(filename='ding0_tests_grids_1.pkl')
        nw_2 = load_nd_from_pickle(filename='ding0_tests_grids_2.pkl')
        #test equality
        print('\n########################################')
        print('  Testing equality...')
        equals_e, msg = dataframe_equal(nw_1,nw_1)
        print('  ...'+msg)
        #test difference
        print('\n########################################')
        print('  Testing difference...')
        equals_d, msg = dataframe_equal(nw_1,nw_2)
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

    def test_ding0_file(self):
        print('\n########################################')
        print('Test ding0 vs File')
        print('\n########################################')
        print('  Loading data...')
        nw_1 = load_nd_from_pickle(filename='ding0_tests_grids_1.pkl')

        print('\n########################################')
        print('  Running ding0 for the same configuration...')

        # database connection/ session
        engine = db.connection(section='oedb')
        session = sessionmaker(bind=engine)()
        mv_grid_districts = [3545]

        nw_2 = NetworkDing0(name='network')
        nw_2.run_ding0(session=session, mv_grid_districts_no=mv_grid_districts)

        #test equality
        print('  Testing equality...')
        passed, msg = dataframe_equal(nw_1,nw_2)
        print('    ...'+msg)

        self.assertTrue(passed,msg=msg)

    def test_ding0(self):
        print('\n########################################')
        print('Test ding0 vs ding0')
        # database connection/ session
        engine = db.connection(section='oedb')
        session = sessionmaker(bind=engine)()

        mv_grid_districts = [3545]

        print('\n########################################')
        print('  Running ding0 once...')
        nw_1 = NetworkDing0(name='network')
        nw_1.run_ding0(session=session, mv_grid_districts_no=mv_grid_districts)

        print('\n########################################')
        print('  Running ding0 twice...')
        nw_2 = NetworkDing0(name='network')
        nw_2.run_ding0(session=session, mv_grid_districts_no=mv_grid_districts)

        #test equality
        print('\n########################################')
        print('  Testing equality...')
        passed, msg = dataframe_equal(nw_1,nw_2)
        print('    ...'+msg)

        self.assertTrue(passed,msg=msg)

def dataframe_equal(network_one, network_two):
    ''' Compare two networks and returns True if they are identical
    
    Parameters
    ----------
    network_one: :class:`~.ding0.core.GridDing0`
    network_two: :class:`~.ding0.core.GridDing0`
    
    Returns
    -------
    bool
        True if both networks are identical, False otherwise.
    str
        A message explaining the result.
    '''
    #initiate dataframes through to_dataframe method
    nodes_one_df, edges_one_df = network_one.to_dataframe()
    nodes_two_df, edges_two_df = network_two.to_dataframe()

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
    #Circuit Breakers don't always have the same numeration,
    #so we need to reorder them according to their position
    cb_one = nodes_one_df[nodes_one_df['type']=='Switch Disconnector'].sort_values('geom')['node_id'].reset_index(drop=True).tolist()
    cb_two = nodes_two_df[nodes_two_df['type']=='Switch Disconnector'].sort_values('geom')['node_id'].reset_index(drop=True).tolist()
    cb_one_new_name_list = ['Circuit_breaker_'+str(n+1) for n in range(0,len(cb_one))]
    cb_two_new_name_list = ['Circuit_breaker_'+str(n+1) for n in range(0,len(cb_two))]

    nodes_one_df['node_id'].replace(to_replace=cb_one,value=cb_one_new_name_list,inplace=True)
    nodes_two_df['node_id'].replace(to_replace=cb_two,value=cb_two_new_name_list,inplace=True)
    nodes_one_df = nodes_one_df.sort_values('node_id').reset_index(drop=True)
    nodes_two_df = nodes_two_df.sort_values('node_id').reset_index(drop=True)

    #Similar for edges, but the extreme nodes of an edge can be switched
    #    first, convert to shape and rescue coordinates
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

    decimal_places = 4
    tol = 10 ** -decimal_places
    #avoid rounding of s_res and length
    delta_edges = edges_one_df[['length','s_res0', 's_res1']] - \
                  edges_two_df[['length','s_res0', 's_res1']]
    for idx, row in delta_edges.abs().iterrows():
        if row['s_res0']<tol:
            edges_two_df.loc[idx,'s_res0']=edges_one_df.loc[idx,'s_res0']
        if row['s_res1']<tol:
            edges_two_df.loc[idx,'s_res1']=edges_one_df.loc[idx,'s_res1']
        if row['length']<tol:
            edges_two_df.loc[idx,'length']=edges_one_df.loc[idx,'length']
    #avoid rounding of v_res
    delta_nodes = nodes_one_df[['v_res0', 'v_res1']] - \
                  nodes_two_df[['v_res0', 'v_res1']]
    for idx, row in delta_nodes.abs().iterrows():
        if row['v_res0']<tol:
            nodes_two_df.loc[idx,'v_res0']=nodes_one_df.loc[idx,'v_res0']
        if row['v_res1']<tol:
            nodes_two_df.loc[idx,'v_res1']=nodes_one_df.loc[idx,'v_res1']

    # compare things
    flag_nodes = nodes_one_df[nodes_one_df['type']!='Switch Disconnector'].equals(nodes_two_df[nodes_one_df['type']!='Switch Disconnector'])
    flag_edges = edges_one_df.equals(edges_two_df)
    flag_cb    = nodes_one_df[nodes_one_df['type']=='Switch Disconnector']['geom'].equals(nodes_two_df[nodes_two_df['type']=='Switch Disconnector']['geom'])
    passed     = flag_nodes and flag_edges and flag_cb

    #return result of test
    msg = 'Data sets are '
    if passed:
        msg = msg + 'identical.'
    elif (not flag_edges) and (not flag_nodes):
        msg = msg + 'different in nodes and edges'
    elif not flag_edges:
        msg = msg + 'different in edges'
    elif (not flag_cb) and flag_nodes:
        msg = msg + 'different only in circuit breakers: allocated in (slightly) different places'
    elif not flag_nodes:
        msg = msg + 'different in nodes'
    return passed, msg

def init_files_for_tests( mv_grid_districts= [3545],filename='ding0_tests_grids_1.pkl'):
    '''Runs ding0 over the districtis selected in mv_grid_districts and writes the result in filename.
    
    Parameters
    ----------
    mv_grid_districts: :obj:`list` of :obj:`int`
        Districts IDs: Defaults to [3545]
    filename: :obj:`str`
        Defaults to 'ding0_tests_grids_1.pkl'
    
    '''
    print('\n########################################')
    print('  Running ding0 for district',mv_grid_districts)

    # database connection/ session
    engine = db.connection(section='oedb')
    session = sessionmaker(bind=engine)()

    # instantiate new ding0 network object
    nd = NetworkDing0(name='network')

    # run DING0 on selected MV Grid District
    nd.run_ding0(session=session,mv_grid_districts_no=mv_grid_districts)

    # export grid to file (pickle)
    print('\n########################################')
    print('  Saving result in ',filename)
    save_nd_to_pickle(nd, filename=filename)


def manual_ding0_test(mv_grid_districts=[3545],
                      filename='ding0_tests_grids_1.pkl'):
    ''' Compares a new run of ding0 over districts and an old one saved in
    filename.
    
    Parameters
    ----------
    mv_grid_districts: :obj:`list` of :obj:`int`
        Districts IDs: Defaults to [3545]
    filename: :obj:`str`
        Defaults to 'ding0_tests_grids_1.pkl'
    '''
    print('\n########################################')
    print('Test ding0 vs File')
    print('\n########################################')
    print('  Loading file', filename,'...')
    nw_1 = load_nd_from_pickle(filename=filename)

    print('\n########################################')
    print('  Running ding0 for district',mv_grid_districts, '...')

    # database connection/ session
    engine = db.connection(section='oedb')
    session = sessionmaker(bind=engine)()

    nw_2 = NetworkDing0(name='network')
    nw_2.run_ding0(session=session, mv_grid_districts_no=mv_grid_districts)

    # test equality
    print('\n########################################')
    print('  Testing equality...')
    passed, msg = dataframe_equal(nw_1, nw_2)
    print('    ...' + msg)

def update_stats_test_data(path, pkl_file=None, pkl_path = ''):
    '''
    If changes in electrical data have been made, run this function to update the saved test data in folder.
    Test are run on mv_grid_district 460.
    :param path: directory where testdata ist stored. Normally: ...ding0/tests/core/network/testdata
    :param pkl_file: string of pkl-file of network; optionally, if None new Network is initiated.
    :return:
    '''

    if pkl_file is not None:
        nd = load_nd_from_pickle(pkl_file,pkl_path)
    else:
        # database connection/ session
        engine = db.connection(section='oedb')
        session = sessionmaker(bind=engine)()

        # instantiate new ding0 network object
        nd = NetworkDing0(name='network')

        # choose MV Grid Districts to import
        mv_grid_districts = [460]

        # run DING0 on selected MV Grid District
        nd.run_ding0(session=session,
                     mv_grid_districts_no=mv_grid_districts)

    mvgd_stats = calculate_mvgd_stats(nd)
    mvgd_voltage_current_stats = calculate_mvgd_voltage_current_stats(nd)
    mvgd_voltage_nodes = mvgd_voltage_current_stats[0]
    mvgd_current_branches = mvgd_voltage_current_stats[1]
    mvgd_stats.to_csv(os.path.join(path,'mvgd_stats.csv'))
    mvgd_voltage_nodes.to_csv(os.path.join(path, 'mvgd_voltage_nodes.csv'))
    mvgd_current_branches.to_csv(os.path.join(path, 'mvgd_current_branches.csv'))

    lvgd_stats = calculate_lvgd_stats(nd)
    lvgd_voltage_current_stats = calculate_lvgd_voltage_current_stats(nd)
    lvgd_voltage_nodes = lvgd_voltage_current_stats[0]
    lvgd_current_branches = lvgd_voltage_current_stats[1]
    lvgd_stats.to_csv(os.path.join(path, 'lvgd_stats.csv'))
    lvgd_voltage_nodes.to_csv(os.path.join(path, 'lvgd_voltage_nodes.csv'))
    lvgd_current_branches.to_csv(os.path.join(path, 'lvgd_current_branches.csv'))


if __name__ == "__main__":
    #To run default tests, decomment following line
    #support.run_unittest(Ding0RunTest)

    #To initialize tests comparison files, decomment the following
    #init_files_for_tests()
    #init_files_for_tests([438],'ding0_tests_grids_2.pkl')

    #To test a ding0 run with respect to a saved file
    manual_ding0_test()
    #manual_ding0_test([438],'ding0_tests_grids_2.pkl')
