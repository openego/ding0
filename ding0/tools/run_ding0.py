import argparse
import sys
import os
from egoio.tools import db
# from ding0.tools.logger import setup_logger
import logging
from ding0.core import NetworkDing0
from ding0.tools.results import save_nd_to_pickle, load_nd_from_pickle
from ding0.tools.tests import dataframe_equal
from sqlalchemy.orm import sessionmaker


def convert_grid_id_input_to_list(args):

    if args == 'all':
        grid_ids = list(range(1,3609))
    elif os.path.exists(args):
        raise NotImplementedError("Sorry! Grid IDs from file currently not "
                              "implemented.")
    elif '..' in args:
        range_split = args.split('..')
        if  len(range_split) == 2:
            grid_ids = list(range(int(range_split[0]), int(range_split[1]) + 1))
    elif args.isdigit():
        grid_ids = [int(args)]
    else:
        raise IOError('Wrong input for grid id(s).'
                      'See help for details.')

    return grid_ids



def run_ding0():

    parser = argparse.ArgumentParser(
        description="Commandline running" + \
                    "of ding0",
        epilog='',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'grid_ids',
        default='all',
        help="Select grids by its IDs."
             "IDs can be provided as\n"
             " * integer\n"
             " * 234..250 (range)"
             " * /path/to/file (file containing one grid ID per row)"
             " * all (default, select all available grids equals 1..3608)" 
             "Defaults to 'all'.")
    parser.add_argument(
        '--db-config-file',
        dest='db_config_file',
        help='See ego.io docs for explanation.',
        default=os.path.join(os.path.expanduser('~'), '.egoio/config.ini'))
    parser.add_argument(
        '--db-config-section',
        dest='db_config_section',
        help='Section in config file that should be used for database connection.',
        default='oedb')

    args = parser.parse_args(sys.argv[1:])
    # args = parser.parse_args(['54',
    #                           '--db-config-section', 'oedb_remote'])

    grid_id_list = convert_grid_id_input_to_list(args.grid_ids)

    # database connection/ session
    engine = db.connection(section=args.db_config_section,
                           filepath=args.db_config_file)
    session = sessionmaker(bind=engine)()

    # logger = setup_logger()


    for grid_id in grid_id_list:
        # instantiate new ding0 network object
        nd = NetworkDing0(name='network')

        # run DING0 on selected MV Grid District
        nd.run_ding0(session=session,
                     mv_grid_districts_no=[grid_id])

        # export grids to database
        # nd.export_mv_grid(conn, mv_grid_districts)
        # nd.export_mv_grid_new(conn, mv_grid_districts)

        # export grid to file (pickle)
        save_nd_to_pickle(
            nd,
            filename='ding0_grids__{}.pkl'.format(grid_id))



def ding0_compare():
    parser = argparse.ArgumentParser(
        description="Compare to ding0 files",
        epilog='',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'file1',
        help="Path to file 1")
    parser.add_argument(
        'file2',
        help="Path to file 1")

    args = parser.parse_args(sys.argv[1:])

    network_1 = load_nd_from_pickle(filename=args.file1)
    network_2 = load_nd_from_pickle(filename=args.file2)

    passed, msg = dataframe_equal(network_1, network_2)

    print("Equal: ", passed)
    print(msg)


if __name__ == '__main__':
    # run_ding0()

    ding0_compare()