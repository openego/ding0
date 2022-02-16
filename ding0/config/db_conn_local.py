"""
Session for db connection can be instantiated.
After mandatory OSM data is added to DB,
local loading from sandbox_bw will be deprecated.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def create_session_osm():
    """
    return Session() for sql queries
    """


'''
    # sqlachemy local osm data
    DB = 'postgresql'
    DB_USER = 'postgres'
    DB_PW   = 'labor'
    DB_Name = 'sandbox_bw' # osm data baden-wuerttemberg


    engine_osm = create_engine(DB + '://' + DB_USER + ': ' + DB_PW + '@localhost:5432/' + DB_Name, echo=False)

    Session = sessionmaker(bind=engine_osm)

    return Session()


'''


def create_session_osm():
    """SQLAlchemy session object with valid connection to sonnja database"""

    # print('Please provide connection parameters to database:\n' +
    #          'Hit [Enter] to take defaults')
    host = 'localhost'  # input('host (default 127.0.0.1): ')
    port = '5432'  # input('port (default 5432): ')
    database = 'berlin_osm'  # 'berlin_osm' # bawu_rob_upd
    user = 'RL-INSTITUT\paul.dubielzig'  # input('user (default sonnja): ')
    password = 'labor'
    # password = input('password: ')
    # password = getpass.getpass(prompt='password: ',
    #                               stream=sys.stderr)
    engine_osm = create_engine(
        'postgresql://' + '%s:%s@%s:%s/%s' % (user,
                                              password,
                                              host,
                                              port,
                                              database)).connect()
    print('Password correct! Database connection established.')

    Session = sessionmaker(bind=engine_osm)

    return Session()
