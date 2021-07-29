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

    # sqlachemy local osm data
    DB = 'postgresql'
    DB_USER = 'postgres'
    DB_PW   = 'labor'
    DB_Name = 'sandbox_bw' # osm data baden-wuerttemberg


    engine_osm = create_engine(DB + '://' + DB_USER + ': ' + DB_PW + '@localhost:5432/' + DB_Name, echo=False)

    Session = sessionmaker(bind=engine_osm)
    
    return Session()

