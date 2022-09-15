"""
Classes(Base) are set up to load data from local postgresql via sqlalchemy.
"""

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import Column, Integer, Float, String
from geoalchemy2 import Geometry
from sqlalchemy.dialects import postgresql

from ding0.data.egon_data import egon_db
from sqlalchemy import create_engine

# todo: read from config
# srid = 3035
# '''
#  # sqlachemy local osm data
# DB = 'postgresql'
# DB_USER = 'postgres'
# DB_PW   = 'labor'
# DB_Name = 'sandbox_bw'
# engine_osm = create_engine(DB + '://' + DB_USER + ': ' + DB_PW + '@localhost:5432/' + DB_Name, echo=False)
#
#
# '''
# host = 'localhost'  # input('host (default 127.0.0.1): ')
# port = '5432'  # input('port (default 5432): ')
# database = 'berlin_osm'  # 'berlin_osm' # bawu_rob_upd
# user = 'RL-INSTITUT\paul.dubielzig'  # input('user (default sonnja): ')
# password = 'labor'
# # password = input('password: ')
# # password = getpass.getpass(prompt='password: ',
# #                               stream=sys.stderr)
# engine_osm = create_engine(
#     'postgresql://' + '%s:%s@%s:%s/%s' % (user,
#                                           password,
#                                           host,
#                                           port,
#                                           database)).connect()

engine_osm = egon_db.engine()

Base = declarative_base()


class Building_wo_Amenity(Base):
    """Building_without_Amenity Model"""

    __tablename__ = "buildings_without_amenities"

    osm_id = Column(Integer, primary_key=True)
    building = Column(String(50))
    area = Column(Float)
    geometry = Column(Geometry('POLYGON'))
    geo_center = Column(Geometry('POINT'))
    name = Column(String(50))
    tags = Column(String(50))
    n_apartments = Column(Float)


class Amenities_ni_Buildings(Base):
    """Amenities_not_in_Buildings Model"""

    __tablename__ = "amenities_not_in_buildings"

    osm_id = Column(Integer, primary_key=True)
    amenity = Column(String(50))
    name = Column(String(50))
    geometry = Column(Geometry('POINT'))
    tags = Column(String(50))


class Buildings_with_Amenities(Base):
    """Buildings_with_Amenities Model"""

    __tablename__ = "buildings_with_amenities"

    osm_id_amenity = Column(Integer, primary_key=True)
    osm_id_building = Column(Integer)
    building = Column(String(50))
    area = Column(Float)
    geometry_building = Column(Geometry('POLYGON'))
    geometry_amenity = Column(Geometry('POINT'))
    geo_center = Column(Geometry('POINT'))
    name = Column(String(50))
    building_tags = Column(String(50))
    amenity_tags = Column(String(50))
    n_amenities_inside = Column(Float)
    n_apartments = Column(Float)


class Way(Base):
    """Way Model"""

    __tablename__ = "ways_with_segments_3035"

    osm_id = Column(Integer, primary_key=True)
    nodes = Column(postgresql.ARRAY(Integer))
    highway = Column(String(50))
    geometry = Column(Geometry('LINESTRING', srid="3035"))
    length_segments = Column(postgresql.ARRAY(Float))


# Migrate the initial model changes.
# But to which connection/enginee should it reflect is refered passing enginee.
Base.metadata.create_all(engine_osm)
