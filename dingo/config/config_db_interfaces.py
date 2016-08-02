from dingo.tools import config as cfg_dingo

from geoalchemy2 import Geometry
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# ================ DEFINITIONS FOR EXPORTED DATA ===============
class sqla_mv_grid_viz(Base):
    """ SQLAlchemy table definition for the export of MV grids for visualization purposes
    """

    __tablename__ = 'ego_deu_mv_grids_vis'
    __table_args__ = {'schema': 'calc_ego_grid'}

    grid_id                     = sa.Column('grid_id', sa.Integer(), primary_key=True)
    geom_mv_station             = sa.Column('geom_mv_station', Geometry(geometry_type='POINT', srid=4326))
    geom_mv_cable_dist          = sa.Column('geom_mv_cable_dist', Geometry(geometry_type='MULTIPOINT', srid=4326))
    geom_mv_circuit_breakers    = sa.Column('geom_mv_circuit_breakers', Geometry(geometry_type='MULTIPOINT', srid=4326))
    geom_lv_stations            = sa.Column('geom_lv_stations', Geometry(geometry_type='MULTIPOINT', srid=4326))
    geom_mv_lines               = sa.Column('geom_mv_lines', Geometry(geometry_type='MULTILINESTRING', srid=4326))
