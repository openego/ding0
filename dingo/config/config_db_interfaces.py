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
    geom_mv_cable_dists         = sa.Column('geom_mv_cable_dists', Geometry(geometry_type='MULTIPOINT', srid=4326))
    geom_mv_circuit_breakers    = sa.Column('geom_mv_circuit_breakers', Geometry(geometry_type='MULTIPOINT', srid=4326))
    geom_lv_load_area_centres   = sa.Column('geom_lv_load_area_centres', Geometry(geometry_type='MULTIPOINT', srid=4326))
    geom_lv_stations            = sa.Column('geom_lv_stations', Geometry(geometry_type='MULTIPOINT', srid=4326))
    geom_mv_generators          = sa.Column('geom_mv_generators', Geometry(geometry_type='MULTIPOINT', srid=4326))
    geom_mv_lines               = sa.Column('geom_mv_lines', Geometry(geometry_type='MULTILINESTRING', srid=4326))


class sqla_mv_grid_viz_branches(Base):
    """ SQLAlchemy table definition for the export of MV grids' branches for visualization purposes
    """

    __tablename__ = 'ego_deu_mv_grids_vis_branches'
    __table_args__ = {'schema': 'calc_ego_grid'}

    branch_id                   = sa.Column(sa.String(25), primary_key=True)
    grid_id                     = sa.Column('grid_id', sa.Integer)
    type_name                   = sa.Column('type_name', sa.String(25))
    type_kind                   = sa.Column('type_kind', sa.String(5))
    type_v_nom                  = sa.Column('type_v_nom', sa.Integer)
    type_s_nom                  = sa.Column('type_s_nom', sa.Float(53))
    length                      = sa.Column('length', sa.Float(53))
    geom                        = sa.Column('geom', Geometry(geometry_type='LINESTRING', srid=4326))
    s_res0                      = sa.Column('s_res0', sa.Float(53))
    s_res1                      = sa.Column('s_res1', sa.Float(53))


class sqla_mv_grid_viz_nodes(Base):
    """ SQLAlchemy table definition for the export of MV grids' branches for visualization purposes
    """

    __tablename__ = 'ego_deu_mv_grids_vis_nodes'
    __table_args__ = {'schema': 'calc_ego_grid'}


    node_id                     = sa.Column(sa.String(100), primary_key=True)
    grid_id                     = sa.Column('grid_id', sa.Integer)
    v_nom                       = sa.Column('v_nom', sa.Integer)
    geom                        = sa.Column('geom', Geometry(geometry_type='POINT', srid=4326))
    v_res0                      = sa.Column('v_res0', sa.Float(53))
    v_res1                      = sa.Column('v_res1', sa.Float(53))
