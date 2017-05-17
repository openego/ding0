"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DINGO lives at github: https://github.com/openego/dingo/
The documentation is available on RTD: http://dingo.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/dingo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from dingo.tools import config as cfg_dingo

from geoalchemy2 import Geometry
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# ================ DEFINITIONS FOR EXPORTED DATA ===============
class sqla_mv_grid_viz(Base):
    """ SQLAlchemy table definition for the export of MV grids for visualization purposes
    """

    __tablename__ = 'ego_grid_mv_visualization_bunch'
    __table_args__ = {'schema': 'model_draft'}

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

    __tablename__ = 'ego_grid_mv_visualization_branches'
    __table_args__ = {'schema': 'model_draft'}

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

    __tablename__ = 'ego_grid_mv_visualization_nodes'
    __table_args__ = {'schema': 'model_draft'}


    node_id                     = sa.Column(sa.String(100), primary_key=True)
    grid_id                     = sa.Column('grid_id', sa.Integer)
    v_nom                       = sa.Column('v_nom', sa.Integer)
    geom                        = sa.Column('geom', Geometry(geometry_type='POINT', srid=4326))
    v_res0                      = sa.Column('v_res0', sa.Float(53))
    v_res1                      = sa.Column('v_res1', sa.Float(53))
