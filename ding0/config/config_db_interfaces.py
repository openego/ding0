"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from ding0.tools import config as cfg_ding0

from geoalchemy2 import Geometry
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# TODO: check docstrings


# ================ DEFINITIONS FOR EXPORTED DATA ===============
class sqla_mv_grid_viz(Base):
    """ SQLAlchemy table definition for the export of MV grids for visualization purposes
    
    #TODO: Check docstrings *before* definitions! is that ok?
    """

    __tablename__ = 'ego_grid_mv_visualization_bunch'
    __table_args__ = {'schema': 'model_draft'}

    #::obj:`type`: Description.
    grid_id                     = sa.Column('grid_id', sa.Integer(), primary_key=True)
    #::obj:`type`: Description.
    geom_mv_station             = sa.Column('geom_mv_station', Geometry(geometry_type='POINT', srid=4326))
    #::obj:`type`: Description.
    geom_mv_cable_dists         = sa.Column('geom_mv_cable_dists', Geometry(geometry_type='MULTIPOINT', srid=4326))
    #::obj:`type`: Description.
    geom_mv_circuit_breakers    = sa.Column('geom_mv_circuit_breakers', Geometry(geometry_type='MULTIPOINT', srid=4326))
    #::obj:`type`: Description.
    geom_lv_load_area_centres   = sa.Column('geom_lv_load_area_centres', Geometry(geometry_type='MULTIPOINT', srid=4326))
    #::obj:`type`: Description.
    geom_lv_stations            = sa.Column('geom_lv_stations', Geometry(geometry_type='MULTIPOINT', srid=4326))
    #::obj:`type`: Description.
    geom_mv_generators          = sa.Column('geom_mv_generators', Geometry(geometry_type='MULTIPOINT', srid=4326))
    #::obj:`type`: Description.
    geom_mv_lines               = sa.Column('geom_mv_lines', Geometry(geometry_type='MULTILINESTRING', srid=4326))


class sqla_mv_grid_viz_branches(Base):
    """ SQLAlchemy table definition for the export of MV grids' branches for visualization purposes
    
    #TODO: Check docstrings *after* definitions! is that ok?
    """

    __tablename__ = 'ego_grid_mv_visualization_branches'
    __table_args__ = {'schema': 'model_draft'}

    branch_id                   = sa.Column(sa.String(25), primary_key=True)
    """:obj:`type`: Description."""
    grid_id                     = sa.Column('grid_id', sa.Integer)
    """:obj:`type`: Description."""
    type_name                   = sa.Column('type_name', sa.String(25))
    """:obj:`type`: Description."""
    type_kind                   = sa.Column('type_kind', sa.String(5))
    """:obj:`type`: Description."""
    type_v_nom                  = sa.Column('type_v_nom', sa.Integer)
    """:obj:`type`: Description."""
    type_s_nom                  = sa.Column('type_s_nom', sa.Float(53))
    """:obj:`type`: Description."""
    length                      = sa.Column('length', sa.Float(53))
    """:obj:`type`: Description."""
    geom                        = sa.Column('geom', Geometry(geometry_type='LINESTRING', srid=4326))
    """:obj:`type`: Description."""
    s_res0                      = sa.Column('s_res0', sa.Float(53))
    """:obj:`type`: Description."""
    s_res1                      = sa.Column('s_res1', sa.Float(53))
    """:obj:`type`: Description."""


class sqla_mv_grid_viz_nodes(Base):
    """ SQLAlchemy table definition for the export of MV grids' branches for visualization purposes
    
    #TODO: Check docstrings *before* definitions! is that ok?
    """

    __tablename__ = 'ego_grid_mv_visualization_nodes'
    __table_args__ = {'schema': 'model_draft'}


    #::obj:`type`: Description.
    node_id                     = sa.Column(sa.String(100), primary_key=True)
    #::obj:`type`: Description.
    grid_id                     = sa.Column('grid_id', sa.Integer)
    #::obj:`type`: Description.
    v_nom                       = sa.Column('v_nom', sa.Integer)
    #::obj:`type`: Description.
    geom                        = sa.Column('geom', Geometry(geometry_type='POINT', srid=4326))
    #::obj:`type`: Description.
    v_res0                      = sa.Column('v_res0', sa.Float(53))
    #::obj:`type`: Description.
    v_res1                      = sa.Column('v_res1', sa.Float(53))
