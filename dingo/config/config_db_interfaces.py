from dingo.tools import config as cfg_dingo

from geoalchemy2 import Geometry
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# ================ DEFINITIONS FOR IMPORTED DATA ===============
class sqla_mv_region(Base):
    """ SQLAlchemy table definition for MV regions
    """

    __tablename__ = 'grid_district'
    __table_args__ = {'schema': 'calc_ego_grid_district'}

    subst_id    = sa.Column('subst_id', sa.Integer(), primary_key=True)
    geom        = sa.Column('geom', Geometry(geometry_type='MULTIPOLYGON', srid=3035))


class sqla_mv_station(Base):
    """ SQLAlchemy table definition for MV substations
    """

    __tablename__ = 'ego_deu_substations'
    __table_args__ = {'schema': 'calc_ego_substation'}

    id      = sa.Column('id', sa.Integer(), primary_key=True)
    geom    = sa.Column('geom', Geometry(geometry_type='POINT', srid=4326))


class sqla_lv_region(Base):
    """ SQLAlchemy table definition for LV load areas
    """

    __tablename__ = 'ego_deu_load_area_ta'
    __table_args__ = {'schema': 'calc_ego_loads'}

    id                          = sa.Column('id', sa.Integer(), primary_key=True)
    zensus_sum                  = sa.Column('zensus_sum', sa.Integer())
    zensus_count                = sa.Column('zensus_count', sa.Integer())
    ioer_sum                    = sa.Column('ioer_sum', sa.Numeric())
    ioer_count                  = sa.Column('ioer_count', sa.Integer())
    area_ha                     = sa.Column('area_ha', sa.Numeric())

    sector_area_residential     = sa.Column('sector_area_residential', sa.Numeric())
    sector_area_retail          = sa.Column('sector_area_retail', sa.Numeric())
    sector_area_industrial      = sa.Column('sector_area_industrial', sa.Numeric())
    sector_area_agricultural    = sa.Column('sector_area_agricultural', sa.Numeric())

    sector_share_residential    = sa.Column('sector_share_residential', sa.Numeric())
    sector_share_retail         = sa.Column('sector_share_retail', sa.Numeric())
    sector_share_industrial     = sa.Column('sector_share_industrial', sa.Numeric())
    sector_share_agricultural   = sa.Column('sector_share_agricultural', sa.Numeric())

    sector_count_residential    = sa.Column('sector_count_residential', sa.Integer())
    sector_count_retail         = sa.Column('sector_count_retail', sa.Integer())
    sector_count_industrial     = sa.Column('sector_count_industrial', sa.Integer())
    sector_count_agricultural   = sa.Column('sector_count_agricultural', sa.Integer())
    nuts                        = sa.Column('nuts', sa.VARCHAR(5))

    geom                        = sa.Column('geom', Geometry(geometry_type='POLYGON', srid=3035))
    geom_centre                 = sa.Column('geom_centre', Geometry(geometry_type='POINT', srid=3035))


class sqla_lv_peakload(Base):
    """ SQLAlchemy table definition for LV peak loads
    """

    __tablename__ = 'calc_ego_peak_load_ta'
    __table_args__ = {'schema': 'calc_ego_loads'}

    id              = sa.Column('id', sa.BigInteger(), primary_key=True)
    residential     = sa.Column('residential', sa.Numeric())
    retail          = sa.Column('retail', sa.Numeric())
    industrial      = sa.Column('industrial', sa.Numeric())
    agricultural    = sa.Column('agricultural', sa.Numeric())


# ================ DEFINITIONS FOR EXPORTED DATA ===============
class sqla_mv_grid_viz(Base):
    """ SQLAlchemy table definition for the export of MV grids for visualization purposes
    """

    __tablename__ = 'ego_deu_mv_grids_vis'
    __table_args__ = {'schema': 'calc_ego_grid'}

    grid_id             = sa.Column('grid_id', sa.Integer(), primary_key=True)
    timestamp           = sa.Column('timestamp', sa.TIMESTAMP())
    geom_mv_station     = sa.Column('geom_mv_station', Geometry(geometry_type='POINT', srid=4326))
    geom_lv_stations    = sa.Column('geom_lv_stations', Geometry(geometry_type='MULTIPOINT', srid=4326))
    geom_mv_lines       = sa.Column('geom_mv_lines', Geometry(geometry_type='MULTILINESTRING', srid=4326))
