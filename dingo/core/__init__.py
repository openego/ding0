from dingo.core.network.grids import *
from dingo.core.network.stations import *
from dingo.core.structure.regions import *
from dingo.tools import config as cfg_dingo
from oemof import db

#import networkx as nx
import pandas as pd
from geopy.distance import vincenty

class NetworkDingo():
    """ Defines the DINGO Network - not a real grid but a container for the MV-grids. Contains the NetworkX graph and
    associated attributes.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self._mv_regions = []

    def mv_regions(self):
        """Returns a generator for iterating over MV regions"""
        for region in self._mv_regions:
            yield region

    def add_mv_region(self, mv_region):
        """Adds a MV region to _mv_regions if not already existing"""
        if mv_region not in self.mv_regions():
            self._mv_regions.append(mv_region)

    def build_mv_region(self, name, region_geo_data, station_geo_data):
        """initiates single MV region including station and grid

        Parameters
        ----------
        name: name of station, grid and region
        region_geo_data: Polygon (shapely object) of region
        station_geo_data: Point (shapely object) of station

        """
        # TODO: validate input params (try..except)

        mv_station = MVStationDingo(name='mvstat'+str(name), geo_data=station_geo_data)
        mv_grid = MVGridDingo(name='mvgrid'+str(name), station=mv_station)
        mv_region = MVRegionDingo(name='mvreg'+str(name), mv_grid=mv_grid, geo_data=region_geo_data)

        self.add_mv_region(mv_region)

        return mv_region

    def import_mv_data(self, mv_regions=None):
        """imports MV regions and MV stations from database

        Parameters
        ----------
        mv_regions : List of MV regions/stations to be imported (if empty, all regions & stations are imported)
        """

        conn = db.connection(section='ontohub_oedb')

        mv_regions_schema_table = cfg_dingo.get('regions', 'mv_regions')
        mv_stations_schema_table = cfg_dingo.get('stations', 'mv_stations')
        mv_regions_index_col = 'id'
        #mv_stations_index_col = 'id'
        srid = '4326' #WGS84: 4326, TODO: Move to global settings

        # build SQL query
        where_clause = ''
        if mv_regions is not None:
            where_clause = 'WHERE subst_id in (' + ','.join(mv_regions) + ')'

        sql = """SELECT ST_AsText(ST_TRANSFORM(polys.geom, {0})) as poly_geom,
                        ST_AsText(ST_TRANSFORM(subs.geom, {0})) as subs_geom
                 FROM {1} AS polys
                        INNER JOIN {2} AS subs
	                    ON (polys.subst_id = subs.id) {3};""".format(srid,
                                                                     mv_regions_schema_table,
                                                                     mv_stations_schema_table,
                                                                     where_clause)

        # read data from db
        mv_data = pd.read_sql_query(sql, conn, index_col=mv_regions_index_col)

        # iterate over region/station datasets and initiate objects
        for idx, row in mv_data.iterrows():
            region_geo_data=row['poly_geom']
            station_geo_data=row['subs_geom']

            mv_region = self.build_mv_region(idx, region_geo_data, station_geo_data)
            self.import_lv_regions(conn, mv_region)

        conn.close()

    def import_lv_regions(self, conn, mv_region):
        """imports LV regions (load areas) from database for a single MV region

        Table definition for load areas can be found here:
        http://vernetzen.uni-flensburg.de/redmine/projects/open_ego/wiki/Methoden_AP_26_DataProc

        Parameters
        ----------
        mv_region : MV region/station for which the import of load areas is performed
        """

        #conn = db.connection(section='ontohub_oedb')

        schema_table = cfg_dingo.get('regions', 'lv_regions')
        index_col = 'la_id'
        srid = '4326' #WGS84: 4326, TODO: Move to global settings

        # build SQL query
        where_clause = 'WHERE mv_poly_id=' + str(mv_region)

        # TODO: Update columns when db table is final
        sql = """SELECT lgid,
                        zensus_sum,
                        zensus_count,
                        zensus_density,
                        ioer_sum,
                        ioer_count,
                        ioer_density,
                        area_lg,
                        sector_area_residential,
                        sector_area_retail,
                        sector_area_industrial,
                        sector_area_agricultural,
                        sector_share_residential,
                        sector_share_retail,
                        sector_share_industrial,
                        sector_share_agricultural,
                        sector_count_residential,
                        sector_count_retail,
                        sector_count_industrial,
                        sector_count_agricultural,
                        sector_consumption_residential,
                        sector_consumption_retail,
                        sector_consumption_industrial,
                        sector_consumption_agricultural,
                        nuts,
                        ST_AsText(ST_TRANSFORM(geom, {0})) as geo_area,
                        ST_AsText(ST_TRANSFORM(geom_centroid, {0})) as geo_centroid,
                        ST_AsText(ST_TRANSFORM(geom_surfacepoint, {0})) as geo_surfacepoint
                 FROM {1} {2};""".format(srid, schema_table, where_clause)

        # read data from db
        lv_regions = pd.read_sql_query(sql, conn, index_col)

        # create region objects from rows and add them to graph
        for idx, row in lv_regions.iterrows():
            lv_region = LVRegionDingo(name='lvreg'+str(idx), db_data=row, mv_region=mv_region)#, db_cols=lv_regions.columns.values)
            mv_region.add_lv_region(lv_region)
            mv_region.mv_grid.graph.add_node(lv_region)

        #conn.close()

    def __repr__(self):
        return str(self.name)