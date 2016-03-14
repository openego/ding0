from dingo.core.network.grids import *
from dingo.core.network.stations import *
from dingo.core.structure.regions import *
from dingo.tools import config as cfg_dingo
from oemof import db

import networkx as nx
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
        self.mv_regions = kwargs.get('mv_regions', [])

        #self.graph = nx.Graph()

    def build_mv_structure(self, name, region_geo_data, station_geo_data):
        """initiates single MV station, grid and region

        Parameters
        ----------
        name: name of station, grid and region
        region_geo_data: Polygon (shapely object) of region
        station_geo_data: Point (shapely object) of station

        """
        # TODO: validate input params (try..except)

        mv_station = MVStationDingo(name='mvstation_'+str(name), geo_data=station_geo_data)
        mv_grid = MVGridDingo(name='mvgrid_'+str(name), station=mv_station)
        mv_region = MVRegionDingo(name='mvregion_'+str(name), mv_grid=mv_grid, geo_data=region_geo_data)

        self.mv_regions.append(mv_region)

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
        srid = '4326' #WGS84: 4326

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

            self.build_mv_structure(idx, region_geo_data, station_geo_data)
            self.import_lv_regions(conn, idx)

        conn.close()

    def import_lv_regions(self, conn, mv_regions=None):
        """imports LV regions (load areas) from database

        Table definition for load areas can be found here:
        http://vernetzen.uni-flensburg.de/redmine/projects/open_ego/wiki/Methoden_AP_26_DataProc

        Parameters
        ----------
        mv_regions : List of MV regions/stations for which the import of load ares is performed (if empty, the import is
                     done for all regions & stations)
        """

        #conn = db.connection(section='ontohub_oedb')

        schema_table = cfg_dingo.get('regions', 'lv_regions')
        index_col = 'lgid'
        srid = '4326' #WGS84: 4326

        # build SQL query
        # TODO: insert WHERE-statement here according to Lui's upcoming table definition. Pseudo-Code: SELECT stuff FROM all LV-regions which are within MV-Region (Polygon) xy
        # TODO: LUI: mv_poly_id

        for mv_region in mv_regions:

            where_clause = 'WHERE subst_id=' + str(mv_region)

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
                            ST_AsText(ST_TRANSFORM(geom, {0})) as geom_area,
                            ST_AsText(ST_TRANSFORM(geom_centroid, {0})) as geom_centroid,
                            ST_AsText(ST_TRANSFORM(geom_surfacepoint, {0})) as geom_surfacepoint
                     FROM {1} {2};""".format(srid, schema_table, where_clause)

            # read data from db
            lv_regions = pd.read_sql_query(sql, conn, index_col)

            # create region objects from rows and add them to graph
            for idx, row in lv_regions.iterrows():
                region_obj = LVRegionDingo(db_data=row, mv_region=)#, db_cols=lv_regions.columns.values)
                #self.graph.add_node(station_obj)

        conn.close()