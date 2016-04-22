from dingo.core.network.grids import *
from dingo.core.network.stations import *
from dingo.core.structure.regions import *
from dingo.tools import config as cfg_dingo

import pandas as pd
from shapely.wkt import loads as wkt_loads


class NetworkDingo:
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
        # TODO: use setter method here (make attribute '_mv_regions' private)
        if mv_region not in self.mv_regions():
            self._mv_regions.append(mv_region)

    def build_mv_region(self, poly_id, subst_id, region_geo_data, station_geo_data):
        """initiates single MV region including station and grid

        Parameters
        ----------
        poly_id: ID of region according to database table. Also used as ID for created grid
        subst_id: ID of station according to database table
        region_geo_data: Polygon (shapely object) of region
        station_geo_data: Point (shapely object) of station

        """
        # TODO: validate input params

        mv_station = MVStationDingo(id_db=subst_id, geo_data=station_geo_data)

        mv_grid = MVGridDingo(id_db=poly_id, station=mv_station)
        mv_region = MVRegionDingo(id_db=poly_id, mv_grid=mv_grid, geo_data=region_geo_data)
        mv_grid.region = mv_region

        self.add_mv_region(mv_region)

        return mv_region

    def import_mv_regions(self, conn, mv_regions=None):
        """imports MV regions and MV stations from database

        Parameters
        ----------
        conn: Database connection
        mv_regions : List of MV regions/stations (int) to be imported (if empty, all regions & stations are imported)
        """

        # check arguments
        if not all(isinstance(_, int) for _ in mv_regions):
            raise Exception('Type error: `mv_regions` has to be a list of integers.')

        # get database naming settings from config
        mv_regions_schema_table = cfg_dingo.get('regions', 'mv_regions')
        mv_stations_schema_table = cfg_dingo.get('stations', 'mv_stations')

        srid = '4326'  # WGS84: 4326, TODO: Move to global settings

        # build SQL query
        where_clause = ''
        if mv_regions is not None:
            where_clause = 'WHERE polys.id in (' + ','.join(str(_) for _ in mv_regions) + ')'

        sql = """SELECT polys.subst_id as subst_id,
                        polys.id as poly_id,
                        ST_AsText(ST_TRANSFORM(polys.geom, {0})) as poly_geom,
                        ST_AsText(ST_TRANSFORM(subs.geom, {0})) as subs_geom
                 FROM {1} AS polys
                        INNER JOIN {2} AS subs
                        ON (polys.subst_id = subs.subst_id) {3};""".format(srid,
                                                                           mv_regions_schema_table,
                                                                           mv_stations_schema_table,
                                                                           where_clause)

        # read data from db
        mv_data = pd.read_sql_query(sql, conn, index_col='poly_id')
        #mv_data2 = gpd.read_postgis(sql,conn,geom_col=['poly_geom', 'subs_geom', 'pgeom'], index_col='id_db')

        # iterate over region/station datasets and initiate objects
        for poly_id, row in mv_data.iterrows():
            subst_id = row['subst_id']
            region_geo_data = wkt_loads(row['poly_geom'])
            station_geo_data = wkt_loads(row['subs_geom'])

            mv_region = self.build_mv_region(poly_id, subst_id, region_geo_data, station_geo_data)
            self.import_lv_regions(conn, mv_region)

            # add sum of peak loads of underlying lv regions to mv_region
            mv_region.add_peak_demand()

    def import_lv_regions(self, conn, mv_region):
        """imports LV regions (load areas) from database for a single MV region

        Table definition for load areas can be found here:
        http://vernetzen.uni-flensburg.de/redmine/projects/open_ego/wiki/Methoden_AP_26_DataProc

        Parameters
        ----------
        conn: Database connection
        mv_region : MV region/station (instance of MVRegionDingo class) for which the import of load areas is performed
        """

        lv_regions_schema_table = cfg_dingo.get('regions', 'lv_regions')    # alias in sql statement: `regs`
        lv_loads_schema_table = cfg_dingo.get('loads', 'lv_loads')          # alias in sql statement: `ploads`

        srid = '4326'  # WGS84: 4326, TODO: Move to global settings
        load_scaling_factor = 10**6  # load in database is in GW -> scale to kW

        # build SQL query
        #where_clause = 'WHERE areas.mv_poly_id=' + str(mv_region.id_db)
        where_clause = 'WHERE mv_poly_id=' + str(mv_region.id_db)

        sql = """SELECT regs.la_id as id_db,
                        regs.zensus_sum,
                        regs.zensus_count as zensus_cnt,
                        regs.ioer_sum,
                        regs.ioer_count as ioer_cnt,
                        regs.area_ha as area,
                        regs.sector_area_residential,
                        regs.sector_area_retail,
                        regs.sector_area_industrial,
                        regs.sector_area_agricultural,
                        regs.sector_share_residential,
                        regs.sector_share_retail,
                        regs.sector_share_industrial,
                        regs.sector_share_agricultural,
                        regs.sector_count_residential,
                        regs.sector_count_retail,
                        regs.sector_count_industrial,
                        regs.sector_count_agricultural,
                        regs.nuts as nuts_code,
                        ST_AsText(ST_TRANSFORM(regs.geom, {0})) as geo_area,
                        ST_AsText(ST_TRANSFORM(regs.geom_centroid, {0})) as geo_centroid,
                        ST_AsText(ST_TRANSFORM(regs.geom_surfacepoint, {0})) as geo_surfacepnt,
                        round(ploads.residential::numeric * {1}) as peak_load_residential,
                        round(ploads.retail::numeric * {1}) as peak_load_retail,
                        round(ploads.industrial::numeric * {1}) as peak_load_industrial,
                        round(ploads.agricultural::numeric * {1}) as peak_load_agricultural,
                        round((ploads.residential::numeric + ploads.retail::numeric + ploads.industrial::numeric + ploads.agricultural::numeric) * {1}) as peak_load_sum
                 FROM {2} AS regs
                        INNER JOIN {3} AS ploads
                        ON (regs.la_id = ploads.la_id) {4};""".format(srid,
                                                                      load_scaling_factor,
                                                                      lv_regions_schema_table,
                                                                      lv_loads_schema_table,
                                                                      where_clause)

        # read data from db
        lv_regions = pd.read_sql_query(sql, conn, index_col='id_db')

        # create region objects from rows and add them to graph
        for id_db, row in lv_regions.iterrows():
            # create LV region object
            lv_region = LVRegionDingo(id_db=id_db, db_data=row, mv_region=mv_region)#, db_cols=lv_regions.columns.values)

            # TODO: Following code is for testing purposes only! (create 1 LV grid and 1 station for every LV region)
            # TODO: The objective is to create stations according to kind of loads (e.g. 1 station for residential, 1 for retail etc.)
            # === START TESTING ===
            # create LV station object
            station_geo_data = wkt_loads(row['geo_surfacepnt'])
            lv_station = LVStationDingo(id_db=id_db, geo_data=station_geo_data, peak_load=row['peak_load_sum'])
            lv_grid = LVGridDingo(region=lv_region, id_db=id_db, geo_data=station_geo_data)
            lv_station.grid = lv_grid
            # add LV station to LV grid
            lv_grid.add_station(lv_station)
            # add LV grid to LV region
            lv_region.add_lv_grid(lv_grid)
            # === END TESTING ===

            # add LV region to MV region
            mv_region.add_lv_region(lv_region)

            # OLD:
            # add LV region to MV grid graph
            # TODO: add LV station instead of LV region
            #mv_region.mv_grid.graph_add_node(lv_region)

    def export_mv_grid(self, mv_regions):
        """ Exports MV grids to database

        Args:
            mv_regions: List of MV regions (instances of MVRegionDingo class) whose MV grids are exported.
        """

        mv_grids_schema_table = cfg_dingo.get('grids', 'mv_grids')

        srid = '4326'  # WGS84: 4326, TODO: Move to global settings

        # TODO: Breath life into this method :). Prior to this the table structure has to be defined

    def mv_routing(self, debug=False):
        """ Performs routing on all MV grids, see method `routing` in class `MVGridDingo` for details

        Args:
            debug: If True, information is printed while routing
        """

        for region in self.mv_regions():
            region.mv_grid.routing(debug)

    def parametrize_grid(self):
        """Paramtrization of grid equipment"""

        # Parameters of possible transformers
        # TODO: move to database of config file
        transformers = {
            20000: {
                'voltage_level': 10,
                'apparent_power': 20000},
            31500: {
                'voltage_level': 20,
                'apparent_power': 31500},
            40000: {
                'voltage_level': 20,
                'apparent_power': 40000}}

        for mv_region in self._mv_regions:

            # choose appropriate transformers for each sub-station
            mv_region.mv_grid._station.choose_transformers(transformers,
               **{'peak_load': mv_region.peak_load})

    def __repr__(self):
        return str(self.name)