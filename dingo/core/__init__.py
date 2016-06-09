from dingo.core.network.grids import *
from dingo.core.network.stations import *
from dingo.core.structure.regions import *
from dingo.tools import config as cfg_dingo
from dingo.config import config_db_interfaces as db_int

import pandas as pd

from sqlalchemy.orm import sessionmaker
from geoalchemy2.shape import from_shape
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from shapely.geometry import Point, MultiPoint, MultiLineString

from functools import partial
import pyproj
from shapely.ops import transform
import numpy
from psycopg2.extensions import register_adapter, AsIs

from datetime import datetime

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
        mv_station.grid = mv_grid

        self.add_mv_region(mv_region)

        return mv_region

    def import_mv_regions(self, conn, mv_regions=None):
        """Imports MV regions and MV stations from database, reprojects geo data and and initiates objects.

        Parameters
        ----------
        conn : sqlalchemy.engine.base.Connection object
               Database connection
        mv_regions : List of MV regions/stations (int) to be imported (if empty, all regions & stations are imported)

        Returns
        -------
        Nothing

        See Also
        --------
        build_mv_region : used to instantiate MV region objects
        import_lv_regions : used to import LV regions for every single MV region
        add_peak_demand : used to summarize peak loads of underlying LV regions
        """

        # check arguments
        if not all(isinstance(_, int) for _ in mv_regions):
            raise TypeError('`mv_regions` has to be a list of integers.')

        # get database naming and srid settings from config
        try:
            mv_regions_schema_table = cfg_dingo.get('regions', 'mv_regions')
            mv_stations_schema_table = cfg_dingo.get('stations', 'mv_stations')
            srid = str(int(cfg_dingo.get('geo', 'srid')))
        except OSError:
            print('cannot open config file.')


        # build SQL query
        where_clause = ''
        if mv_regions is not None:
            where_clause = 'WHERE polys.subst_id in (' + ','.join(str(_) for _ in mv_regions) + ')'

        sql = """SELECT polys.subst_id as subst_id,
                        ST_AsText(ST_TRANSFORM(polys.geom, {0})) as poly_geom,
                        ST_AsText(ST_TRANSFORM(subs.geom, {0})) as subs_geom
                 FROM {1} AS polys
                        INNER JOIN {2} AS subs
                        ON (polys.subst_id = subs.id) {3};""".format(srid,
                                                                           mv_regions_schema_table,
                                                                           mv_stations_schema_table,
                                                                           where_clause)

        # read data from db
        mv_data = pd.read_sql_query(sql, conn, index_col='subst_id')
        #mv_data2 = gpd.read_postgis(sql,conn,geom_col=['poly_geom', 'subs_geom', 'pgeom'], index_col='id_db')  # testing geopandas

        # iterate over region/station datasets and initiate objects
        try:
            for poly_id, row in mv_data.iterrows():
                subst_id = poly_id
                region_geo_data = wkt_loads(row['poly_geom'])

                # transform `region_geo_data` to epsg 3035 (from originally 4326)
                # to achieve correct area calculation of mv_region
                # TODO: consider to generally switch to 3035 representation
                station_geo_data = wkt_loads(row['subs_geom'])
                projection = partial(
                    pyproj.transform,
                    pyproj.Proj(init='epsg:4326'),  # source coordinate system
                    pyproj.Proj(init='epsg:3035'))  # destination coordinate system

                region_geo_data = transform(projection, region_geo_data)  # apply projection

                mv_region = self.build_mv_region(poly_id, subst_id, region_geo_data, station_geo_data)
                self.import_lv_regions(conn, mv_region)

                # add sum of peak loads of underlying lv regions to mv_region
                mv_region.add_peak_demand()
        except:
            print('unexpected error while initiating MV regions from DB dataset.')

    def import_lv_regions(self, conn, mv_region):
        """imports LV regions (load areas) from database for a single MV region

        Table definition for load areas can be found here:
        http://vernetzen.uni-flensburg.de/redmine/projects/open_ego/wiki/Methoden_AP_26_DataProc

        Parameters
        ----------
        conn: Database connection
        mv_region : MV region/station (instance of MVRegionDingo class) for which the import of load areas is performed
        """

        lv_regions_schema_table = cfg_dingo.get('regions', 'lv_regions')            # alias in sql statement: `regs`
        lv_loads_schema_table = cfg_dingo.get('loads', 'lv_loads')                  # alias in sql statement: `ploads`
        srid = str(int(cfg_dingo.get('geo', 'srid')))

        lv_loads_threshold = cfg_dingo.get('assumptions', 'load_area_threshold')    # threshold: load area peak load


        load_scaling_factor = 10**6  # load in database is in GW -> scale to kW

        # build SQL query
        # where_clause = 'WHERE areas.mv_poly_id=' + str(mv_region.id_db)
        # where_clause = 'WHERE mv_poly_id=' + str(mv_region.id_db)
        where_clause = 'WHERE subst_id=' + str(mv_region.mv_grid._station.id_db)

        sql = """SELECT regs.id as id_db,
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
                        ST_AsText(ST_TRANSFORM(regs.geom_centre, {0})) as geo_centre,
                        round(ploads.residential::numeric * {1}) as peak_load_residential,
                        round(ploads.retail::numeric * {1}) as peak_load_retail,
                        round(ploads.industrial::numeric * {1}) as peak_load_industrial,
                        round(ploads.agricultural::numeric * {1}) as peak_load_agricultural,
                        round((ploads.residential::numeric + ploads.retail::numeric + ploads.industrial::numeric + ploads.agricultural::numeric) * {1}) as peak_load_sum
                 FROM {2} AS regs
                        INNER JOIN {3} AS ploads
                        ON (regs.id = ploads.id) {4};""".format(srid,
                                                                      load_scaling_factor,
                                                                      lv_regions_schema_table,
                                                                      lv_loads_schema_table,
                                                                      where_clause)

        # read data from db
        lv_regions = pd.read_sql_query(sql, conn, index_col='id_db')

        # create region objects from rows and add them to graph
        for id_db, row in lv_regions.iterrows():

            # only pick load areas with peak load greater than lv_loads_threshold
            # TODO: When migrating to SQLAlchemy, move condition to query
            if row['peak_load_sum'] >= lv_loads_threshold:
                # create LV region object
                lv_region = LVRegionDingo(id_db=id_db, db_data=row, mv_region=mv_region)#, db_cols=lv_regions.columns.values)

                # TODO: Following code is for testing purposes only! (create 1 LV grid and 1 station for every LV region)
                # TODO: The objective is to create stations according to kind of loads (e.g. 1 station for residential, 1 for retail etc.)
                # === START TESTING ===
                # create LV station object
                station_geo_data = wkt_loads(row['geo_centre'])
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

    def adapt_numpy_int64(self, numpy_int64):
        """ Adapting numpy.int64 type to SQL-conform int type using psycopg extension, see [1]_ for more info.

        References
        ----------
        .. [1] http://initd.org/psycopg/docs/advanced.html#adapting-new-python-types-to-sql-syntax
        """
        return AsIs(numpy_int64)

    # TODO: Move to more general place (ego.io repo)

    def export_mv_grid(self, conn, mv_regions):
        """ Exports MV grids to database for visualization purposes

        Parameters
        ----------
        conn : sqlalchemy.engine.base.Connection object
               Database connection
        mv_regions : List of MV regions (instances of MVRegionDingo class) whose MV grids are exported.

        """
        # TODO: currently only station- & line-positions are exported (no further electric data)
        # TODO: method has to be extended to cover more data

        # register adapter
        register_adapter(numpy.int64, self.adapt_numpy_int64)

        # check arguments
        if not all(isinstance(_, int) for _ in mv_regions):
            raise TypeError('`mv_regions` has to be a list of integers.')

        srid = str(int(cfg_dingo.get('geo', 'srid')))

        Session = sessionmaker(bind=conn)
        session = Session()

        # delete all existing datasets
        session.query(db_int.sqla_mv_grid_viz).delete()
        session.commit()

        # build data array from grids (nodes and branches)
        for region in self.mv_regions():
            grid_id = region.mv_grid.id_db
            mv_stations = []
            lv_stations = []
            lines = []

            for node in region.mv_grid._graph.nodes():
                if isinstance(node, LVStationDingo):
                    lv_stations.append((node.geo_data.x, node.geo_data.y))
                elif isinstance(node, MVStationDingo):
                    mv_stations.append((node.geo_data.x, node.geo_data.y))

            # create shapely obj from stations and convert to geoalchemy2.types.WKBElement
            lv_stations_wkb = from_shape(MultiPoint(lv_stations), srid=srid)
            mv_stations_wkb = from_shape(Point(mv_stations), srid=srid)

            for branch in region.mv_grid.graph_edges():
                line = branch['adj_nodes']
                lines.append(((line[0].geo_data.x, line[0].geo_data.y), (line[1].geo_data.x, line[1].geo_data.y)))

            # create shapely obj from lines and convert to geoalchemy2.types.WKBElement
            mv_lines_wkb = from_shape(MultiLineString(lines), srid=srid)

            # add dataset to session
            dataset = db_int.sqla_mv_grid_viz(grid_id=grid_id, timestamp=datetime.now(), geom_mv_station=mv_stations_wkb, geom_lv_stations=lv_stations_wkb, geom_mv_lines=mv_lines_wkb)
            session.add(dataset)

        # commit changes to db
        session.commit()


    def mv_routing(self, debug=False):
        """ Performs routing on all MV grids, see method `routing` in class `MVGridDingo` for details.

        Args:
            debug: If True, information is printed while routing
        """

        for region in self.mv_regions():
            region.mv_grid.routing(debug)

    def mv_parametrize_grid(self, debug=False):
        """ Performs Parametrization of grid equipment of all MV grids, see method `parametrize_grid` in class
            `MVGridDingo` for details.

        Args:
            debug: If True, information is printed while parametrization
        """

        for region in self.mv_regions():
            region.mv_grid.parametrize_grid(debug)

    def __repr__(self):
        return str(self.name)
