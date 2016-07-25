from dingo.core.network.grids import *
from dingo.core.network.stations import *
from dingo.core.structure.regions import *
from dingo.tools import config as cfg_dingo
from dingo.tools.animation import AnimationDingo
from dingo.config import config_db_interfaces as db_int

# import ORM classes for oedb access depending on input in config file
cfg_dingo.load_config('config_db_tables')
GridDistrict_name = cfg_dingo.get('regions', 'grid_district')
EgoDeuSubstation_name = cfg_dingo.get('stations', 'mv_stations')
EgoDeuLoadArea_name = cfg_dingo.get('regions', 'lv_load_areas')
CalcEgoPeakLoad_name = cfg_dingo.get('loads', 'lv_loads')
EgoDeuOnts_name = cfg_dingo.get('stations', 'lv_stations')

from egoio.db_tables import calc_ego_substation as orm_mod_calc_ego_substation
from egoio.db_tables import calc_ego_grid_district as orm_calc_ego_grid_district
from egoio.db_tables import calc_ego_loads as orm_calc_ego_loads

orm_EgoDeuSubstation = orm_mod_calc_ego_substation.\
    __getattribute__(EgoDeuSubstation_name)
orm_GridDistrict = orm_calc_ego_grid_district.\
    __getattribute__(GridDistrict_name)
orm_EgoDeuLoadArea = orm_calc_ego_loads.__getattribute__(EgoDeuLoadArea_name)
orm_CalcEgoPeakLoad = orm_calc_ego_loads.__getattribute__(CalcEgoPeakLoad_name)
orm_EgoDeuOnts = orm_mod_calc_ego_substation.__getattribute__(EgoDeuOnts_name)

import pandas as pd

from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, Numeric
from geoalchemy2.shape import from_shape
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from shapely.geometry import Point, MultiPoint, MultiLineString

from functools import partial
import pyproj
from shapely.ops import transform

from datetime import datetime


class NetworkDingo:
    """ Defines the DINGO Network - not a real grid but a container for the
    MV-grids. Contains the NetworkX graph and associated attributes.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self._mv_grid_districts = []

    def mv_grid_districts(self):
        """Returns a generator for iterating over MV grid_districts"""
        for grid_district in self._mv_grid_districts:
            yield grid_district

    def add_mv_grid_district(self, mv_grid_district):
        """Adds a MV grid_district to _mv_grid_districts if not already existing"""
        # TODO: use setter method here (make attribute '_mv_grid_districts' private)
        if mv_grid_district not in self.mv_grid_districts():
            self._mv_grid_districts.append(mv_grid_district)

    def build_mv_grid_district(self, poly_id, subst_id, grid_district_geo_data,
                        station_geo_data):
        """initiates single MV grid_district including station and grid

        Parameters
        ----------
        poly_id: ID of grid_district according to database table. Also used as ID for
            created grid
        subst_id: ID of station according to database table
        grid_district_geo_data: Polygon (shapely object) of grid district
        station_geo_data: Point (shapely object) of station

        """
        # TODO: validate input params

        mv_station = MVStationDingo(id_db=subst_id, geo_data=station_geo_data)

        mv_grid = MVGridDingo(id_db=poly_id,
                              station=mv_station)
        mv_grid_district = MVGridDistrictDingo(id_db=poly_id,
                                               mv_grid=mv_grid,
                                               geo_data=grid_district_geo_data)
        mv_grid.grid_district = mv_grid_district
        mv_station.grid = mv_grid

        self.add_mv_grid_district(mv_grid_district)

        return mv_grid_district

    def import_mv_grid_districts(self, conn, mv_grid_districts=None):
        """Imports MV grid_districts and MV stations from database, reprojects geodata
        and and initiates objects.

        Parameters
        ----------
        conn : sqlalchemy.engine.base.Connection object
               Database connection
        mv_grid_districts : List of MV grid_districts/stations (int) to be imported (if empty,
            all grid_districts & stations are imported)

        Returns
        -------
        Nothing

        See Also
        --------
        build_mv_grid_district : used to instantiate MV grid_district objects
        import_lv_load_areas : used to import load_areas for every single MV grid_district
        add_peak_demand : used to summarize peak loads of underlying load_areas
        """

        # check arguments
        if not all(isinstance(_, int) for _ in mv_grid_districts):
            raise TypeError('`mv_grid_districts` has to be a list of integers.')

        # get database naming and srid settings from config
        try:
            mv_grid_districts_schema_table = cfg_dingo.get('regions', 'mv_grid_districts')
            mv_stations_schema_table = cfg_dingo.get('stations', 'mv_stations')
            srid = str(int(cfg_dingo.get('geo', 'srid')))
        except OSError:
            print('cannot open config file.')


        # build SQL query
        Session = sessionmaker(bind=conn)
        session = Session()
        grid_districts = session.query(orm_GridDistrict.subst_id,
                                       func.ST_AsText(func.ST_Transform(
                                           orm_GridDistrict.geom, srid)).\
                                        label('poly_geom'),
                                       func.ST_AsText(func.ST_Transform(
                                           orm_EgoDeuSubstation.geom, srid)).\
                                       label('subs_geom')).\
            join(orm_EgoDeuSubstation, orm_GridDistrict.subst_id==
                 orm_EgoDeuSubstation.id).\
            filter(orm_GridDistrict.subst_id.in_(mv_grid_districts))

        # read data from db
        mv_data = pd.read_sql_query(grid_districts.statement,
                                    session.bind,
                                    index_col='subst_id')


        # iterate over grid_district/station datasets and initiate objects
        try:
            for poly_id, row in mv_data.iterrows():
                subst_id = poly_id
                region_geo_data = wkt_loads(row['poly_geom'])

                # transform `region_geo_data` to epsg 3035
                # to achieve correct area calculation of mv_grid_district
                station_geo_data = wkt_loads(row['subs_geom'])
                projection = partial(
                    pyproj.transform,
                    pyproj.Proj(init='epsg:4326'),  # source coordinate system
                    pyproj.Proj(init='epsg:3035'))  # destination coordinate system

                region_geo_data = transform(projection, region_geo_data)

                mv_grid_district = self.build_mv_grid_district(poly_id,
                                                 subst_id,
                                                 region_geo_data,
                                                 station_geo_data)
                self.import_lv_load_areas(conn, mv_grid_district)

                # add sum of peak loads of underlying lv grid_districts to mv_grid_district
                mv_grid_district.add_peak_demand()
        except:
            raise ValueError('unexpected error while initiating MV grid_districts' \
                             'from DB dataset.')

    def import_lv_load_areas(self, conn, mv_grid_district):
        """imports load_areas (load areas) from database for a single MV grid_district

        Table definition for load areas can be found here:
        http://vernetzen.uni-flensburg.de/redmine/projects/open_ego/wiki/
        Methoden_AP_26_DataProc

        Parameters
        ----------
        conn: Database connection
        mv_grid_district : MV grid_district/station (instance of MVGridDistrictDingo class) for
            which the import of load areas is performed
        """

        srid = str(int(cfg_dingo.get('geo', 'srid')))

        # threshold: load area peak load, if peak load < threshold => disregard
        # load area
        lv_loads_threshold = cfg_dingo.get('mv_routing', 'load_area_threshold')


        load_scaling_factor = 10**6  # load in database is in GW -> scale to kW

        # build SQL query
        Session = sessionmaker(bind=conn)
        session = Session()

        lv_load_areas_sqla = session.query(
            orm_EgoDeuLoadArea.id.label('id_db'),
            orm_EgoDeuLoadArea.zensus_sum,
            orm_EgoDeuLoadArea.zensus_count.label('zensus_cnt'),
            orm_EgoDeuLoadArea.ioer_sum,
            orm_EgoDeuLoadArea.ioer_count.label('ioer_cnt'),
            orm_EgoDeuLoadArea.area_ha.label('area'),
            orm_EgoDeuLoadArea.sector_area_residential,
            orm_EgoDeuLoadArea.sector_area_retail,
            orm_EgoDeuLoadArea.sector_area_industrial,
            orm_EgoDeuLoadArea.sector_area_agricultural,
            orm_EgoDeuLoadArea.sector_share_residential,
            orm_EgoDeuLoadArea.sector_share_retail,
            orm_EgoDeuLoadArea.sector_share_industrial,
            orm_EgoDeuLoadArea.sector_share_agricultural,
            orm_EgoDeuLoadArea.sector_count_residential,
            orm_EgoDeuLoadArea.sector_count_retail,
            orm_EgoDeuLoadArea.sector_count_industrial,
            orm_EgoDeuLoadArea.sector_count_agricultural,
            orm_EgoDeuLoadArea.nuts.label('nuts_code'),
            func.ST_AsText(func.ST_Transform(orm_EgoDeuLoadArea.geom, srid)).\
                label('geo_area'),
            func.ST_AsText(func.ST_Transform(orm_EgoDeuLoadArea.geom_centre, srid)).\
                label('geo_centre'),
            func.round(orm_CalcEgoPeakLoad.residential * load_scaling_factor).\
                label('peak_load_residential'),
            func.round(orm_CalcEgoPeakLoad.retail * load_scaling_factor).\
                label('peak_load_retail'),
            func.round(orm_CalcEgoPeakLoad.industrial * load_scaling_factor).\
                label('peak_load_industrial'),
            func.round(orm_CalcEgoPeakLoad.agricultural * load_scaling_factor).\
                label('peak_load_agricultural'),
            func.round((orm_CalcEgoPeakLoad.residential
                        + orm_CalcEgoPeakLoad.retail
                        + orm_CalcEgoPeakLoad.industrial
                        + orm_CalcEgoPeakLoad.agricultural)
                       * load_scaling_factor).label('peak_load_sum')). \
            join(orm_CalcEgoPeakLoad, orm_EgoDeuLoadArea.id
                 == orm_CalcEgoPeakLoad.id).\
            filter(orm_EgoDeuLoadArea.subst_id == mv_grid_district.\
                   mv_grid._station.id_db)

        # read data from db
        lv_load_areas = pd.read_sql_query(lv_load_areas_sqla.statement,
                                       session.bind,
                                       index_col='id_db')

        # create load_area objects from rows and add them to graph
        for id_db, row in lv_load_areas.iterrows():

            # only pick load areas with peak load greater than
            # lv_loads_threshold
            # TODO: When migrating to SQLAlchemy, move condition to query
            if row['peak_load_sum'] >= lv_loads_threshold:
                # create LV load_area object
                lv_load_area = LVLoadAreaDingo(id_db=id_db,
                                               db_data=row,
                                               mv_grid_district=mv_grid_district,
                                               peak_load_sum=row['peak_load_sum'])

                # # TODO: Modify following commented-out part when LVGridDistrict (Input Jonas) is implemented
                # # === START TESTING ===
                lv_grid_districts = self.import_lv_grid_districts(conn, lv_load_area)
                lv_stations = self.import_lv_stations(conn, lv_load_area)

                # Associate lv_grid_district to load_area
                for id_grid_district, geom_grid_district in lv_grid_districts.iterrows():

                    # TODO: add geom to lv_grid_district
                    lv_grid_district = LVGridDistrictDingo(
                        id_db=id_grid_district)

                    # be aware, lv_grid takes grid district's geom!
                    lv_grid = LVGridDingo(grid_district=lv_grid_district,
                                          id_db=id_grid_district,
                                          geo_data=lv_grid_district.geom)

                    lv_station = LVStationDingo(
                        id_db=id_grid_district, # is equal to station id
                        grid=lv_grid,
                        geo_data=lv_stations.loc[id_grid_district, 'geom'])

                    lv_grid.add_station(lv_station)

                    lv_grid_district.lv_grid = lv_grid

                    lv_load_area.add_lv_grid_district(lv_grid_district)


                # # create LV station object
                # station_geo_data = wkt_loads(row['geo_centre'])
                # lv_station = LVStationDingo(id_db=id_db,
                #                             geo_data=station_geo_data,
                #                             peak_load=row['peak_load_sum'])
                # lv_grid = LVGridDingo(region=lv_load_area,
                #                       id_db=id_db,
                #                       geo_data=station_geo_data)
                # lv_station.grid = lv_grid
                #
                # # add LV station to LV grid
                # lv_grid.add_station(lv_station)
                #
                # # add LV grid to LV load_area
                # lv_load_area.add_lv_grid(lv_grid)
                # # === END TESTING ===

                centre_geo_data = wkt_loads(row['geo_centre'])

                # create new centre object for LV load area
                lv_load_area_centre = LVLoadAreaCentreDingo(id_db=id_db,
                                                            geo_data=centre_geo_data,
                                                            lv_load_area=lv_load_area)
                # links the centre object to LV load area
                lv_load_area.lv_load_area_centre = lv_load_area_centre

                # add LV load area to MV grid district (and add centre object to MV gris district's graph)
                mv_grid_district.add_lv_load_area(lv_load_area)

                # OLD:
                # add LV load_area to MV grid graph
                # TODO: add LV station instead of LV load_area
                #mv_grid_district.mv_grid.graph_add_node(lv_load_area)

    def import_lv_grid_districts(self, conn, lv_load_area):
        """Imports all lv grid districts within given load area

        Parameters
        ----------
        conn: SQLalchemy database connection
        lv_load_area: LVLoadAreaDingo instance
            Load area for which LV grid districts should be imported

        Returns
        -------
        lv_grid_districts: pandas Dataframe
            Table of lv_grid_districts
        """

        # TODO: build sql query

        # 1. filter grid districts of relevant load area
        Session = sessionmaker(bind=conn)
        session = Session()

        # TODO: select grid districts instead of load areas
        lv_grid_districs_sqla = session.query(orm_EgoDeuOnts.load_area_id,
                                              orm_EgoDeuOnts.geom). \
            filter(orm_EgoDeuOnts.load_area_id == lv_load_area.id_db)


        # read data from db
        lv_grid_districs = pd.read_sql_query(lv_grid_districs_sqla.statement,
                                             session.bind,
                                             index_col='load_area_id')

        return lv_grid_districs


    def import_lv_stations(self, conn, lv_load_area):
        """
        Import lv_stations within the given load_area
        Parameters
        ----------
        conn: SQLalchemy database connection
        lv_load_area: LVLoadAreaDingo instance
            Load area for which LV grid districts should be imported

        Returns
        -------
        lv_stations: pandas Dataframe
            Table of lv_stations
        """
        Session = sessionmaker(bind=conn)
        session = Session()

        lv_stations_sqla = session.query(orm_EgoDeuOnts.load_area_id,
                                              orm_EgoDeuOnts.geom). \
            filter(orm_EgoDeuOnts.load_area_id == lv_load_area.id_db)

        # read data from db
        lv_grid_stations = pd.read_sql_query(lv_stations_sqla.statement,
                                             session.bind,
                                             index_col='load_area_id')
        return lv_grid_stations


    def export_mv_grid(self, conn, mv_grid_districts):
        """ Exports MV grids to database for visualization purposes

        Parameters
        ----------
        conn : sqlalchemy.engine.base.Connection object
               Database connection
        mv_grid_districts : List of MV grid_districts (instances of MVGridDistrictDingo class)
            whose MV grids are exported.

        """
        # TODO: currently only station- & line-positions are exported
        # TODO: (no further electric data)

        # TODO: method has to be extended to cover more data

        # check arguments
        if not all(isinstance(_, int) for _ in mv_grid_districts):
            raise TypeError('`mv_grid_districts` has to be a list of integers.')

        srid = str(int(cfg_dingo.get('geo', 'srid')))

        Session = sessionmaker(bind=conn)
        session = Session()

        # delete all existing datasets
        session.query(db_int.sqla_mv_grid_viz).delete()
        session.commit()

        # build data array from grids (nodes and branches)
        for grid_district in self.mv_grid_districts():
            grid_id = grid_district.mv_grid.id_db
            mv_stations = []
            mv_cable_distributors = []
            lv_stations = []
            lines = []

            for node in grid_district.mv_grid._graph.nodes():
                if isinstance(node, LVLoadAreaCentreDingo):
                    lv_stations.append((node.geo_data.x, node.geo_data.y))
                elif isinstance(node, CableDistributorDingo):
                    mv_cable_distributors.append((node.geo_data.x,
                                                  node.geo_data.y))
                elif isinstance(node, MVStationDingo):
                    mv_stations.append((node.geo_data.x, node.geo_data.y))

            # create shapely obj from stations and convert to
            # geoalchemy2.types.WKBElement
            lv_stations_wkb = from_shape(MultiPoint(lv_stations), srid=srid)
            mv_cable_distributors_wkb = from_shape(
                MultiPoint(mv_cable_distributors), srid=srid)
            mv_stations_wkb = from_shape(Point(mv_stations), srid=srid)

            for branch in grid_district.mv_grid.graph_edges():
                line = branch['adj_nodes']
                lines.append(((line[0].geo_data.x,
                               line[0].geo_data.y),
                              (line[1].geo_data.x,
                               line[1].geo_data.y)))

            # create shapely obj from lines and convert to
            # geoalchemy2.types.WKBElement
            mv_lines_wkb = from_shape(MultiLineString(lines), srid=srid)

            # add dataset to session
            dataset = db_int.sqla_mv_grid_viz(
                grid_id=grid_id,
                timestamp=datetime.now(),
                geom_mv_station=mv_stations_wkb,
                geom_mv_cable_dist=mv_cable_distributors_wkb ,
                geom_lv_stations=lv_stations_wkb,
                geom_mv_lines=mv_lines_wkb)
            session.add(dataset)

        # commit changes to db
        session.commit()


    def mv_routing(self, debug=False, animation=False):
        """ Performs routing on all MV grids, see method `routing` in class
        `MVGridDingo` for details.

        Parameters
        ----------
            debug: If True, information is printed while routing
            animation: If True, images of route modification steps are exported
                during routing process - a new animation
                object is created, refer to class 'AnimationDingo()' for a more
                detailed description.
        """

        if animation:
            anim = AnimationDingo()
        else:
            anim = None

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.routing(debug, anim)

    def mv_parametrize_grid(self, debug=False):
        """ Performs Parametrization of grid equipment of all MV grids, see
        method `parametrize_grid` in class
        `MVGridDingo` for details.

        Parameters
        ----------
        debug: If True, information is printed while parametrization
        """

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.parametrize_grid(debug)

    def __repr__(self):
        return str(self.name)
