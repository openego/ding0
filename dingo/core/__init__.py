import dingo
from dingo.config import config_db_interfaces as db_int
from dingo.core.network import GeneratorDingo
from dingo.core.network.cable_distributors import MVCableDistributorDingo
from dingo.core.network.grids import *
from dingo.core.network.stations import *
from dingo.core.structure.regions import *
from dingo.tools import config as cfg_dingo
from dingo.tools.animation import AnimationDingo

# import ORM classes for oedb access depending on input in config file
cfg_dingo.load_config('config_db_tables')
GridDistrict_name = cfg_dingo.get('regions', 'grid_district')
EgoDeuSubstation_name = cfg_dingo.get('stations', 'mv_stations')
EgoDeuLoadArea_name = cfg_dingo.get('regions', 'lv_load_areas')
CalcEgoPeakLoad_name = cfg_dingo.get('loads', 'lv_loads')
EgoDeuOnts_name = cfg_dingo.get('stations', 'lv_stations')
LVGridDistrict_name = cfg_dingo.get('regions', 'lv_grid_district')
EgoDeuDea_name = cfg_dingo.get('generators', 're_generators')

from egoio.db_tables import calc_ego_substation as orm_mod_calc_ego_substation
from egoio.db_tables import calc_ego_grid_district as orm_calc_ego_grid_district
from egoio.db_tables import calc_ego_loads as orm_calc_ego_loads
from egoio.db_tables import calc_ego_re as orm_calc_ego_re

orm_EgoDeuSubstation = orm_mod_calc_ego_substation.\
    __getattribute__(EgoDeuSubstation_name)
orm_GridDistrict = orm_calc_ego_grid_district.\
    __getattribute__(GridDistrict_name)
orm_LVGridDistrict = orm_calc_ego_grid_district.\
    __getattribute__(LVGridDistrict_name)
orm_EgoDeuLoadArea = orm_calc_ego_loads.__getattribute__(EgoDeuLoadArea_name)
orm_CalcEgoPeakLoad = orm_calc_ego_loads.__getattribute__(CalcEgoPeakLoad_name)
orm_EgoDeuOnts = orm_mod_calc_ego_substation.__getattribute__(EgoDeuOnts_name)
orm_EgoDeuDea = orm_calc_ego_re.__getattribute__(EgoDeuDea_name)

import pandas as pd

from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, or_
from geoalchemy2.shape import from_shape
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Point, MultiPoint, MultiLineString

from functools import partial
import pyproj
from shapely.ops import transform


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

    def get_mvgd_lvla_obj_from_id(self):
        """ Build dict with mapping from LVLoadAreaDingo id to LVLoadAreaDingo object and
                                         MVGridDistrictDingo id to MVGridDistrictDingo object

        Returns:
            lv_load_areas:      dict with Format {lv_load_area_id_1: lv_load_area_obj_1,
                                                  ...,
                                                  lv_load_area_id_n: lv_load_area_obj_n}
            mv_grid_districts:  dict with Format {mv_grid_district_id_1: mv_grid_district_obj_1,
                                                  ...,
                                                  mv_grid_district_id_n: mv_grid_district_obj_n}
        """

        mv_grid_districts = {}
        lv_load_areas = {}

        for mv_grid_district in self.mv_grid_districts():
            mv_grid_districts[mv_grid_district.id_db] = mv_grid_district
            for lv_load_area in mv_grid_district.lv_load_areas():
                lv_load_areas[lv_load_area.id_db] = lv_load_area

        return lv_load_areas, mv_grid_districts

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

    def build_lv_grid_district(self, conn, lv_load_area, string_properties,
                               apartment_string, apartment_trafo,
                               lv_grid_districts, lv_stations):
        """
        Instantiates and associates lv_grid_district incl grid and station

        Parameters
        ----------
        conn: SQLalchemy database connection object
        lv_load_area: load_area object
        string_properties: DataFrame
            Properties of LV typified model grids
        apartment_string: DataFrame
            Relational table of apartment count and strings of model grid
        apartment_trafo: DataFrame
            Relational table of apartment count and trafo size
        lv_grid_districts: DataFrame
            Table containing lv_grid_districts of according load_area
        lv_stations : DataFrame
            Table containing lv_stations of according load_area
        """

        # Associate lv_grid_district to load_area
        for id, row in lv_grid_districts.iterrows():
            lv_grid_district = LVGridDistrictDingo(
                id_db=id,
                lv_load_area=lv_load_area,
                geo_data=wkt_loads(row['geom']),
                population=row['population'])

            # be aware, lv_grid takes grid district's geom!
            lv_grid = LVGridDingo(grid_district=lv_grid_district,
                                  id_db=id,
                                  geo_data=wkt_loads(row['geom']))

            lv_station = LVStationDingo(
                id_db=id,  # is equal to station id
                grid=lv_grid,
                lv_load_area=lv_load_area,
                geo_data=wkt_loads(lv_stations.loc[id, 'geom']))

            model_grid, transformer = lv_grid.select_typified_grid_model(
                string_properties,
                apartment_string,
                apartment_trafo,
                lv_grid_district.population)

            lv_transformer = TransformerDingo(equip_trans_id=id,
                                              v_level=0.4,
                                              s_max_longterm=transformer)

            lv_station.add_transformer(lv_transformer)

            lv_grid.add_station(lv_station)

            lv_grid.build_lv_graph(model_grid)

            lv_grid_district.lv_grid = lv_grid

            lv_load_area.add_lv_grid_district(lv_grid_district)

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
        # TODO: Complete "See Also"-List

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
                                           orm_EgoDeuSubstation.point, srid)).\
                                       label('subs_geom')).\
            join(orm_EgoDeuSubstation, orm_GridDistrict.subst_id ==
                 orm_EgoDeuSubstation.id).\
            filter(orm_GridDistrict.subst_id.in_(mv_grid_districts))

        # read MV data from db
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

                # import all lv_grid_districts wihtin mv_grid_district
                lv_grid_districts = self.import_lv_grid_districts(conn)

                # import all lv_stations wihtin mv_grid_district
                lv_stations = self.import_lv_stations(conn)

                self.import_lv_load_areas(conn,
                                          mv_grid_district,
                                          lv_grid_districts,
                                          lv_stations)

                # add sum of peak loads of underlying lv grid_districts to mv_grid_district
                mv_grid_district.add_peak_demand()
        except:
            raise ValueError('unexpected error while initiating MV grid_districts' \
                             'from DB dataset.')

    def import_lv_load_areas(self, conn, mv_grid_district, lv_grid_districts,
                             lv_stations):
        """imports load_areas (load areas) from database for a single MV grid_district

        Table definition for load areas can be found here:
        http://vernetzen.uni-flensburg.de/redmine/projects/open_ego/wiki/
        Methoden_AP_26_DataProc

        Parameters
        ----------
        conn: Database connection
        mv_grid_district : MV grid_district/station (instance of MVGridDistrictDingo class) for
            which the import of load areas is performed
        lv_grid_districts: DataFrame
            LV grid districts within this mv_grid_district
        lv_stations: DataFrame
            LV stations within this mv_grid_district
        """

        # get dingos' standard CRS (SRID)
        srid = str(int(cfg_dingo.get('geo', 'srid')))

        # threshold: load area peak load, if peak load < threshold => disregard
        # load area
        lv_loads_threshold = cfg_dingo.get('mv_routing', 'load_area_threshold')


        load_scaling_factor = 10**6  # load in database is in GW -> scale to kW

        # build SQL query
        Session = sessionmaker(bind=conn)
        session = Session()

        # TODO: move to a more sophisticated location
        import pandas as pd
        import os

        global lv_cable_parameters
        lv_cable_parameters = pd.read_csv(os.path.join(
            'dingo',
            'data',
            'equipment-parameters_LV_cables.csv'), index_col='name')

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

        # read LV model grid data from CSV file
        string_properties, apartment_string, apartment_trafo = self.import_lv_model_grids()

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

                # sub-selection of lv_grid_districts/lv_stations within one
                # specific load area
                lv_grid_districts_per_load_area = lv_grid_districts.\
                    loc[lv_grid_districts['load_area_id'] == id_db]
                lv_stations_per_load_area = lv_stations.\
                    loc[lv_stations['load_area_id'] == id_db]

                # # ===== DEBUG STUFF (BUG JONAS) =====
                # TODO: Remove when fixed!
                if len(lv_grid_districts_per_load_area) == 0:
                    print('No LV grid district for', lv_load_area, 'found! (blame Jonas)')
                if len(lv_stations_per_load_area) == 0:
                    print('No station for', lv_load_area, 'found! (blame Jonas)')
                # ===================================

                self.build_lv_grid_district(conn,
                                            lv_load_area,
                                            string_properties,
                                            apartment_string,
                                            apartment_trafo,
                                            lv_grid_districts_per_load_area,
                                            lv_stations_per_load_area)

                # create new centre object for LV load area
                lv_load_area_centre = LVLoadAreaCentreDingo(id_db=id_db,
                                                            geo_data=wkt_loads(row['geo_centre']),
                                                            lv_load_area=lv_load_area)
                # links the centre object to LV load area
                lv_load_area.lv_load_area_centre = lv_load_area_centre

                # add LV load area to MV grid district (and add centre object to MV gris district's graph)
                mv_grid_district.add_lv_load_area(lv_load_area)

                # OLD:
                # add LV load_area to MV grid graph
                # TODO: add LV station instead of LV load_area
                #mv_grid_district.mv_grid.graph_add_node(lv_load_area)

    def import_lv_grid_districts(self, conn):
        """Imports all lv grid districts within given load area

        Parameters
        ----------
        conn: SQLalchemy database connection

        Returns
        -------
        lv_grid_districts: pandas Dataframe
            Table of lv_grid_districts
        """

        # get dingos' standard CRS (SRID)
        srid = str(int(cfg_dingo.get('geo', 'srid')))

        # 1. filter grid districts of relevant load area
        Session = sessionmaker(bind=conn)
        session = Session()

        lv_grid_districs_sqla = session.query(orm_LVGridDistrict.load_area_id,
                                              func.ST_AsText(func.ST_Transform(
                                                orm_LVGridDistrict.geom, srid)).label('geom'),
                                              orm_LVGridDistrict.population,
                                              orm_LVGridDistrict.id)

        # read data from db
        lv_grid_districs = pd.read_sql_query(lv_grid_districs_sqla.statement,
                                             session.bind,
                                             index_col='id')

        return lv_grid_districs

    def import_lv_stations(self, conn):
        """
        Import lv_stations within the given load_area
        Parameters
        ----------
        conn: SQLalchemy database connection

        Returns
        -------
        lv_stations: pandas Dataframe
            Table of lv_stations
        """

        # get dingos' standard CRS (SRID)
        srid = str(int(cfg_dingo.get('geo', 'srid')))

        Session = sessionmaker(bind=conn)
        session = Session()

        lv_stations_sqla = session.query(orm_EgoDeuOnts.id,
                                         orm_EgoDeuOnts.load_area_id,
                                         func.ST_AsText(func.ST_Transform(
                                           orm_EgoDeuOnts.geom, srid)).label('geom'))

        # read data from db
        lv_grid_stations = pd.read_sql_query(lv_stations_sqla.statement,
                                             session.bind,
                                             index_col='id')
        return lv_grid_stations

    def import_lv_model_grids(self):
        """
        Import typified model grids

        Returns
        -------
        string_properties: Pandas Dataframe
            Table describing each string
        apartment_string: Pandas Dataframe
            Relational table of apartments and strings
        apartment_trafo: Pandas Dataframe
            Relational table assigning trafos to apartments
        """

        string_properties_file = cfg_dingo.get("model_grids",
                                               "string_properties")
        apartment_string_file = cfg_dingo.get("model_grids",
                                               "apartment_string")
        apartment_trafo_file = cfg_dingo.get("model_grids",
                                               "apartment_trafo")

        package_path = dingo.__path__[0]

        string_properties = pd.read_csv(os.path.join(
            package_path, 'data', string_properties_file),
            comment='#', delimiter=';', decimal=',',
            index_col='string_id')
        apartment_string = pd.read_csv(os.path.join(
            package_path, 'data', apartment_string_file),
            comment='#', delimiter=';', decimal=',',
            index_col='apartment_count')
        apartment_trafo = pd.read_csv(os.path.join(
            package_path, 'data', apartment_trafo_file),
            comment='#', delimiter=';', decimal=',',
            index_col='apartment_count')

        return string_properties, apartment_string, apartment_trafo

    def import_generators(self, conn, mv_grid_districts, debug=False):
        """ Imports generators

        """
        # TODO: Currently only the renewable power plants are imported here, no conventional ones! (has to be done)

        # get dingos' standard CRS (SRID)
        srid = str(int(cfg_dingo.get('geo', 'srid')))

        Session = sessionmaker(bind=conn)
        session = Session()

        for mv_grid_district in mv_grid_districts:

            generators_sqla = session.query(orm_EgoDeuDea.id,
                                            orm_EgoDeuDea.subst_id,
                                            orm_EgoDeuDea.la_id,
                                            orm_EgoDeuDea.electrical_capacity,
                                            orm_EgoDeuDea.generation_type,
                                            orm_EgoDeuDea.generation_subtype,
                                            orm_EgoDeuDea.voltage_level,
                                             func.ST_AsText(func.ST_Transform(
                                            orm_EgoDeuDea.geom_new, srid)).label('geom')).\
                                            filter(orm_EgoDeuDea.subst_id == mv_grid_district).\
                                            filter(or_(orm_EgoDeuDea.voltage_level == '05 (MS)',
                                                       orm_EgoDeuDea.voltage_level == '04 (HS/MS)'))

            # read data from db
            generators = pd.read_sql_query(generators_sqla.statement,
                                           session.bind,
                                           index_col='id')

            #print(generators)

            for id_db, row in generators.iterrows():
                generator = GeneratorDingo(id_db=id_db,
                                           mv_grid=XXX)
                if row['voltage_level'] == '05 (MS)':
                    # find co


    def export_mv_grid(self, conn, mv_grid_districts):
        """ Exports MV grids to database for visualization purposes

        Parameters
        ----------
        conn : sqlalchemy.engine.base.Connection object
               Database connection
        mv_grid_districts : List of MV grid_districts (instances of MVGridDistrictDingo class)
            whose MV grids are exported.

        """
        # TODO: currently only station- & line-positions are exported (no further electric data)
        # TODO: method has to be extended to cover more data and to export different object types to different tables
        # TODO: (make all attributes visible in GIS :)

        # check arguments
        if not all(isinstance(_, int) for _ in mv_grid_districts):
            raise TypeError('`mv_grid_districts` has to be a list of integers.')

        srid = str(int(cfg_dingo.get('geo', 'srid')))

        Session = sessionmaker(bind=conn)
        session = Session()

        # delete all existing datasets
        session.query(db_int.sqla_mv_grid_viz).delete()
        session.commit()

        # build data array from MV grids (nodes and branches)
        for grid_district in self.mv_grid_districts():

            grid_id = grid_district.mv_grid.id_db

            # init arrays for nodes
            mv_stations = []
            mv_cable_distributors = []
            mv_circuit_breakers = []
            lv_load_area_centres = []
            lv_stations = []
            lines = []

            # get nodes from grid's graph and append to corresponding array
            for node in grid_district.mv_grid._graph.nodes():
                if isinstance(node, LVLoadAreaCentreDingo):
                    lv_load_area_centres.append((node.geo_data.x, node.geo_data.y))
                elif isinstance(node, MVCableDistributorDingo):
                    mv_cable_distributors.append((node.geo_data.x, node.geo_data.y))
                elif isinstance(node, MVStationDingo):
                    mv_stations.append((node.geo_data.x, node.geo_data.y))
                elif isinstance(node, CircuitBreakerDingo):
                    mv_circuit_breakers.append((node.geo_data.x, node.geo_data.y))

            # create shapely obj from stations and convert to
            # geoalchemy2.types.WKBElement
            # set to None if no objects found (otherwise SQLAlchemy will throw an error).
            if lv_load_area_centres:
                lv_load_area_centres_wkb = from_shape(MultiPoint(lv_load_area_centres), srid=srid)
            else:
                lv_load_area_centres_wkb = None

            if mv_cable_distributors:
                mv_cable_distributors_wkb = from_shape(MultiPoint(mv_cable_distributors), srid=srid)
            else:
                mv_cable_distributors_wkb = None

            if mv_circuit_breakers:
                mv_circuit_breakers_wkb = from_shape(MultiPoint(mv_circuit_breakers), srid=srid)
            else:
                mv_circuit_breakers_wkb = None

            if mv_stations:
                mv_stations_wkb = from_shape(Point(mv_stations), srid=srid)
            else:
                mv_stations_wkb = None

            # get edges (lines) from grid's graph and append to corresponding array
            for branch in grid_district.mv_grid.graph_edges():
                line = branch['adj_nodes']
                lines.append(((line[0].geo_data.x,
                               line[0].geo_data.y),
                              (line[1].geo_data.x,
                               line[1].geo_data.y)))

            # create shapely obj from lines and convert to
            # geoalchemy2.types.WKBElement
            mv_lines_wkb = from_shape(MultiLineString(lines), srid=srid)

            # get nodes from lv grid districts and append to corresponding array
            for lv_load_area in grid_district.lv_load_areas():
                for lv_grid_district in lv_load_area.lv_grid_districts():
                    station = lv_grid_district.lv_grid.station()
                    lv_stations.append((station.geo_data.x, station.geo_data.y))
            lv_stations_wkb = from_shape(MultiPoint(lv_stations), srid=srid)

            # add dataset to session
            dataset = db_int.sqla_mv_grid_viz(
                grid_id=grid_id,
                geom_mv_station=mv_stations_wkb,
                geom_mv_cable_dists=mv_cable_distributors_wkb,
                geom_mv_circuit_breakers=mv_circuit_breakers_wkb,
                geom_lv_load_area_centres=lv_load_area_centres_wkb,
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
            method `parametrize_grid()` in class `MVGridDingo` for details.

        Parameters
        ----------
        debug: If True, information is printed while parametrization
        """

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.parametrize_grid(debug)

    def set_branch_ids(self):
        """ Performs generation and setting of ids of branches for all MV and underlying LV grids, see
            method `set_branch_ids()` in class `MVGridDingo` for details.
        """

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.set_branch_ids()

    def __repr__(self):
        return str(self.name)
