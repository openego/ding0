import dingo
from dingo.config import config_db_interfaces as db_int
from dingo.core.network import GeneratorDingo
from dingo.core.network.cable_distributors import MVCableDistributorDingo
from dingo.core.network.grids import *
from dingo.core.network.stations import *
from dingo.core.structure.regions import *
from dingo.core.powerflow import *
from dingo.tools import pypsa_io, config as cfg_dingo
from dingo.tools import config as cfg_dingo
from dingo.tools.animation import AnimationDingo
from dingo.flexopt.reinforce_grid import *

import os

# import ORM classes for oedb access depending on input in config file
cfg_dingo.load_config('config_db_tables.cfg')
mv_grid_districts_name = cfg_dingo.get('regions', 'mv_grid_districts')
mv_stations_name = cfg_dingo.get('stations', 'mv_stations')
lv_load_areas_name = cfg_dingo.get('regions', 'lv_load_areas')
lv_grid_district_name = cfg_dingo.get('regions', 'lv_grid_district')
lv_loads_name = cfg_dingo.get('loads', 'lv_loads')
lv_stations_name = cfg_dingo.get('stations', 'lv_stations')
conv_generators_name = cfg_dingo.get('generators', 'conv_generators')
re_generators_name = cfg_dingo.get('generators', 're_generators')

from egoio.db_tables import model_draft as orm_model_draft, supply as orm_supply
# from egoio.db_tables import calc_ego_grid_district as orm_calc_ego_grid_district
# from egoio.db_tables import calc_ego_loads as orm_calc_ego_loads
# from egoio.db_tables import calc_ego_re as orm_calc_ego_re
# from egoio.db_tables import supply as orm_supply

orm_mv_grid_districts = orm_model_draft.__getattribute__(mv_grid_districts_name)
orm_mv_stations = orm_model_draft.__getattribute__(mv_stations_name)
orm_lv_load_areas = orm_model_draft.__getattribute__(lv_load_areas_name)
orm_lv_grid_district = orm_model_draft.__getattribute__(lv_grid_district_name)
orm_lv_loads = orm_model_draft.__getattribute__(lv_loads_name)
orm_lv_stations = orm_model_draft.__getattribute__(lv_stations_name)
orm_conv_generators = orm_model_draft.__getattribute__(conv_generators_name)
orm_re_generators = orm_model_draft.__getattribute__(re_generators_name)

import pandas as pd

from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, or_
from geoalchemy2.shape import from_shape
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Point, MultiPoint, MultiLineString, LineString

from functools import partial
import pyproj
from shapely.ops import transform
from math import isnan
import random

package_path = dingo.__path__[0]


class NetworkDingo:
    """ Defines the DINGO Network - not a real grid but a container for the
    MV-grids. Contains the NetworkX graph and associated attributes.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self._mv_grid_districts = []
        self._pf_config = kwargs.get('pf_config', None)
        self._static_data = kwargs.get('static_data', {})

        self.import_pf_config()
        self.import_static_data()

    def mv_grid_districts(self):
        """Returns a generator for iterating over MV grid_districts"""
        for grid_district in self._mv_grid_districts:
            yield grid_district

    def add_mv_grid_district(self, mv_grid_district):
        """Adds a MV grid_district to _mv_grid_districts if not already existing"""
        # TODO: use setter method here (make attribute '_mv_grid_districts' private)
        if mv_grid_district not in self.mv_grid_districts():
            self._mv_grid_districts.append(mv_grid_district)

    @property
    def pf_config(self):
        """Returns PF config object"""
        return self._pf_config

    @property
    def static_data(self):
        """Returns static data"""
        return self._static_data

    def get_mvgd_lvla_lvgd_obj_from_id(self):
        """ Build dict with mapping from LVLoadAreaDingo id to LVLoadAreaDingo object and
                                         MVGridDistrictDingo id to MVGridDistrictDingo object

        Returns:
            mv_grid_districts:  dict with Format {mv_grid_district_id_1: mv_grid_district_obj_1,
                                                  ...,
                                                  mv_grid_district_id_n: mv_grid_district_obj_n}
            lv_load_areas:      dict with Format {lv_load_area_id_1: lv_load_area_obj_1,
                                                  ...,
                                                  lv_load_area_id_n: lv_load_area_obj_n}
        """

        mv_grid_districts_dict = {}
        lv_load_areas_dict = {}
        lv_grid_districts_dict = {}

        for mv_grid_district in self.mv_grid_districts():
            mv_grid_districts_dict[mv_grid_district.id_db] = mv_grid_district
            for lv_load_area in mv_grid_district.lv_load_areas():
                lv_load_areas_dict[lv_load_area.id_db] = lv_load_area
                for lv_grid_district in lv_load_area.lv_grid_districts():
                    lv_grid_districts_dict[lv_grid_district.id_db] = lv_grid_district

        return mv_grid_districts_dict, lv_load_areas_dict, lv_grid_districts_dict

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

        mv_station = MVStationDingo(id_db=subst_id, geo_data=station_geo_data)

        mv_grid = MVGridDingo(network=self,
                              id_db=poly_id,
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
                               trafo_parameters,
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
        trafo_parameters: DataFrame
            Equipment parameters of LV transformers
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
                population=row['population'],
                # TODO: current state: use LVGD count to calc peak load of station -> equal distribution (temporarily)
                # TODO: use area share (based on total area of LA)
                peak_load=lv_load_area.peak_load_sum / len(lv_grid_districts))

            # be aware, lv_grid takes grid district's geom!
            lv_grid = LVGridDingo(network=self,
                                  grid_district=lv_grid_district,
                                  id_db=id,
                                  geo_data=wkt_loads(row['geom']))

            # TODO: assign "real" peak_load value to lv_station when available
            lv_station = LVStationDingo(
                id_db=row['mvlv_subst_id'],
                grid=lv_grid,
                lv_load_area=lv_load_area,
                geo_data=wkt_loads(lv_stations.loc[row['mvlv_subst_id'], 'geom']),
                #peak_load=lv_load_area.peak_load_sum / lv_grid_districts.size)
                # TODO: current state: use LVGD count to calc peak load of station -> equal distribution (temporarily)
                # TODO: use area share (based on total area of LA)
                peak_load=lv_load_area.peak_load_sum / len(lv_grid_districts))

            # Choice of typified lv model grid depends on population within lv
            # grid district. If no population is given, lv grid is omitted and
            # load is represented by lv station's peak load
            if lv_grid_district.population > 0:

                model_grid, transformer = lv_grid.select_typified_grid_model(
                    string_properties,
                    apartment_string,
                    apartment_trafo,
                    trafo_parameters,
                    lv_grid_district.population)

                lv_transformer = TransformerDingo(
                    equip_trans_id=id,
                    v_level=0.4,
                    s_max_longterm=transformer['trafo_apparent_power'],
                    r=transformer['r'],
                    x=transformer['x'])

                lv_station.add_transformer(lv_transformer)


                lv_grid.build_lv_graph(model_grid)

            lv_grid.add_station(lv_station)
            lv_grid_district.lv_grid = lv_grid

            lv_load_area.add_lv_grid_district(lv_grid_district)

    def import_mv_grid_districts(self, conn, mv_grid_districts_no=None):
        """Imports MV grid_districts and MV stations from database, reprojects geodata
        and and initiates objects.

        Parameters
        ----------
        conn : sqlalchemy.engine.base.Connection object
               Database connection
        mv_grid_districts : List of MV grid_districts/stations (int) to be imported (if empty,
            all grid_districts & stations are imported)

        See Also
        --------
        build_mv_grid_district : used to instantiate MV grid_district objects
        import_lv_load_areas : used to import load_areas for every single MV grid_district
        add_peak_demand : used to summarize peak loads of underlying load_areas
        """

        # check arguments
        if not all(isinstance(_, int) for _ in mv_grid_districts_no):
            raise TypeError('`mv_grid_districts` has to be a list of integers.')

        # get srid settings from config
        try:
            srid = str(int(cfg_dingo.get('geo', 'srid')))
        except OSError:
            print('cannot open config file.')

        # build SQL query
        Session = sessionmaker(bind=conn)
        session = Session()
        grid_districts = session.query(orm_mv_grid_districts.subst_id,
                                       func.ST_AsText(func.ST_Transform(
                                           orm_mv_grid_districts.geom, srid)). \
                                       label('poly_geom'),
                                       func.ST_AsText(func.ST_Transform(
                                           orm_mv_stations.point, srid)). \
                                       label('subs_geom')).\
            join(orm_mv_stations, orm_mv_grid_districts.subst_id ==
                 orm_mv_stations.subst_id).\
            filter(orm_mv_grid_districts.subst_id.in_(mv_grid_districts_no))

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

                # import all lv_stations within mv_grid_district
                lv_stations = self.import_lv_stations(conn)

                # import all lv_grid_districts within mv_grid_district
                lv_grid_districts = self.import_lv_grid_districts(conn, lv_stations)
                lv_grid_districts['population'] = [random.randrange(1, 450) for x in lv_grid_districts.index.values]

                # lv_grid_districts = pd.DataFrame(
                #     columns=['load_area_id', 'population'],
                #     index=list(lv_stations.index.values))
                # lv_grid_districts['load_area_id'] = lv_stations['load_area_id'].tolist()
                # lv_grid_districts['population'] = [random.randrange(1, 450) for x in lv_grid_districts.index.values]
                # lv_grid_districts['geom'] = lv_stations[
                #     'geom'].tolist()

                self.import_lv_load_areas(conn,
                                          mv_grid_district,
                                          lv_grid_districts,
                                          lv_stations)

                # add sum of peak loads of underlying lv grid_districts to mv_grid_district
                mv_grid_district.add_peak_demand()
        except:
            raise ValueError('unexpected error while initiating MV grid_districts from DB dataset.')

        print('=====> MV Grid Districts imported')

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

        package_path = dingo.__path__[0]

        # get dingos' standard CRS (SRID)
        srid = str(int(cfg_dingo.get('geo', 'srid')))

        # threshold: load area peak load, if peak load < threshold => disregard
        # load area
        lv_loads_threshold = cfg_dingo.get('mv_routing', 'load_area_threshold')

        load_scaling_factor = 10**6  # load in database is in GW -> scale to kW

        # build SQL query
        Session = sessionmaker(bind=conn)
        session = Session()

        cable_parameters_file = cfg_dingo.get("equipment",
                                              "equipment_lv_parameters_cables")

        global lv_cable_parameters
        lv_cable_parameters = pd.read_csv(os.path.join(
            package_path,
            'data',
            cable_parameters_file),
            comment='#',
            index_col='name')

        lv_load_areas_sqla = session.query(
            orm_lv_load_areas.id.label('id_db'),
            orm_lv_load_areas.zensus_sum,
            orm_lv_load_areas.zensus_count.label('zensus_cnt'),
            orm_lv_load_areas.ioer_sum,
            orm_lv_load_areas.ioer_count.label('ioer_cnt'),
            orm_lv_load_areas.area_ha.label('area'),
            orm_lv_load_areas.sector_area_residential,
            orm_lv_load_areas.sector_area_retail,
            orm_lv_load_areas.sector_area_industrial,
            orm_lv_load_areas.sector_area_agricultural,
            orm_lv_load_areas.sector_share_residential,
            orm_lv_load_areas.sector_share_retail,
            orm_lv_load_areas.sector_share_industrial,
            orm_lv_load_areas.sector_share_agricultural,
            orm_lv_load_areas.sector_count_residential,
            orm_lv_load_areas.sector_count_retail,
            orm_lv_load_areas.sector_count_industrial,
            orm_lv_load_areas.sector_count_agricultural,
            orm_lv_load_areas.nuts.label('nuts_code'),
            func.ST_AsText(func.ST_Transform(orm_lv_load_areas.geom, srid)).\
                label('geo_area'),
            func.ST_AsText(func.ST_Transform(orm_lv_load_areas.geom_centre, srid)).\
                label('geo_centre'),
            func.round(orm_lv_loads.residential * load_scaling_factor).\
                label('peak_load_residential'),
            func.round(orm_lv_loads.retail * load_scaling_factor).\
                label('peak_load_retail'),
            func.round(orm_lv_loads.industrial * load_scaling_factor).\
                label('peak_load_industrial'),
            func.round(orm_lv_loads.agricultural * load_scaling_factor).\
                label('peak_load_agricultural'),
            func.round((orm_lv_loads.residential
                        + orm_lv_loads.retail
                        + orm_lv_loads.industrial
                        + orm_lv_loads.agricultural)
                       * load_scaling_factor).label('peak_load_sum')). \
            join(orm_lv_loads, orm_lv_load_areas.id
                 == orm_lv_loads.id).\
            filter(orm_lv_load_areas.subst_id == mv_grid_district. \
                   mv_grid._station.id_db).\
            filter(((orm_lv_loads.residential  # only pick load areas with peak load > lv_loads_threshold
                     + orm_lv_loads.retail
                     + orm_lv_loads.industrial
                     + orm_lv_loads.agricultural)
                       * load_scaling_factor) > lv_loads_threshold)

        # read data from db
        lv_load_areas = pd.read_sql_query(lv_load_areas_sqla.statement,
                                          session.bind,
                                          index_col='id_db')

        # read LV model grid data from CSV file
        string_properties, apartment_string, apartment_trafo, trafo_parameters\
            = self.import_lv_model_grids()

        # create load_area objects from rows and add them to graph
        for id_db, row in lv_load_areas.iterrows():

            # create LV load_area object
            lv_load_area = LVLoadAreaDingo(id_db=id_db,
                                           db_data=row,
                                           mv_grid_district=mv_grid_district,
                                           peak_load_sum=row['peak_load_sum'])

            # sub-selection of lv_grid_districts/lv_stations within one
            # specific load area
            lv_grid_districts_per_load_area = lv_grid_districts.\
                loc[lv_grid_districts['la_id'] == id_db]
            lv_stations_per_load_area = lv_stations.\
                loc[lv_stations['la_id'] == id_db]

            # # ===== DEBUG STUFF (BUG JONAS) =====
            # TODO: Remove when fixed!
            if len(lv_grid_districts_per_load_area) == 0:
                print('No LV grid district for', lv_load_area, 'found! (Bug jong42)')
            if len(lv_stations_per_load_area) == 0:
                print('No station for', lv_load_area, 'found! (Bug jong42)')
            # ===================================

            self.build_lv_grid_district(conn,
                                        lv_load_area,
                                        string_properties,
                                        apartment_string,
                                        apartment_trafo,
                                        trafo_parameters,
                                        lv_grid_districts_per_load_area,
                                        lv_stations_per_load_area)

            # create new centre object for LV load area
            lv_load_area_centre = LVLoadAreaCentreDingo(id_db=id_db,
                                                        geo_data=wkt_loads(row['geo_centre']),
                                                        lv_load_area=lv_load_area,
                                                        grid=mv_grid_district.mv_grid)
            # links the centre object to LV load area
            lv_load_area.lv_load_area_centre = lv_load_area_centre

            # add LV load area to MV grid district (and add centre object to MV gris district's graph)
            mv_grid_district.add_lv_load_area(lv_load_area)

    def import_lv_grid_districts(self, conn, lv_stations):
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

        #load_areas = list(self.get_mvgd_lvla_lvgd_obj_from_id()[1])

        lv_grid_districs_sqla = session.query(orm_lv_grid_district.la_id,
                                              func.ST_AsText(func.ST_Transform(
                                                orm_lv_grid_district.geom, srid)).label('geom'),
                                              orm_lv_grid_district.mvlv_subst_id,
                                              orm_lv_grid_district.id). \
            filter(orm_lv_grid_district.mvlv_subst_id.in_(lv_stations.index.tolist()))
            #filter(orm_lv_grid_district.load_area_id.in_(load_areas))

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

        # get list of mv grid districts
        mv_grid_districts = list(self.get_mvgd_lvla_lvgd_obj_from_id()[0])

        lv_stations_sqla = session.query(orm_lv_stations.mvlv_subst_id,
                                         orm_lv_stations.la_id,
                                         func.ST_AsText(func.ST_Transform(
                                           orm_lv_stations.geom, srid)). \
                                         label('geom')).\
            filter(orm_lv_stations.subst_id.in_(mv_grid_districts))

        # read data from db
        lv_grid_stations = pd.read_sql_query(lv_stations_sqla.statement,
                                             session.bind,
                                             index_col='mvlv_subst_id')
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
                                               "model_grids_lv_string_properties")
        apartment_string_file = cfg_dingo.get("model_grids",
                                               "model_grids_lv_apartment_string")
        apartment_trafo_file = cfg_dingo.get("model_grids",
                                               "model_grids_lv_apartment_trafo")
        trafo_parameters_file = cfg_dingo.get("equipment",
                                              "equipment_lv_parameters_trafos")

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
        trafo_parameters = pd.read_csv(os.path.join(
            package_path, 'data', trafo_parameters_file),
            comment='#', delimiter=',', decimal='.',
            index_col='transformer_size')

        return string_properties, apartment_string, apartment_trafo,\
            trafo_parameters

    def import_generators(self, conn, debug=False):
        """
        Imports renewable (res) and conventional (conv) generators

        Args:
            conn: SQLalchemy database connection
            debug: If True, information is printed during process
        Notes:
            Connection of generators is done later on in NetworkDingo's method connect_generators()
        """

        def import_res_generators():
            """Imports renewable (res) generators"""

            # build query
            generators_sqla = session.query(orm_re_generators.id,
                                            orm_re_generators.subst_id,
                                            orm_re_generators.la_id,
                                            orm_re_generators.mvlv_subst_id,
                                            orm_re_generators.electrical_capacity,
                                            orm_re_generators.generation_type,
                                            orm_re_generators.generation_subtype,
                                            orm_re_generators.voltage_level,
                                            func.ST_AsText(func.ST_Transform(
                                            orm_re_generators.rea_geom_new, srid)).label('geom_new'),
                                            func.ST_AsText(func.ST_Transform(
                                            orm_re_generators.geom, srid)).label('geom')
                                            ).\
                                            filter(orm_re_generators.subst_id.in_(list(mv_grid_districts_dict))). \
                                            filter(orm_re_generators.voltage_level.in_([4, 5, 6, 7]))
                                            #filter(orm_re_generators.voltage_level.in_([4, 5]))
                                            #filter(orm_re_generators.voltage_level.in_([4, 5, 6, 7]))

            # TODO: Currently only MV generators are imported, please include LV!

            # read data from db
            generators = pd.read_sql_query(generators_sqla.statement,
                                           session.bind,
                                           index_col='id')

            for id_db, row in generators.iterrows():
                # ===== DEBUG STUFF (NOT ALL GENERATOR GEOMS IN DATABASE YET -> catch empty geoms) =====
                # TODO: Remove when fixed! And delete column 'geom' (original geom from EnergyMap) from query above
                if not row['geom_new']:
                    geo_data = wkt_loads(row['geom'])
                    print('Generator', str(id_db),
                          'has no geom_new entry, bad day dude! (EnergyMap\'s geom entry will be used)')
                elif not row['geom_new']:
                    print('Generator', str(id_db), 'has no geom entry either, your day is getting worse dude!')
                else:
                    geo_data = wkt_loads(row['geom_new'])
                # ======================================================================================

                # look up MV grid
                mv_grid_district_id = row['subst_id']
                mv_grid = mv_grid_districts_dict[mv_grid_district_id].mv_grid

                # look up LV load area
                lv_load_area_id = row['la_id']
                lv_grid_district_id = row['mvlv_subst_id']
                if lv_load_area_id and not isnan(lv_load_area_id):
                    lv_load_area = lv_load_areas_dict[lv_load_area_id]
                    # TODO: current state: no alloc of geno to lvgd / lv grid
                    # TODO: id of LVGD (mvlv_subst_id) is used for alloc geno to lvgd / lv grid
                    lv_grid_district = None
                    #lv_grid_district = lv_grid_districts_dict[lv_grid_district_id]

                else:
                    lv_load_area = None
                    lv_grid_district = None

                # create generator object
                generator = GeneratorDingo(id_db=id_db,
                                           geo_data=geo_data,
                                           mv_grid=mv_grid,
                                           lv_load_area=lv_load_area,
                                           lv_grid=lv_grid_district,
                                           capacity=row['electrical_capacity'],
                                           type=row['generation_type'],
                                           subtype=row['generation_subtype'],
                                           v_level=int(row['voltage_level']))

                # add generators to graph
                if generator.v_level in [4, 5]:
                    mv_grid.add_generator(generator)
                elif generator.v_level in [6, 7]:
                    # TODO: current state: append LV genos to LA list "genos_collected_temp"
                    if lv_load_area is not None:
                        lv_load_area.genos_collected_temp.append(generator)
                    else:
                        print('Error: Generator', str(id_db), 'has no la_id and cannot be assigned!')
                    #lv_grid_district.lv_grid.add_generator(generator)

        def import_conv_generators():
            """Imports conventional (conv) generators"""
            # TODO: Alloc to LA required (add to peak_generation of LV stations) !

            # build query
            generators_sqla = session.query(orm_conv_generators.gid,
                                            orm_conv_generators.subst_id,
                                            orm_conv_generators.name,
                                            orm_conv_generators.capacity,
                                            orm_conv_generators.fuel,
                                            orm_conv_generators.voltage_level,
                                            func.ST_AsText(func.ST_Transform(
                                            orm_conv_generators.geom, srid)).label('geom')).\
                                            filter(orm_conv_generators.subst_id.in_(list(mv_grid_districts_dict))).\
                                            filter(orm_conv_generators.voltage_level.in_([4, 5, 6]))

            # read data from db
            generators = pd.read_sql_query(generators_sqla.statement,
                                           session.bind,
                                           index_col='gid')

            for id_db, row in generators.iterrows():

                # look up MV grid
                mv_grid_district_id = row['subst_id']
                mv_grid = mv_grid_districts_dict[mv_grid_district_id].mv_grid

                # create generator object
                generator = GeneratorDingo(id_db=id_db,
                                           name=row['name'],
                                           geo_data=wkt_loads(row['geom']),
                                           mv_grid=mv_grid,
                                           capacity=row['capacity'],
                                           type=row['fuel'],
                                           v_level=int(row['voltage_level']))

                # add generators to graph
                if generator.v_level in [4, 5]:
                    mv_grid.add_generator(generator)
                elif generator.v_level in [6]:
                    pass
                    #raise NotImplementedError

        # get dingos' standard CRS (SRID)
        srid = str(int(cfg_dingo.get('geo', 'srid')))

        # make DB session
        Session = sessionmaker(bind=conn)
        session = Session()

        # build dicts to map MV grid district and LV load area ids to related objects
        mv_grid_districts_dict, lv_load_areas_dict, lv_grid_districts_dict = self.get_mvgd_lvla_lvgd_obj_from_id()

        # import renewable generators
        import_res_generators()

        # TODO: current state: LV genos are collected in "genos_collected_temp" and distributed equally to LV stations
        # TODO: this can be dropped as soon as genos's "mvlv_subst_id" is correct (+further modifications required!)
        for grid_district in self.mv_grid_districts():
            for lv_load_area in grid_district.lv_load_areas():
                genos_count = len(lv_load_area.genos_collected_temp)
                lv_grid_district_count = lv_load_area.lv_grid_districts_count()
                genos_per_lvgd = genos_count // lv_grid_district_count
                genos_rest = genos_count % lv_grid_district_count

                # alloc genos to lvgds (equal chunks)
                i = 0
                genos_rest = lv_load_area.genos_collected_temp
                for lv_grid_district in lv_load_area.lv_grid_districts():
                    for geno in lv_load_area.genos_collected_temp[i*genos_per_lvgd:i*genos_per_lvgd + genos_per_lvgd]:
                        lv_grid_district.lv_grid.add_generator(geno)
                        lv_grid_district.lv_grid._station.peak_generation += geno.capacity
                        genos_rest.remove(geno)
                    i += 1
                # alloc genos to lvgds (rest)
                i = 0
                while genos_rest != []:
                    geno = genos_rest[0]
                    lv_grid_district = list(lv_load_area.lv_grid_districts())[i]
                    lv_grid_district.lv_grid.add_generator(geno)
                    lv_grid_district.lv_grid._station.peak_generation += geno.capacity
                    genos_rest.remove(geno)
                    i += 1

        # import conventional generators
        import_conv_generators()

        print('=====> Generators imported')

    def import_pf_config(self):
        """ Creates power flow config class and imports config from file """

        scenario = cfg_dingo.get("powerflow", "test_grid_stability_scenario")
        start_hour = int(cfg_dingo.get("powerflow", "start_hour"))
        end_hour = int(cfg_dingo.get("powerflow", "end_hour"))
        start_time = datetime(1970, 1, 1, 00, 00, 0)

        resolution = cfg_dingo.get("powerflow", "resolution")
        srid = str(int(cfg_dingo.get('geo', 'srid')))

        self._pf_config = PFConfigDingo(scenarios=[scenario],
                                        timestep_start=start_time,
                                        timesteps_count=end_hour-start_hour,
                                        srid=srid,
                                        resolution=resolution)

    def import_static_data(self):
        """ Imports static data into NetworkDingo such as equipment."""

        package_path = dingo.__path__[0]

        # import equipment
        equipment_parameters_file = cfg_dingo.get('equipment',
                                                  'equipment_mv_parameters_lines')
        self.static_data['MV_overhead_lines'] = pd.read_csv(os.path.join(package_path, 'data',
                                                equipment_parameters_file),
                                                comment='#',
                                                converters={'I_max_th': lambda x: int(x), 'U_n': lambda x: int(x)})

        equipment_parameters_file = cfg_dingo.get('equipment',
                                                  'equipment_mv_parameters_cables')
        self.static_data['MV_cables'] = pd.read_csv(os.path.join(package_path, 'data',
                                        equipment_parameters_file),
                                        comment='#',
                                        converters={'I_max_th': lambda x: int(x), 'U_n': lambda x: int(x)})

        # TODO: Include LV equipment



    def validate_grid_districts(self):
        """ Tests MV grid districts for validity concerning imported data such as:

        Invalid MV grid districts are subsequently deleted from Network.
        """

        msg_invalidity = []

        for grid_district in self.mv_grid_districts():

            # there's only one node (MV station) => grid is empty
            if len(grid_district.mv_grid._graph.nodes()) == 1:
                self._mv_grid_districts.remove(grid_district)
                msg_invalidity.append('MV Grid District {} seems to be empty ' \
                                      'and ' \
                                      'was removed'.format(grid_district))

            # there're only aggregated load areas
            elif all([lvla.is_aggregated for lvla in
                      grid_district.lv_load_areas()]):
                self._mv_grid_districts.remove(grid_district)
                msg_invalidity.append("MV Grid District {} contains only " \
                                 "aggregated LV Load Areas and was removed" \
                                 "".format(grid_district))

        print("\n".join(msg_invalidity))
        print('=====> MV Grids validated')
        return msg_invalidity

    def export_mv_grid(self, conn, mv_grid_districts):
        """ Exports MV grids to database for visualization purposes

        Parameters
        ----------
        conn : sqlalchemy.engine.base.Connection object
               Database connection
        mv_grid_districts : List of MV grid_districts (instances of MVGridDistrictDingo class)
            whose MV grids are exported.

        """

        # check arguments
        if not all(isinstance(_, int) for _ in mv_grid_districts):
            raise TypeError('`mv_grid_districts` has to be a list of integers.')

        srid = str(int(cfg_dingo.get('geo', 'srid')))

        Session = sessionmaker(bind=conn)
        session = Session()

        # delete all existing datasets
        # db_int.sqla_mv_grid_viz.__table__.create(conn) # create if not exist
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
            mv_generators = []
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
                elif isinstance(node, GeneratorDingo):
                    mv_generators.append((node.geo_data.x, node.geo_data.y))

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

            if mv_generators:
                mv_generators_wkb = from_shape(MultiPoint(mv_generators), srid=srid)
            else:
                mv_generators_wkb = None

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
                    if station not in grid_district.mv_grid.graph_isolated_nodes():
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
                geom_mv_generators=mv_generators_wkb,
                geom_mv_lines=mv_lines_wkb)
            session.add(dataset)

        # commit changes to db
        session.commit()

        print('=====> MV Grids exported')

    def export_mv_grid_new(self, conn, mv_grid_districts):
        """ Exports MV grids to database for visualization purposes

        Parameters
        ----------
        conn : sqlalchemy.engine.base.Connection object
               Database connection
        mv_grid_districts : List of MV grid_districts (instances of MVGridDistrictDingo class)
            whose MV grids are exported.

        """

        # check arguments
        if not all(isinstance(_, int) for _ in mv_grid_districts):
            raise TypeError('`mv_grid_districts` has to be a list of integers.')

        srid = str(int(cfg_dingo.get('geo', 'srid')))

        Session = sessionmaker(bind=conn)
        session = Session()

        # delete all existing datasets
        # db_int.sqla_mv_grid_viz_branches.__table__.create(conn) # create if not exist
        # db_int.sqla_mv_grid_viz_nodes.__table__.create(conn) # create if not exist
        session.query(db_int.sqla_mv_grid_viz_branches).delete()
        session.query(db_int.sqla_mv_grid_viz_nodes).delete()
        session.commit()

        # build data array from MV grids (nodes and branches)
        for grid_district in self.mv_grid_districts():

            # get nodes from grid's graph and create datasets
            for node in grid_district.mv_grid._graph.nodes():
                if hasattr(node, 'voltage_res'):
                    node_name = '_'.join(['MV',
                                          str(grid_district.mv_grid.id_db),
                                          repr(node)])

                    node_dataset = db_int.sqla_mv_grid_viz_nodes(
                        node_id=node_name,
                        grid_id=grid_district.mv_grid.id_db,
                        v_nom=grid_district.mv_grid.v_level,
                        geom=from_shape(Point(node.geo_data), srid=srid),
                        v_res0=node.voltage_res[0],
                        v_res1=node.voltage_res[1]
                    )
                    session.add(node_dataset)

            # get branches (lines) from grid's graph and create datasets
            for branch in grid_district.mv_grid.graph_edges():
                if hasattr(branch['branch'], 's_res'):
                    branch_name = '_'.join(['MV',
                                            str(grid_district.mv_grid.id_db),
                                            'lin',
                                            str(branch['branch'].id_db)])

                    branch_dataset = db_int.sqla_mv_grid_viz_branches(
                        branch_id=branch_name,
                        grid_id=grid_district.mv_grid.id_db,
                        type_name=branch['branch'].type['name'],
                        type_kind=branch['branch'].kind,
                        type_v_nom=branch['branch'].type['U_n'],
                        type_s_nom=3**0.5 * branch['branch'].type['I_max_th'] * branch['branch'].type['U_n'],
                        length=branch['branch'].length / 1e3,
                        geom=from_shape(LineString([branch['adj_nodes'][0].geo_data,
                                                    branch['adj_nodes'][1].geo_data]),
                                        srid=srid),
                        s_res0=branch['branch'].s_res[0],
                        s_res1=branch['branch'].s_res[1]
                    )
                    session.add(branch_dataset)

        # commit changes to db
        session.commit()

        print('=====> MV Grids exported (NEW)')

    def to_dataframe(self):
        """Export grid data to dataframes for statistical analysis

        The export to dataframe is similar to db tables exported by
        `export_mv_grid_new`.

        Returns
        -------
        df : pandas.DataFrame
        """

        node_cols = ['node_id', 'grid_id', 'v_nom', 'geom', 'v_res0', 'v_res1',
                     'peak_load', 'generation_capacity', 'type']
        edges_cols = ['branch_id', 'grid_id', 'type_name', 'type_kind',
                      'type_v_nom', 'type_s_nom', 'length', 'geom', 's_res0',
                      's_res1']

        nodes_df = pd.DataFrame(columns=node_cols)
        edges_df = pd.DataFrame(columns=edges_cols)

        srid = str(int(cfg_dingo.get('geo', 'srid')))

        for grid_district in self.mv_grid_districts():

            # get nodes from grid's graph and create datasets
            for node in grid_district.mv_grid._graph.nodes():
                if hasattr(node, 'voltage_res'):
                    node_name = '_'.join(['MV',
                                          str(grid_district.mv_grid.id_db),
                                          repr(node)])
                    if isinstance(node, LVStationDingo):
                        peak_load = node.peak_load
                        generation_capacity = node.peak_generation
                        type = 'LV Station'
                    elif isinstance(node, GeneratorDingo):
                        peak_load = 0
                        generation_capacity = node.capacity
                        type = node.type
                    elif isinstance(node, MVCableDistributorDingo):
                        peak_load = 0
                        generation_capacity = 0
                        type = 'Cable distributor'
                    elif isinstance(node, LVLoadAreaCentreDingo):
                        #TODO: replace zero at generation/peak load
                        peak_load = 0
                        generation_capacity = 0
                        type = 'Load area center'
                    else:
                        peak_load = 0
                        generation_capacity = 0
                        type = 'Unknown'
                    nodes_df = nodes_df.append(pd.Series(
                        {'node_id': node_name,
                         'grid_id': grid_district.mv_grid.id_db,
                         'v_nom': grid_district.mv_grid.v_level,
                         'geom': from_shape(Point(node.geo_data), srid=srid),
                         'peak_load': peak_load,
                         'generation_capacity': generation_capacity,
                         'v_res0': node.voltage_res[0],
                         'v_res1': node.voltage_res[1],
                         #'type': type,
                        }), ignore_index=True)

            # get branches (lines) from grid's graph and create datasets
            for branch in grid_district.mv_grid.graph_edges():
                if hasattr(branch['branch'], 's_res'):
                    branch_name = '_'.join(['MV',
                                            str(
                                                grid_district.mv_grid.id_db),
                                            'lin',
                                            str(branch[
                                                    'branch'].id_db)])

                    edges_df = edges_df.append(pd.Series(
                        {'branch_id': branch_name,
                        'grid_id': grid_district.mv_grid.id_db,
                        'type_name': branch['branch'].type['name'],
                        'type_kind': branch['branch'].kind,
                        'type_v_nom': branch['branch'].type['U_n'],
                        'type_s_nom': 3 ** 0.5 * branch['branch'].type[
                            'I_max_th'] * branch['branch'].type['U_n'],
                        'length': branch['branch'].length / 1e3,
                        'geom': from_shape(
                            LineString([branch['adj_nodes'][0].geo_data,
                                        branch['adj_nodes'][
                                            1].geo_data]),
                            srid=srid),
                        's_res0': branch['branch'].s_res[0],
                        's_res1': branch['branch'].s_res[1]}), ignore_index=True)

        return nodes_df, edges_df


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

        print('=====> MV Routing (Routing, Connection of Satellites & Stations) performed')

    def connect_generators(self, debug=False):
        """ Connects generators (graph nodes) to grid (graph) for every MV grid district

        Args:
            debug: If True, information is printed during process
        """

        for mv_grid_district in self.mv_grid_districts():
            mv_grid_district.mv_grid.connect_generators(debug)

            # TODO: Currently only MV generators are connected, use following snippet to call connect LV generators
            #for lv_load_area in mv_grid_district.lv_load_areas():
            #    CALL FUTURE METHOD FOR LV connect_generators HERE

        print('=====> Generators connected')

    def mv_parametrize_grid(self, debug=False):
        """ Performs Parametrization of grid equipment of all MV grids, see
            method `parametrize_grid()` in class `MVGridDingo` for details.

        Parameters
        ----------
        debug: If True, information is printed while parametrization
        """

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.parametrize_grid(debug)

        print('=====> MV Grids parametrized')

    def set_branch_ids(self):
        """ Performs generation and setting of ids of branches for all MV and underlying LV grids, see
            method `set_branch_ids()` in class `MVGridDingo` for details.
        """

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.set_branch_ids()

        print('=====> Branch IDs set')

    def set_circuit_breakers(self, debug=False):
        """ Calculates the optimal position of the existing circuit breakers and relocates them within the graph for
            all MV grids, see method `set_circuit_breakers` in dingo.grid.mv_grid.tools for details.
        Args:
            debug: If True, information is printed during process
        """

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.set_circuit_breakers(debug)

        print('=====> MV Circuit Breakers relocated')

    def control_circuit_breakers(self, mode=None):
        """ Opens or closes all circuit breakers of all MV grids.

        Args:
            mode: Set mode='open' to open, mode='close' to close
            debug: If True, information is printed during process
        """

        for grid_district in self.mv_grid_districts():
            if mode is 'open':
                grid_district.mv_grid.open_circuit_breakers()
            elif mode is 'close':
                grid_district.mv_grid.close_circuit_breakers()
            else:
                raise ValueError('\'mode\' is invalid.')

        if mode is 'open':
            print('=====> MV Circuit Breakers opened')
        elif mode is 'close':
            print('=====> MV Circuit Breakers closed')

    def run_powerflow(self, conn, method='onthefly', export_pypsa=False, debug=False):
        """ Performs power flow calculation for all MV grids

        Args:
            conn: SQLalchemy database connection
            method: str
                Specify export method
                If method='db' grid data will be exported to database
                If method='onthefly' grid data will be passed to PyPSA directly (default)
            export_pypsa: bool
                If True PyPSA networks will be exported as csv to output/debug/grid/<MV-GRID_NAME>/
            debug: If True, information is printed during process
        """

        Session = sessionmaker(bind=conn)
        session = Session()

        if method is 'db':
            # Empty tables
            pypsa_io.delete_powerflow_tables(session)

            for grid_district in self.mv_grid_districts():
                if export_pypsa:
                    export_pypsa_dir = repr(grid_district.mv_grid)
                else:
                    export_pypsa_dir = None
                grid_district.mv_grid.run_powerflow(session, method='db',
                                                    export_pypsa_dir=export_pypsa_dir,
                                                    debug=debug)

        elif method is 'onthefly':
            for grid_district in self.mv_grid_districts():
                if export_pypsa:
                    export_pypsa_dir = repr(grid_district.mv_grid)
                else:
                    export_pypsa_dir = None
                grid_district.mv_grid.run_powerflow(session,
                                                    method='onthefly',
                                                    export_pypsa_dir=export_pypsa_dir,
                                                    debug=debug)

    def control_generators(self):
        """
        Returns:

        """

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.reinforce_grid()

    def reinforce_grid(self):
        """ Performs grid reinforcement measures for all MV and LV grids
        Args:

        Returns:

        """
        # TODO: Finish method and enable LV case

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.reinforce_grid()

            # ===== LV PART (currently disabled) =====
            # for lv_load_area in grid_district.lv_load_areas():
            #     for lv_grid_district in lv_load_area.lv_grid_districts():
            #         lv_grid_district.lv_grid.reinforce_grid()

    def __repr__(self):
        return str(self.name)
