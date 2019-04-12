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
# TODO: check docstrings


import ding0
from ding0.config import config_db_interfaces as db_int
from ding0.core.network import GeneratorDing0, GeneratorFluctuatingDing0
from ding0.core.network.cable_distributors import MVCableDistributorDing0
from ding0.core.network.grids import *
from ding0.core.network.stations import *
from ding0.core.structure.regions import *
from ding0.core.powerflow import *
from ding0.tools import pypsa_io
from ding0.tools.animation import AnimationDing0
from ding0.tools.plots import plot_mv_topology
from ding0.flexopt.reinforce_grid import *

import os
import logging
import pandas as pd
import random
import time
from math import isnan

from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from geoalchemy2.shape import from_shape
import subprocess
import oedialect

if not 'READTHEDOCS' in os.environ:
    from shapely.wkt import loads as wkt_loads
    from shapely.geometry import Point, MultiPoint, MultiLineString, LineString
    from shapely.geometry import shape, mapping
    from shapely.wkt import dumps as wkt_dumps

logger = logging.getLogger('ding0')

package_path = ding0.__path__[0]


class NetworkDing0:
    """
    Defines the DING0 Network - not a real grid but a container for the
    MV-grids. Contains the NetworkX graph and associated attributes.

    This object behaves like a location to
    store all the constituent objects required to estimate the grid topology
    of a give set of shapes that need to be connected.

    The most important function that defines ding0's use case is initiated
    from this class i.e. :meth:`~.core.NetworkDing0.run_ding0`.


    Parameters
    ----------
    name : :obj:`str`
        A name given to the network. This defaults to `Network`.

    run_id : :obj:`str`
        A unique identification number to identify different runs of
        Ding0. This is usually the date and the time in some compressed
        format. e.g. 201901010900.


    Attributes
    ----------
    mv_grid_districts: :obj:`list iterator`
        Contains the MV Grid Districts where the topology has to be estimated
        A list of :class:`~.ding0.core.structure.regions.MVGridDistrictDing0`
        objects whose data is stored in the current instance of
        the :class:`~.ding0.core.NetworkDing0` Object.
        By default the list is empty. MV grid districts can be added by
        using the function :meth:`~.core.NetworkDing0.add_mv_grid_district`. This is done
        within the function :meth:`~.core.NetworkDing0.build_mv_grid_district`
        in the normal course upon calling :meth:`~.core.NetworkDing0.run_ding0`.

    config : :obj:`dict`
        These are the configurations that are required for the
        construction of the network topology given the areas to be connected
        together. The configuration is imported by calling
        :meth:`~.core.NetworkDing0.import_config`.
        The configurations are stored in text files within the
        ding0 package in the config folder. These get imported into a
        python dictionary-like configuration object.

    pf_config : :class:`~.ding0.core.powerflow.PFConfigDing0`
        These are the configuration of the power flows that are
        run to ensure that the generated network is plausible and is
        capable of a reasonable amount of loading without causing any
        grid issues. This object cannot be set at inititation, it gets set by
        the function :meth:`~.core.NetworkDing0.import_pf_config` which
        takes the configurations from :attr:_config and sets up
        the configurations for running power flow calculations.

    static_data : :obj:`dict`
        Data such as electrical and mechanical properties
        of typical assets in the energy system are stored in ding0.
        These are used in many parts of ding0's calculations.
        Data values:

        * Typical cable types, and typical line types' electrical impedences,
            thermal ratings, operating voltage level.
        * Typical transformers types' electrical impedences, voltage drops,
            thermal ratings, winding voltages
        * Typical LV grid topologies' line types, line lengths and
            distribution

    orm : :obj:`dict`
        The connection parameters to the OpenEnergy Platform and
        the tables and datasets required for the functioning of ding0

    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self._run_id = kwargs.get('run_id', None)
        self._mv_grid_districts = []

        self._config = self.import_config()
        self._pf_config = self.import_pf_config()
        self._static_data = self.import_static_data()
        self._orm = self.import_orm()

    def mv_grid_districts(self):
        """
        A generator for iterating over MV grid_districts

        Returns
        ------
        :obj:`list iterator`
            A list iterator containing the
            :class:`~.ding0.core.structure.regions.MVGridDistrictDing0` objects.
        """
        for grid_district in self._mv_grid_districts:
            yield grid_district

    def add_mv_grid_district(self, mv_grid_district):
        """
        A method to add mv_grid_districts to the
        :class:`~.core.NetworkDing0` Object by adding it to the
        :attr:`~.core.NetworkDing0.mv_grid_districts`.
        """
        # TODO: use setter method here (make attribute '_mv_grid_districts' private)
        if mv_grid_district not in self.mv_grid_districts():
            self._mv_grid_districts.append(mv_grid_district)

    @property
    def config(self):
        """
        Getter for the configuration dictionary.


        Returns
        -------
         :obj:`dict`
         """
        return self._config

    @property
    def pf_config(self):
        """
        Getter for the power flow calculation configurations.

        Returns
        -------
        :class:`~.ding0.core.powerflow.PFConfigDing0`
        """
        return self._pf_config

    @property
    def static_data(self):
        """
        Getter for the static data

        Returns
        -------
        :obj: `dict`
        """
        return self._static_data

    @property
    def orm(self):
        """
        Getter for the stored ORM configurations.

        Returns
        -------
        :obj: `dict`
        """
        return self._orm

    def run_ding0(self, session, mv_grid_districts_no=None, debug=False, export_figures=False):
        """
        Let DING0 run by shouting at this method (or just call
        it from NetworkDing0 instance). This method is a wrapper
        for the main functionality of DING0.

        Parameters
        ----------
        session : :obj:`sqlalchemy.orm.session.Session`
            Database session
        mv_grid_districts_no : :obj:`list` of :obj:`int` objects.
            List of MV grid_districts/stations to be imported (if empty,
            all grid_districts & stations are imported)
        debug : obj:`bool`, defaults to False
            If True, information is printed during process
        export_figures : :obj:`bool`, defaults to False
            If True, figures are shown or exported (default path: ~/.ding0/) during run.

        Returns
        -------
        msg : obj:`str`
            Message of invalidity of a grid district

        Note
        -----
        The steps performed in this method are to be kept in the given order
        since there are hard dependencies between them. Short description of
        all steps performed:

        * STEP 1: Import MV Grid Districts and subjacent objects

            Imports MV Grid Districts, HV-MV stations, Load Areas, LV Grid Districts
            and MV-LV stations, instantiates and initiates objects.

        * STEP 2: Import generators

            Conventional and renewable generators of voltage levels 4..7 are imported
            and added to corresponding grid.

        * STEP 3: Parametrize grid

            Parameters of MV grid are set such as voltage level and cable/line types
            according to MV Grid District's characteristics.

        * STEP 4: Validate MV Grid Districts

            Tests MV grid districts for validity concerning imported data such as
            count of Load Areas.

        * STEP 5: Build LV grids

            Builds LV grids for every non-aggregated LA in every MV Grid District
            using model grids.

        * STEP 6: Build MV grids

            Builds MV grid by performing a routing on Load Area centres to build
            ring topology.

        * STEP 7: Connect MV and LV generators

            Generators are connected to grids, used approach depends on voltage
            level.

        * STEP 8: Set IDs for all branches in MV and LV grids

            While IDs of imported objects can be derived from dataset's ID, branches
            are created in steps 5+6 and need unique IDs (e.g. for PF calculation).

        * STEP 9: Relocate switch disconnectors in MV grid

            Switch disconnectors are set during routing process (step 6) according
            to the load distribution within a ring. After further modifications of
            the grid within step 6+7 they have to be relocated (note: switch
            disconnectors are called circuit breakers in DING0 for historical reasons).

        * STEP 10: Open all switch disconnectors in MV grid

            Under normal conditions, rings are operated in open state (half-rings).
            Furthermore, this is required to allow powerflow for MV grid.

        * STEP 11: Do power flow analysis of MV grid

            The technically working MV grid created in step 6 was extended by satellite
            loads and generators. It is finally tested again using powerflow calculation.

        * STEP 12: Reinforce MV grid

            MV grid is eventually reinforced persuant to results from step 11.

        STEP 13: Close all switch disconnectors in MV grid
            The rings are finally closed to hold a complete graph (if the SDs are open,
            the edges adjacent to a SD will not be exported!)
        """
        if debug:
            start = time.time()

        # STEP 1: Import MV Grid Districts and subjacent objects
        self.import_mv_grid_districts(session,
                                      mv_grid_districts_no=mv_grid_districts_no)

        # STEP 2: Import generators
        self.import_generators(session, debug=debug)

        # STEP 3: Parametrize MV grid
        self.mv_parametrize_grid(debug=debug)

        # STEP 4: Validate MV Grid Districts
        msg = self.validate_grid_districts()

        # STEP 5: Build LV grids
        self.build_lv_grids()

        # STEP 6: Build MV grids
        self.mv_routing(debug=False)
        if export_figures:
            grid = self._mv_grid_districts[0].mv_grid
            plot_mv_topology(grid, subtitle='Routing completed', filename='1_routing_completed.png')

        # STEP 7: Connect MV and LV generators
        self.connect_generators(debug=False)
        if export_figures:
            plot_mv_topology(grid, subtitle='Generators connected', filename='2_generators_connected.png')

        # STEP 8: Set IDs for all branches in MV and LV grids
        self.set_branch_ids()

        # STEP 9: Relocate switch disconnectors in MV grid
        self.set_circuit_breakers(debug=debug)
        if export_figures:
            plot_mv_topology(grid, subtitle='Circuit breakers relocated', filename='3_circuit_breakers_relocated.png')

        # STEP 10: Open all switch disconnectors in MV grid
        self.control_circuit_breakers(mode='open')

        # STEP 11: Do power flow analysis of MV grid
        self.run_powerflow(session, method='onthefly', export_pypsa=False, debug=debug)
        if export_figures:
            plot_mv_topology(grid, subtitle='PF result (load case)',
                             filename='4_PF_result_load.png',
                             line_color='loading', node_color='voltage', testcase='load')
            plot_mv_topology(grid, subtitle='PF result (feedin case)',
                             filename='5_PF_result_feedin.png',
                             line_color='loading', node_color='voltage', testcase='feedin')

        # STEP 12: Reinforce MV grid
        self.reinforce_grid()

        # STEP 13: Close all switch disconnectors in MV grid
        self.control_circuit_breakers(mode='close')

        if export_figures:
            plot_mv_topology(grid, subtitle='Final grid PF result (load case)',
                             filename='6_final_grid_PF_result_load.png',
                             line_color='loading', node_color='voltage', testcase='load')
            plot_mv_topology(grid, subtitle='Final grid PF result (feedin case)',
                             filename='7_final_grid_PF_result_feedin.png',
                             line_color='loading', node_color='voltage', testcase='feedin')

        if debug:
            logger.info('Elapsed time for {0} MV Grid Districts (seconds): {1}'.format(
                str(len(mv_grid_districts_no)), time.time() - start))

        return msg

    def get_mvgd_lvla_lvgd_obj_from_id(self):
        """
        Build dict with mapping from:

        * :class:`~.ding0.core.structure.regions.LVLoadAreaDing0` ``id`` to
          :class:`~.ding0.core.structure.regions.LVLoadAreaDing0` object,
        * :class:`~.ding0.core.structure.regions.MVGridDistrictDing0` ``id`` to
          :class:`~.ding0.core.structure.regions.MVGridDistrictDing0` object,
        * :class:`~.ding0.core.structure.regions.LVGridDistrictDing0` ``id`` to
          :class:`~.ding0.core.structure.regions.LVGridDistrictDing0` object
        * :class:`~.ding0.core.network.stations.LVStationDing0` ``id`` to
          :class:`~.ding0.core.network.stations.LVStationDing0` object

        Returns
        -------
        :obj:`dict`
            mv_grid_districts_dict::

                {
                  mv_grid_district_id_1: mv_grid_district_obj_1,
                  ...,
                  mv_grid_district_id_n: mv_grid_district_obj_n
                }
        :obj:`dict`
            lv_load_areas_dict::

                {
                  lv_load_area_id_1: lv_load_area_obj_1,
                  ...,
                  lv_load_area_id_n: lv_load_area_obj_n
                }
        :obj:`dict`
            lv_grid_districts_dict::

                {
                  lv_grid_district_id_1: lv_grid_district_obj_1,
                  ...,
                  lv_grid_district_id_n: lv_grid_district_obj_n
                }
        :obj:`dict`
            lv_stations_dict::

                {
                  lv_station_id_1: lv_station_obj_1,
                  ...,
                  lv_station_id_n: lv_station_obj_n
                }
        """

        mv_grid_districts_dict = {}
        lv_load_areas_dict = {}
        lv_grid_districts_dict = {}
        lv_stations_dict = {}

        for mv_grid_district in self.mv_grid_districts():
            mv_grid_districts_dict[mv_grid_district.id_db] = mv_grid_district
            for lv_load_area in mv_grid_district.lv_load_areas():
                lv_load_areas_dict[lv_load_area.id_db] = lv_load_area
                for lv_grid_district in lv_load_area.lv_grid_districts():
                    lv_grid_districts_dict[lv_grid_district.id_db] = lv_grid_district
                    lv_stations_dict[lv_grid_district.lv_grid.station().id_db] = lv_grid_district.lv_grid.station()

        return mv_grid_districts_dict, lv_load_areas_dict, lv_grid_districts_dict, lv_stations_dict

    def build_mv_grid_district(self, poly_id, subst_id, grid_district_geo_data,
                        station_geo_data):
        """
        Initiates single MV grid_district including station and grid

        Parameters
        ----------

        poly_id: :obj:`int`
            ID of grid_district according to database table. Also used as ID for created grid #TODO: check type
        subst_id: :obj:`int`
            ID of station according to database table #TODO: check type
        grid_district_geo_data: :shapely:`Shapely Polygon object<polygons>`
            Polygon of grid district, The geo-spatial polygon
            in the coordinate reference system with the
            SRID:4326 or epsg:4326, this is the project
            used by the ellipsoid WGS 84.
        station_geo_data: :shapely:`Shapely Point object<points>`
            Point of station. The geo-spatial point
            in the coordinate reference
            system with the SRID:4326 or epsg:4326, this
            is the project used by the ellipsoid WGS 84.

        Returns
        -------
        :class:`~.ding0.core.structure.regions.MVGridDistrictDing0`

        """

        mv_station = MVStationDing0(id_db=subst_id, geo_data=station_geo_data)

        mv_grid = MVGridDing0(network=self,
                              id_db=poly_id,
                              station=mv_station)
        mv_grid_district = MVGridDistrictDing0(id_db=poly_id,
                                               mv_grid=mv_grid,
                                               geo_data=grid_district_geo_data)
        mv_grid.grid_district = mv_grid_district
        mv_station.grid = mv_grid

        self.add_mv_grid_district(mv_grid_district)

        return mv_grid_district

    def build_lv_grid_district(self,
                               lv_load_area,
                               lv_grid_districts,
                               lv_stations):
        """
        Instantiates and associates lv_grid_district incl grid and station.

        The instantiation creates more or less empty objects including relevant
        data for transformer choice and grid creation

        Parameters
        ----------
        lv_load_area: :shapely:`Shapely Polygon object<polygons>`
            load_area object
        lv_grid_districts: :pandas:`pandas.DataFrame<dataframe>`
            Table containing lv_grid_districts of according load_area
        lv_stations : :pandas:`pandas.DataFrame<dataframe>`
            Table containing lv_stations of according load_area
        """

        # There's no LVGD for current LA
        # -> TEMP WORKAROUND: Create single LVGD from LA, replace unknown valuess by zero
        # TODO: Fix #155 (see also: data_processing #68)
        if len(lv_grid_districts) == 0:
            # raise ValueError(
            #     'Load Area {} has no LVGD - please re-open #155'.format(
            #         repr(lv_load_area)))
            geom = wkt_dumps(lv_load_area.geo_area)

            lv_grid_districts = \
                lv_grid_districts.append(
                    pd.DataFrame(
                        {'la_id': [lv_load_area.id_db],
                         'geom': [geom],
                         'population': [0],

                         'peak_load_residential': [lv_load_area.peak_load_residential],
                         'peak_load_retail': [lv_load_area.peak_load_retail],
                         'peak_load_industrial': [lv_load_area.peak_load_industrial],
                         'peak_load_agricultural': [lv_load_area.peak_load_agricultural],

                         'sector_count_residential': [0],
                         'sector_count_retail': [0],
                         'sector_count_industrial': [0],
                         'sector_count_agricultural': [0],

                         'sector_consumption_residential': [0],
                         'sector_consumption_retail': [0],
                         'sector_consumption_industrial': [0],
                         'sector_consumption_agricultural': [0]
                         },
                        index=[lv_load_area.id_db]
                    )
                )

        lv_nominal_voltage = cfg_ding0.get('assumptions', 'lv_nominal_voltage')

        # Associate lv_grid_district to load_area
        for id, row in lv_grid_districts.iterrows():
            lv_grid_district = LVGridDistrictDing0(
                id_db=id,
                lv_load_area=lv_load_area,
                geo_data=wkt_loads(row['geom']),
                population=0 if isnan(row['population']) else int(row['population']),
                peak_load_residential=row['peak_load_residential'],
                peak_load_retail=row['peak_load_retail'],
                peak_load_industrial=row['peak_load_industrial'],
                peak_load_agricultural=row['peak_load_agricultural'],
                peak_load=(row['peak_load_residential'] +
                               row['peak_load_retail'] +
                               row['peak_load_industrial'] +
                               row['peak_load_agricultural']),
                sector_count_residential=int(row['sector_count_residential']),
                sector_count_retail=int(row['sector_count_retail']),
                sector_count_industrial=int(row['sector_count_industrial']),
                sector_count_agricultural=int(row['sector_count_agricultural']),
                sector_consumption_residential=row[
                    'sector_consumption_residential'],
                sector_consumption_retail=row['sector_consumption_retail'],
                sector_consumption_industrial=row[
                    'sector_consumption_industrial'],
                sector_consumption_agricultural=row[
                    'sector_consumption_agricultural'])

            # be aware, lv_grid takes grid district's geom!
            lv_grid = LVGridDing0(network=self,
                                  grid_district=lv_grid_district,
                                  id_db=id,
                                  geo_data=wkt_loads(row['geom']),
                                  v_level=lv_nominal_voltage)

            # create LV station
            lv_station = LVStationDing0(
                id_db=id,
                grid=lv_grid,
                lv_load_area=lv_load_area,
                geo_data=wkt_loads(lv_stations.loc[id, 'geom'])
                            if id in lv_stations.index.values
                            else lv_load_area.geo_centre,
                peak_load=lv_grid_district.peak_load)

            # assign created objects
            # note: creation of LV grid is done separately,
            # see NetworkDing0.build_lv_grids()
            lv_grid.add_station(lv_station)
            lv_grid_district.lv_grid = lv_grid
            lv_load_area.add_lv_grid_district(lv_grid_district)

    def import_mv_grid_districts(self, session, mv_grid_districts_no=None):
        """
        Imports MV Grid Districts, HV-MV stations, Load Areas, LV Grid Districts
        and MV-LV stations, instantiates and initiates objects.

        Parameters
        ----------
        session : :obj:`sqlalchemy.orm.session.Session`
            Database session
        mv_grid_districts : :obj:`list` of :obj:`int`
            List of MV grid_districts/stations (int) to be imported (if empty,
            all grid_districts & stations are imported)

        See Also
        --------
        build_mv_grid_district : used to instantiate MV grid_district objects
        import_lv_load_areas : used to import load_areas for every single MV grid_district
        ding0.core.structure.regions.MVGridDistrictDing0.add_peak_demand : used to summarize peak loads of underlying load_areas
        """

        # check arguments
        if not all(isinstance(_, int) for _ in mv_grid_districts_no):
            raise TypeError('`mv_grid_districts` has to be a list of integers.')

        # get srid settings from config
        try:
            srid = str(int(cfg_ding0.get('geo', 'srid')))
        except OSError:
            logger.exception('cannot open config file.')

        # build SQL query
        grid_districts = session.query(self.orm['orm_mv_grid_districts'].subst_id,
                                       func.ST_AsText(func.ST_Transform(
                                           self.orm['orm_mv_grid_districts'].geom, srid)). \
                                       label('poly_geom'),
                                       func.ST_AsText(func.ST_Transform(
                                           self.orm['orm_mv_stations'].point, srid)). \
                                       label('subs_geom')).\
            join(self.orm['orm_mv_stations'], self.orm['orm_mv_grid_districts'].subst_id ==
                 self.orm['orm_mv_stations'].subst_id).\
            filter(self.orm['orm_mv_grid_districts'].subst_id.in_(mv_grid_districts_no)). \
            filter(self.orm['version_condition_mvgd']). \
            filter(self.orm['version_condition_mv_stations']). \
            distinct()

        # read MV data from db
        mv_data = pd.read_sql_query(grid_districts.statement,
                                    session.bind,
                                    index_col='subst_id')

        # iterate over grid_district/station datasets and initiate objects
        for poly_id, row in mv_data.iterrows():
            subst_id = poly_id
            region_geo_data = wkt_loads(row['poly_geom'])

            # transform `region_geo_data` to epsg 3035
            # to achieve correct area calculation of mv_grid_district
            station_geo_data = wkt_loads(row['subs_geom'])
            # projection = partial(
            #     pyproj.transform,
            #     pyproj.Proj(init='epsg:4326'),  # source coordinate system
            #     pyproj.Proj(init='epsg:3035'))  # destination coordinate system
            #
            # region_geo_data = transform(projection, region_geo_data)

            mv_grid_district = self.build_mv_grid_district(poly_id,
                                             subst_id,
                                             region_geo_data,
                                             station_geo_data)

            # import all lv_stations within mv_grid_district
            lv_stations = self.import_lv_stations(session)

            # import all lv_grid_districts within mv_grid_district
            lv_grid_districts = self.import_lv_grid_districts(session, lv_stations)

            # import load areas
            self.import_lv_load_areas(session,
                                      mv_grid_district,
                                      lv_grid_districts,
                                      lv_stations)

            # add sum of peak loads of underlying lv grid_districts to mv_grid_district
            mv_grid_district.add_peak_demand()

        logger.info('=====> MV Grid Districts imported')

    def import_lv_load_areas(self, session, mv_grid_district, lv_grid_districts,
                             lv_stations):
        """
        Imports load_areas (load areas) from database for a single MV grid_district

        Parameters
        ----------
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session
        mv_grid_district : MV grid_district/station (instance of MVGridDistrictDing0 class) for
            which the import of load areas is performed
        lv_grid_districts: :pandas:`pandas.DataFrame<dataframe>`
            LV grid districts within this mv_grid_district
        lv_stations: :pandas:`pandas.DataFrame<dataframe>`
            LV stations within this mv_grid_district
        """

        # get ding0s' standard CRS (SRID)
        srid = str(int(cfg_ding0.get('geo', 'srid')))
        # SET SRID 3035 to achieve correct area calculation of lv_grid_district
        #srid = '3035'

        # threshold: load area peak load, if peak load < threshold => disregard
        # load area
        lv_loads_threshold = cfg_ding0.get('mv_routing', 'load_area_threshold')

        gw2kw = 10 ** 6  # load in database is in GW -> scale to kW

        # build SQL query
        lv_load_areas_sqla = session.query(
            self.orm['orm_lv_load_areas'].id.label('id_db'),
            self.orm['orm_lv_load_areas'].zensus_sum,
            self.orm['orm_lv_load_areas'].zensus_count.label('zensus_cnt'),
            self.orm['orm_lv_load_areas'].ioer_sum,
            self.orm['orm_lv_load_areas'].ioer_count.label('ioer_cnt'),
            self.orm['orm_lv_load_areas'].area_ha.label('area'),
            self.orm['orm_lv_load_areas'].sector_area_residential,
            self.orm['orm_lv_load_areas'].sector_area_retail,
            self.orm['orm_lv_load_areas'].sector_area_industrial,
            self.orm['orm_lv_load_areas'].sector_area_agricultural,
            self.orm['orm_lv_load_areas'].sector_share_residential,
            self.orm['orm_lv_load_areas'].sector_share_retail,
            self.orm['orm_lv_load_areas'].sector_share_industrial,
            self.orm['orm_lv_load_areas'].sector_share_agricultural,
            self.orm['orm_lv_load_areas'].sector_count_residential,
            self.orm['orm_lv_load_areas'].sector_count_retail,
            self.orm['orm_lv_load_areas'].sector_count_industrial,
            self.orm['orm_lv_load_areas'].sector_count_agricultural,
            self.orm['orm_lv_load_areas'].nuts.label('nuts_code'),
            func.ST_AsText(func.ST_Transform(self.orm['orm_lv_load_areas'].geom, srid)).\
                label('geo_area'),
            func.ST_AsText(func.ST_Transform(self.orm['orm_lv_load_areas'].geom_centre, srid)).\
                label('geo_centre'),
            (self.orm['orm_lv_load_areas'].sector_peakload_residential * gw2kw).\
                label('peak_load_residential'),
            (self.orm['orm_lv_load_areas'].sector_peakload_retail * gw2kw).\
                label('peak_load_retail'),
            (self.orm['orm_lv_load_areas'].sector_peakload_industrial * gw2kw).\
                label('peak_load_industrial'),
            (self.orm['orm_lv_load_areas'].sector_peakload_agricultural * gw2kw).\
                label('peak_load_agricultural'),
            ((self.orm['orm_lv_load_areas'].sector_peakload_residential
              + self.orm['orm_lv_load_areas'].sector_peakload_retail
              + self.orm['orm_lv_load_areas'].sector_peakload_industrial
              + self.orm['orm_lv_load_areas'].sector_peakload_agricultural)
             * gw2kw).label('peak_load')). \
            filter(self.orm['orm_lv_load_areas'].subst_id == mv_grid_district. \
                   mv_grid._station.id_db).\
            filter(((self.orm['orm_lv_load_areas'].sector_peakload_residential  # only pick load areas with peak load > lv_loads_threshold
                     + self.orm['orm_lv_load_areas'].sector_peakload_retail
                     + self.orm['orm_lv_load_areas'].sector_peakload_industrial
                     + self.orm['orm_lv_load_areas'].sector_peakload_agricultural)
                       * gw2kw) > lv_loads_threshold). \
            filter(self.orm['version_condition_la'])

        # read data from db
        lv_load_areas = pd.read_sql_query(lv_load_areas_sqla.statement,
                                          session.bind,
                                          index_col='id_db')

        # create load_area objects from rows and add them to graph
        for id_db, row in lv_load_areas.iterrows():

            # create LV load_area object
            lv_load_area = LVLoadAreaDing0(id_db=id_db,
                                           db_data=row,
                                           mv_grid_district=mv_grid_district,
                                           peak_load=row['peak_load'])

            # sub-selection of lv_grid_districts/lv_stations within one
            # specific load area
            lv_grid_districts_per_load_area = lv_grid_districts.\
                loc[lv_grid_districts['la_id'] == id_db]
            lv_stations_per_load_area = lv_stations.\
                loc[lv_stations['la_id'] == id_db]

            self.build_lv_grid_district(lv_load_area,
                                        lv_grid_districts_per_load_area,
                                        lv_stations_per_load_area)

            # create new centre object for Load Area
            lv_load_area_centre = LVLoadAreaCentreDing0(id_db=id_db,
                                                        geo_data=wkt_loads(row['geo_centre']),
                                                        lv_load_area=lv_load_area,
                                                        grid=mv_grid_district.mv_grid)
            # links the centre object to Load Area
            lv_load_area.lv_load_area_centre = lv_load_area_centre

            # add Load Area to MV grid district (and add centre object to MV gris district's graph)
            mv_grid_district.add_lv_load_area(lv_load_area)

    def import_lv_grid_districts(self, session, lv_stations):
        """Imports all lv grid districts within given load area

        Parameters
        ----------
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session

        Returns
        -------
        lv_grid_districts: :pandas:`pandas.DataFrame<dataframe>`
            Table of lv_grid_districts
        """

        # get ding0s' standard CRS (SRID)
        srid = str(int(cfg_ding0.get('geo', 'srid')))
        # SET SRID 3035 to achieve correct area calculation of lv_grid_district
        # srid = '3035'

        gw2kw = 10 ** 6  # load in database is in GW -> scale to kW

        # 1. filter grid districts of relevant load area
        lv_grid_districs_sqla = session.query(
            self.orm['orm_lv_grid_district'].mvlv_subst_id,
            self.orm['orm_lv_grid_district'].la_id,
            self.orm['orm_lv_grid_district'].zensus_sum.label('population'),
            (self.orm[
                 'orm_lv_grid_district'].sector_peakload_residential * gw2kw).
                label('peak_load_residential'),
            (self.orm['orm_lv_grid_district'].sector_peakload_retail * gw2kw).
                label('peak_load_retail'),
            (self.orm[
                 'orm_lv_grid_district'].sector_peakload_industrial * gw2kw).
                label('peak_load_industrial'),
            (self.orm[
                 'orm_lv_grid_district'].sector_peakload_agricultural * gw2kw).
                label('peak_load_agricultural'),
            ((self.orm['orm_lv_grid_district'].sector_peakload_residential
              + self.orm['orm_lv_grid_district'].sector_peakload_retail
              + self.orm['orm_lv_grid_district'].sector_peakload_industrial
              + self.orm['orm_lv_grid_district'].sector_peakload_agricultural)
             * gw2kw).label('peak_load'),
            func.ST_AsText(func.ST_Transform(
                self.orm['orm_lv_grid_district'].geom, srid)).label('geom'),
            self.orm['orm_lv_grid_district'].sector_count_residential,
            self.orm['orm_lv_grid_district'].sector_count_retail,
            self.orm['orm_lv_grid_district'].sector_count_industrial,
            self.orm['orm_lv_grid_district'].sector_count_agricultural,
            (self.orm[
                 'orm_lv_grid_district'].sector_consumption_residential * gw2kw). \
                label('sector_consumption_residential'),
            (self.orm['orm_lv_grid_district'].sector_consumption_retail * gw2kw). \
                label('sector_consumption_retail'),
            (self.orm[
                'orm_lv_grid_district'].sector_consumption_industrial * gw2kw). \
                label('sector_consumption_industrial'),
            (self.orm[
                'orm_lv_grid_district'].sector_consumption_agricultural * gw2kw). \
                label('sector_consumption_agricultural'),
            self.orm['orm_lv_grid_district'].mvlv_subst_id). \
            filter(self.orm['orm_lv_grid_district'].mvlv_subst_id.in_(
            lv_stations.index.tolist())). \
            filter(self.orm['version_condition_lvgd'])

        # read data from db
        lv_grid_districts = pd.read_sql_query(lv_grid_districs_sqla.statement,
                                              session.bind,
                                              index_col='mvlv_subst_id')

        lv_grid_districts[
            ['sector_count_residential',
             'sector_count_retail',
             'sector_count_industrial',
             'sector_count_agricultural']] = lv_grid_districts[
            ['sector_count_residential',
             'sector_count_retail',
             'sector_count_industrial',
             'sector_count_agricultural']].fillna(0)

        return lv_grid_districts

    def import_lv_stations(self, session):
        """
        Import lv_stations within the given load_area

        Parameters
        ----------
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session

        Returns
        -------
        lv_stations: :pandas:`pandas.DataFrame<dataframe>`
            Table of lv_stations
        """

        # get ding0s' standard CRS (SRID)
        srid = str(int(cfg_ding0.get('geo', 'srid')))

        # get list of mv grid districts
        mv_grid_districts = list(self.get_mvgd_lvla_lvgd_obj_from_id()[0])

        lv_stations_sqla = session.query(self.orm['orm_lv_stations'].mvlv_subst_id,
                                         self.orm['orm_lv_stations'].la_id,
                                         func.ST_AsText(func.ST_Transform(
                                           self.orm['orm_lv_stations'].geom, srid)). \
                                         label('geom')).\
            filter(self.orm['orm_lv_stations'].subst_id.in_(mv_grid_districts)). \
            filter(self.orm['version_condition_mvlvst'])

        # read data from db
        lv_grid_stations = pd.read_sql_query(lv_stations_sqla.statement,
                                             session.bind,
                                             index_col='mvlv_subst_id')
        return lv_grid_stations

    def import_generators(self, session, debug=False):
        """
        Imports renewable (res) and conventional (conv) generators

        Parameters
        ----------
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session
        debug: :obj:`bool`, defaults to False
            If True, information is printed during process

        Note
        -----
            Connection of generators is done later on in
            :class:`~.ding0.core.NetworkDing0`'s method
            :meth:`~.core.NetworkDing0.connect_generators`
        """

        def import_res_generators():
            """
            Imports renewable (res) generators
            """

            # build query
            generators_sqla = session.query(
                self.orm['orm_re_generators'].columns.id,
                self.orm['orm_re_generators'].columns.subst_id,
                self.orm['orm_re_generators'].columns.la_id,
                self.orm['orm_re_generators'].columns.mvlv_subst_id,
                self.orm['orm_re_generators'].columns.electrical_capacity,
                self.orm['orm_re_generators'].columns.generation_type,
                self.orm['orm_re_generators'].columns.generation_subtype,
                self.orm['orm_re_generators'].columns.voltage_level,
                self.orm['orm_re_generators'].columns.w_id,
                func.ST_AsText(func.ST_Transform(
                    self.orm['orm_re_generators'].columns.rea_geom_new, srid)).label('geom_new'),
                func.ST_AsText(func.ST_Transform(
                    self.orm['orm_re_generators'].columns.geom, srid)).label('geom')
            ). \
                filter(
                self.orm['orm_re_generators'].columns.subst_id.in_(list(mv_grid_districts_dict))). \
                filter(self.orm['orm_re_generators'].columns.voltage_level.in_([4, 5, 6, 7])). \
                filter(self.orm['version_condition_re'])

            # read data from db
            generators = pd.read_sql_query(generators_sqla.statement,
                                           session.bind,
                                           index_col='id')
            # define generators with unknown subtype as 'unknown'
            generators.loc[generators[
                               'generation_subtype'].isnull(),
                           'generation_subtype'] = 'unknown'

            for id_db, row in generators.iterrows():

                # treat generators' geom:
                # use geom_new (relocated genos from data processing)
                # otherwise use original geom from EnergyMap
                if row['geom_new']:
                    geo_data = wkt_loads(row['geom_new'])
                elif not row['geom_new']:
                    geo_data = wkt_loads(row['geom'])
                    logger.warning(
                        'Generator {} has no geom_new entry,'
                        'EnergyMap\'s geom entry will be used.'.format(
                        id_db))
                # if no geom is available at all, skip generator
                elif not row['geom']:
                    #geo_data =
                    logger.error('Generator {} has no geom entry either'
                                 'and will be skipped.'.format(id_db))
                    continue

                # look up MV grid
                mv_grid_district_id = row['subst_id']
                mv_grid = mv_grid_districts_dict[mv_grid_district_id].mv_grid

                # create generator object
                if row['generation_type'] in ['solar', 'wind']:
                    generator = GeneratorFluctuatingDing0(
                        id_db=id_db,
                        mv_grid=mv_grid,
                        capacity=row['electrical_capacity'],
                        type=row['generation_type'],
                        subtype=row['generation_subtype'],
                        v_level=int(row['voltage_level']),
                        weather_cell_id=row['w_id'])
                else:
                    generator = GeneratorDing0(
                        id_db=id_db,
                        mv_grid=mv_grid,
                        capacity=row['electrical_capacity'],
                        type=row['generation_type'],
                        subtype=row['generation_subtype'],
                        v_level=int(row['voltage_level']))

                # MV generators
                if generator.v_level in [4, 5]:
                    generator.geo_data = geo_data
                    mv_grid.add_generator(generator)

                # LV generators
                elif generator.v_level in [6, 7]:

                    # look up MV-LV substation id
                    mvlv_subst_id = row['mvlv_subst_id']

                    # if there's a LVGD id
                    if mvlv_subst_id and not isnan(mvlv_subst_id):
                        # assume that given LA exists
                        try:
                            # get LVGD
                            lv_station = lv_stations_dict[mvlv_subst_id]
                            lv_grid_district = lv_station.grid.grid_district
                            generator.lv_grid = lv_station.grid

                            # set geom (use original from db)
                            generator.geo_data = geo_data

                        # if LA/LVGD does not exist, choose random LVGD and move generator to station of LVGD
                        # this occurs due to exclusion of LA with peak load < 1kW
                        except:
                            lv_grid_district = random.choice(list(lv_grid_districts_dict.values()))

                            generator.lv_grid = lv_grid_district.lv_grid
                            generator.geo_data = lv_grid_district.lv_grid.station().geo_data

                            logger.warning('Generator {} cannot be assigned to '
                                           'non-existent LV Grid District and was '
                                           'allocated to a random LV Grid District ({}).'.format(
                                            repr(generator), repr(lv_grid_district)))
                            pass

                    else:
                        lv_grid_district = random.choice(list(lv_grid_districts_dict.values()))

                        generator.lv_grid = lv_grid_district.lv_grid
                        generator.geo_data = lv_grid_district.lv_grid.station().geo_data

                        logger.warning('Generator {} has no la_id and was '
                                       'assigned to a random LV Grid District ({}).'.format(
                                        repr(generator), repr(lv_grid_district)))

                    generator.lv_load_area = lv_grid_district.lv_load_area
                    lv_grid_district.lv_grid.add_generator(generator)

        def import_conv_generators():
            """
            Imports conventional (conv) generators
            """

            # build query
            generators_sqla = session.query(
                self.orm['orm_conv_generators'].columns.id,
                self.orm['orm_conv_generators'].columns.subst_id,
                self.orm['orm_conv_generators'].columns.name,
                self.orm['orm_conv_generators'].columns.capacity,
                self.orm['orm_conv_generators'].columns.fuel,
                self.orm['orm_conv_generators'].columns.voltage_level,
                func.ST_AsText(func.ST_Transform(
                    self.orm['orm_conv_generators'].columns.geom, srid)).label('geom')). \
                filter(
                self.orm['orm_conv_generators'].columns.subst_id.in_(list(mv_grid_districts_dict))). \
                filter(self.orm['orm_conv_generators'].columns.voltage_level.in_([4, 5, 6])). \
                filter(self.orm['version_condition_conv'])

            # read data from db
            generators = pd.read_sql_query(generators_sqla.statement,
                                           session.bind,
                                           index_col='id')

            for id_db, row in generators.iterrows():

                # look up MV grid
                mv_grid_district_id = row['subst_id']
                mv_grid = mv_grid_districts_dict[mv_grid_district_id].mv_grid

                # create generator object
                generator = GeneratorDing0(id_db=id_db,
                                           name=row['name'],
                                           geo_data=wkt_loads(row['geom']),
                                           mv_grid=mv_grid,
                                           capacity=row['capacity'],
                                           type=row['fuel'],
                                           subtype='unknown',
                                           v_level=int(row['voltage_level']))

                # add generators to graph
                if generator.v_level in [4, 5]:
                    mv_grid.add_generator(generator)
                # there's only one conv. geno with v_level=6 -> connect to MV grid
                elif generator.v_level in [6]:
                    generator.v_level = 5
                    mv_grid.add_generator(generator)

        # get ding0s' standard CRS (SRID)
        srid = str(int(cfg_ding0.get('geo', 'srid')))

        # get predefined random seed and initialize random generator
        seed = int(cfg_ding0.get('random', 'seed'))
        random.seed(a=seed)

        # build dicts to map MV grid district and Load Area ids to related objects
        mv_grid_districts_dict,\
        lv_load_areas_dict,\
        lv_grid_districts_dict,\
        lv_stations_dict = self.get_mvgd_lvla_lvgd_obj_from_id()

        # import renewable generators
        import_res_generators()

        # import conventional generators
        import_conv_generators()

        logger.info('=====> Generators imported')

    def import_config(self):
        """
        Loads parameters from config files

        Returns
        -------
        :obj:`dict`
            configuration key value pair dictionary
        """

        # load parameters from configs
        cfg_ding0.load_config('config_db_tables.cfg')
        cfg_ding0.load_config('config_calc.cfg')
        cfg_ding0.load_config('config_files.cfg')
        cfg_ding0.load_config('config_misc.cfg')

        cfg_dict = cfg_ding0.cfg._sections

        return cfg_dict

    def import_pf_config(self):
        """
        Creates power flow config class and imports config from file

        Returns
        -------
        :class:`~.ding0.core.powerflow.PFConfigDing0`
        """

        scenario = cfg_ding0.get("powerflow", "test_grid_stability_scenario")
        start_hour = int(cfg_ding0.get("powerflow", "start_hour"))
        end_hour = int(cfg_ding0.get("powerflow", "end_hour"))
        start_time = datetime(1970, 1, 1, 00, 00, 0)

        resolution = cfg_ding0.get("powerflow", "resolution")
        srid = str(int(cfg_ding0.get('geo', 'srid')))

        return PFConfigDing0(scenarios=[scenario],
                             timestep_start=start_time,
                             timesteps_count=end_hour-start_hour,
                             srid=srid,
                             resolution=resolution)

    def import_static_data(self):
        """
        Imports static data into NetworkDing0 such as equipment.

        Returns
        -------
        :obj: `dict`
            Dictionary with equipment data
        """

        package_path = ding0.__path__[0]

        static_data = {}

        equipment_mv_parameters_trafos = cfg_ding0.get('equipment',
                                                       'equipment_mv_parameters_trafos')
        static_data['MV_trafos'] = pd.read_csv(os.path.join(package_path, 'data',
                                   equipment_mv_parameters_trafos),
                                   comment='#',
                                   delimiter=',',
                                   decimal='.',
                                   converters={'S_nom': lambda x: int(x)})

        # import equipment
        equipment_mv_parameters_lines = cfg_ding0.get('equipment',
                                                      'equipment_mv_parameters_lines')
        static_data['MV_overhead_lines'] = pd.read_csv(os.path.join(package_path, 'data',
                                           equipment_mv_parameters_lines),
                                           comment='#',
                                           converters={'I_max_th': lambda x: int(x),
                                                       'U_n': lambda x: int(x),
                                                       'reinforce_only': lambda x: int(x)})

        equipment_mv_parameters_cables = cfg_ding0.get('equipment',
                                                       'equipment_mv_parameters_cables')
        static_data['MV_cables'] = pd.read_csv(os.path.join(package_path, 'data',
                                   equipment_mv_parameters_cables),
                                   comment='#',
                                   converters={'I_max_th': lambda x: int(x),
                                               'U_n': lambda x: int(x),
                                               'reinforce_only': lambda x: int(x)})

        equipment_lv_parameters_cables = cfg_ding0.get('equipment',
                                                       'equipment_lv_parameters_cables')
        static_data['LV_cables'] = pd.read_csv(os.path.join(package_path, 'data',
                                   equipment_lv_parameters_cables),
                                   comment='#',
                                   index_col='name',
                                   converters={'I_max_th': lambda x: int(x), 'U_n': lambda x: int(x)})

        equipment_lv_parameters_trafos = cfg_ding0.get('equipment',
                                                       'equipment_lv_parameters_trafos')
        static_data['LV_trafos'] = pd.read_csv(os.path.join(package_path, 'data',
                                   equipment_lv_parameters_trafos),
                                   comment='#',
                                   delimiter=',',
                                   decimal='.',
                                   converters={'S_nom': lambda x: int(x)})

        # import LV model grids
        model_grids_lv_string_properties = cfg_ding0.get('model_grids',
                                                         'model_grids_lv_string_properties')
        static_data['LV_model_grids_strings'] = pd.read_csv(os.path.join(package_path, 'data',
                                                model_grids_lv_string_properties),
                                                comment='#',
                                                delimiter=';',
                                                decimal=',',
                                                index_col='string_id',
                                                converters={'string_id': lambda x: int(x),
                                                            'type': lambda x: int(x),
                                                            'Kerber Original': lambda x: int(x),
                                                            'count house branch': lambda x: int(x),
                                                            'distance house branch': lambda x: int(x),
                                                            'cable width': lambda x: int(x),
                                                            'string length': lambda x: int(x),
                                                            'length house branch A': lambda x: int(x),
                                                            'length house branch B': lambda x: int(x),
                                                            'cable width A': lambda x: int(x),
                                                            'cable width B': lambda x: int(x)})

        model_grids_lv_apartment_string = cfg_ding0.get('model_grids',
                                                        'model_grids_lv_apartment_string')
        converters_ids = {}
        for id in range(1,47):  # create int() converter for columns 1..46
            converters_ids[str(id)] = lambda x: int(x)
        static_data['LV_model_grids_strings_per_grid'] = pd.read_csv(os.path.join(package_path, 'data',
                                                         model_grids_lv_apartment_string),
                                                         comment='#',
                                                         delimiter=';',
                                                         decimal=',',
                                                         index_col='apartment_count',
                                                         converters=dict({'apartment_count': lambda x: int(x)},
                                                                         **converters_ids))

        return static_data

    def import_orm(self):
        """
        Import ORM classes names for  the correct connection to
        open energy platform and access tables depending on input in config in
        self.config which is loaded from 'config_db_tables.cfg'

        Returns
        -------
        :obj: `dict`
            key value pairs of names of datasets versus
            sqlalchmey maps to acess
            various tables where the datasets
            used to build grids are stored on the open
            energy platform.
        """

        orm = {}

        data_source = self.config['input_data_source']['input_data']
        mv_grid_districts_name = self.config[data_source]['mv_grid_districts']
        mv_stations_name = self.config[data_source]['mv_stations']
        lv_load_areas_name = self.config[data_source]['lv_load_areas']
        lv_grid_district_name = self.config[data_source]['lv_grid_district']
        lv_stations_name = self.config[data_source]['lv_stations']
        conv_generators_name = self.config[data_source]['conv_generators']
        re_generators_name = self.config[data_source]['re_generators']

        from egoio.db_tables import model_draft as orm_model_draft, \
            supply as orm_supply, \
            demand as orm_demand, \
            grid as orm_grid

        if data_source == 'model_draft':
            orm['orm_mv_grid_districts'] = orm_model_draft.__getattribute__(mv_grid_districts_name)
            orm['orm_mv_stations'] = orm_model_draft.__getattribute__(mv_stations_name)
            orm['orm_lv_load_areas'] = orm_model_draft.__getattribute__(lv_load_areas_name)
            orm['orm_lv_grid_district'] = orm_model_draft.__getattribute__(lv_grid_district_name)
            orm['orm_lv_stations'] = orm_model_draft.__getattribute__(lv_stations_name)
            orm['orm_conv_generators'] = orm_model_draft.__getattribute__(conv_generators_name)
            orm['orm_re_generators'] = orm_model_draft.__getattribute__(re_generators_name)
            orm['version_condition_mvgd'] = 1 == 1
            orm['version_condition_mv_stations'] = 1 == 1
            orm['version_condition_la'] = 1 == 1
            orm['version_condition_lvgd'] = 1 == 1
            orm['version_condition_mvlvst'] = 1 == 1
            orm['version_condition_re'] = 1 == 1
            orm['version_condition_conv'] = 1 == 1
        elif data_source == 'versioned':
            orm['orm_mv_grid_districts'] = orm_grid.__getattribute__(mv_grid_districts_name)
            orm['orm_mv_stations'] = orm_grid.__getattribute__(mv_stations_name)
            orm['orm_lv_load_areas'] = orm_demand.__getattribute__(lv_load_areas_name)
            orm['orm_lv_grid_district'] = orm_grid.__getattribute__(lv_grid_district_name)
            orm['orm_lv_stations'] = orm_grid.__getattribute__(lv_stations_name)
            orm['orm_conv_generators'] = orm_supply.__getattribute__(conv_generators_name)
            orm['orm_re_generators'] = orm_supply.__getattribute__(re_generators_name)
            orm['data_version'] = self.config[data_source]['version']
            orm['version_condition_mvgd'] =\
                orm['orm_mv_grid_districts'].version == orm['data_version']
            orm['version_condition_mv_stations'] = \
                orm['orm_mv_stations'].version == orm['data_version']
            orm['version_condition_la'] =\
                orm['orm_lv_load_areas'].version == orm['data_version']
            orm['version_condition_lvgd'] =\
                orm['orm_lv_grid_district'].version == orm['data_version']
            orm['version_condition_mvlvst'] =\
                orm['orm_lv_stations'].version == orm['data_version']
            orm['version_condition_re'] =\
                orm['orm_re_generators'].columns.version == orm['data_version']
            orm['version_condition_conv'] =\
                orm['orm_conv_generators'].columns.version == orm['data_version']
        else:
            logger.error("Invalid data source {} provided. Please re-check the file "
                         "`config_db_tables.cfg`".format(data_source))
            raise NameError("{} is no valid data source!".format(data_source))

        return orm

    def validate_grid_districts(self):
        """
        Method to check the validity of the grid districts.
        MV grid districts are considered valid if:

            1. The number of nodes of the graph should be greater than 1
            2. All the load areas in the grid district are NOT tagged as
               aggregated load areas.

        Invalid MV grid districts are subsequently deleted from Network.
        """

        msg_invalidity = []
        invalid_mv_grid_districts = []

        for grid_district in self.mv_grid_districts():

            # there's only one node (MV station) => grid is empty
            if len(grid_district.mv_grid._graph.nodes()) == 1:
                invalid_mv_grid_districts.append(grid_district)
                msg_invalidity.append('MV Grid District {} seems to be empty ' \
                                      'and ' \
                                      'was removed'.format(grid_district))
            # there're only aggregated load areas
            elif all([lvla.is_aggregated for lvla in
                      grid_district.lv_load_areas()]):
                invalid_mv_grid_districts.append(grid_district)
                msg_invalidity.append("MV Grid District {} contains only " \
                                 "aggregated Load Areas and was removed" \
                                 "".format(grid_district))

        for grid_district in invalid_mv_grid_districts:
            self._mv_grid_districts.remove(grid_district)

        logger.warning("\n".join(msg_invalidity))
        logger.info('=====> MV Grids validated')
        return msg_invalidity

    def export_mv_grid(self, session, mv_grid_districts):
        """
        Exports MV grids to database for visualization purposes

        Parameters
        ----------
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session
        mv_grid_districts : :obj:`list` of
            :class:`~.ding0.core.structure.regions.MVGridDistrictDing0` objects
            whose MV grids are exported.
        """

        # check arguments
        if not all(isinstance(_, int) for _ in mv_grid_districts):
            raise TypeError('`mv_grid_districts` has to be a list of integers.')

        srid = str(int(cfg_ding0.get('geo', 'srid')))

        # delete all existing datasets
        # db_int.sqla_mv_grid_viz.__table__.create(conn) # create if not exist
        # change_owner_to(conn,
        #                 db_int.sqla_mv_grid_viz.__table_args__['schema'],
        #                 db_int.sqla_mv_grid_viz.__tablename__,
        #                 'oeuser')
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
                if isinstance(node, LVLoadAreaCentreDing0):
                    lv_load_area_centres.append((node.geo_data.x, node.geo_data.y))
                elif isinstance(node, MVCableDistributorDing0):
                    mv_cable_distributors.append((node.geo_data.x, node.geo_data.y))
                elif isinstance(node, MVStationDing0):
                    mv_stations.append((node.geo_data.x, node.geo_data.y))
                elif isinstance(node, CircuitBreakerDing0):
                    mv_circuit_breakers.append((node.geo_data.x, node.geo_data.y))
                elif isinstance(node, GeneratorDing0):
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

        # logger.info('=====> MV Grids exported')
        logger.info('MV Grids exported')

    def export_mv_grid_new(self, session, mv_grid_districts):
        """ Exports MV grids to database for visualization purposes

        Parameters
        ----------
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session
        mv_grid_districts : :obj:`list` of
            :class:`~.ding0.core.structure.regions.MVGridDistrictDing0` objects
            whose MV grids are exported.
        """

        # check arguments
        if not all(isinstance(_, int) for _ in mv_grid_districts):
            raise TypeError('`mv_grid_districts` has to be a list of integers.')

        srid = str(int(cfg_ding0.get('geo', 'srid')))

        # delete all existing datasets
        # db_int.sqla_mv_grid_viz_branches.__table__.create(conn) # create if not exist
        # change_owner_to(conn,
        #                 db_int.sqla_mv_grid_viz_branches.__table_args__['schema'],
        #                 db_int.sqla_mv_grid_viz_branches.__tablename__,
        #                 'oeuser')
        # db_int.sqla_mv_grid_viz_nodes.__table__.create(conn) # create if not exist
        # change_owner_to(conn,
        #                 db_int.sqla_mv_grid_viz_nodes.__table_args__['schema'],
        #                 db_int.sqla_mv_grid_viz_nodes.__tablename__,
        #                 'oeuser')
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
                # LA centres of agg. LA
                elif isinstance(node, LVLoadAreaCentreDing0):
                    if node.lv_load_area.is_aggregated:
                        node_name = '_'.join(['MV',
                                              str(grid_district.mv_grid.id_db),
                                              repr(node)])

                        node_dataset = db_int.sqla_mv_grid_viz_nodes(
                            node_id=node_name,
                            grid_id=grid_district.mv_grid.id_db,
                            v_nom=grid_district.mv_grid.v_level,
                            geom=from_shape(Point(node.geo_data), srid=srid),
                            v_res0=0,
                            v_res1=0
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
                else:
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
                        s_res0=0,
                        s_res1=0
                    )
                    session.add(branch_dataset)


        # commit changes to db
        session.commit()

        logger.info('=====> MV Grids exported (NEW)')

    def to_dataframe(self):
        """
        Export grid data to dataframes for statistical analysis.

        The export to dataframe is similar to db tables exported by `export_mv_grid_new`.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Pandas Data Frame

        See Also
        --------
        ding0.core.NetworkDing0.export_mv_grid_new :

        """

        node_cols = ['node_id', 'grid_id', 'v_nom', 'geom', 'v_res0', 'v_res1',
                     'peak_load', 'generation_capacity', 'type']
        edges_cols = ['branch_id', 'grid_id', 'type_name', 'type_kind',
                      'type_v_nom', 'type_s_nom', 'length', 'geom', 's_res0',
                      's_res1']

        nodes_df = pd.DataFrame(columns=node_cols)
        edges_df = pd.DataFrame(columns=edges_cols)

        srid = str(int(self.config['geo']['srid']))

        for grid_district in self.mv_grid_districts():

            # get nodes from grid's graph and create datasets
            for node in grid_district.mv_grid.graph_nodes_sorted():
                node_name = '_'.join(['MV',
                                      str(grid_district.mv_grid.id_db),
                                      repr(node)])
                geom = from_shape(Point(node.geo_data), srid=srid)
                if isinstance(node, LVStationDing0):
                    peak_load = node.peak_load
                    generation_capacity = node.peak_generation
                    if hasattr(node, 'voltage_res'):
                        type = 'LV Station'
                    else:
                        type = 'LV station (aggregated)'
                elif isinstance(node, GeneratorDing0):
                    peak_load = 0
                    generation_capacity = node.capacity
                    type = node.type
                elif isinstance(node, MVCableDistributorDing0):
                    peak_load = 0
                    generation_capacity = 0
                    type = 'Cable distributor'
                elif isinstance(node, LVLoadAreaCentreDing0):
                    peak_load = 0
                    generation_capacity = 0
                    type = 'Load area center of aggregated load area'
                elif isinstance(node, CircuitBreakerDing0):
                    peak_load = 0
                    generation_capacity = 0
                    type = 'Switch Disconnector'
                    #round coordinates of circuit breaker
                    geosjson = mapping(node.geo_data)
                    decimal_places = 4 #with 4, its around 10m error N-S or E-W
                    geosjson['coordinates'] = np.round(np.array(geosjson['coordinates']), decimal_places)
                    geom = from_shape(Point(shape(geosjson)), srid=srid)
                else:
                    peak_load = 0
                    generation_capacity = 0
                    type = 'Unknown'

                # add res voltages from nodes which were part of PF only
                if hasattr(node, 'voltage_res'):
                    v_res0 = node.voltage_res[0]
                    v_res1 = node.voltage_res[1]
                else:
                    v_res0 = v_res1 = 0

                nodes_df = nodes_df.append(pd.Series(
                    {'node_id': node_name,
                     'grid_id': grid_district.mv_grid.id_db,
                     'v_nom': grid_district.mv_grid.v_level,
                     'geom': geom,
                     'peak_load': peak_load,
                     'generation_capacity': generation_capacity,
                     'v_res0': v_res0,
                     'v_res1': v_res1,
                     'type': type,
                     'rings': len(grid_district.mv_grid._rings)
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
        """
        Performs routing on all MV grids.

        Parameters
        ----------
        debug: :obj:`bool`, default to False
            If True, information is printed while routing
        animation: :obj:`bool`, default to False
            If True, images of route modification
            steps are exported during routing process.
            A new animation object is created.

        See Also
        --------
        ding0.core.network.grids.MVGridDing0.routing : for details on MVGridDing0 objects routing
        ding0.tools.animation.AnimationDing0 : for details on animation function.
        """

        if animation:
            anim = AnimationDing0()
        else:
            anim = None

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.routing(debug=debug, anim=anim)

        logger.info('=====> MV Routing (Routing, Connection of Satellites & '
                    'Stations) performed')

    def build_lv_grids(self):
        """
        Builds LV grids for every non-aggregated LA in every MV grid
        district using model grids.
        """

        for mv_grid_district in self.mv_grid_districts():
            for load_area in mv_grid_district.lv_load_areas():
                if not load_area.is_aggregated:
                    for lv_grid_district in load_area.lv_grid_districts():

                        lv_grid_district.lv_grid.build_grid()
                else:
                    logger.info(
                        '{} is of type aggregated. No grid is created.'.format(repr(load_area)))

        logger.info('=====> LV model grids created')

    def connect_generators(self, debug=False):
        """
        Connects generators (graph nodes) to grid (graph) for every MV and LV Grid District

        Parameters
        ----------
        debug: :obj:`bool`, defaults to False
            If True, information is printed during process.
        """

        for mv_grid_district in self.mv_grid_districts():
            mv_grid_district.mv_grid.connect_generators(debug=debug)

            # get predefined random seed and initialize random generator
            seed = int(cfg_ding0.get('random', 'seed'))
            random.seed(a=seed)

            for load_area in mv_grid_district.lv_load_areas():
                if not load_area.is_aggregated:
                    for lv_grid_district in load_area.lv_grid_districts():

                        lv_grid_district.lv_grid.connect_generators(debug=debug)
                        if debug:
                            lv_grid_district.lv_grid.graph_draw(mode='LV')
                else:
                    logger.info(
                        '{} is of type aggregated. LV generators are not connected to LV grids.'.format(repr(load_area)))

        logger.info('=====> Generators connected')

    def mv_parametrize_grid(self, debug=False):
        """
        Performs Parametrization of grid equipment of all MV grids.

        Parameters
        ----------
        debug: :obj:bool, defaults to False
            If True, information is printed during process.

        See Also
        --------
        ding0.core.network.grids.MVGridDing0.parametrize_grid
        """

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.parametrize_grid(debug=debug)

        logger.info('=====> MV Grids parametrized')

    def set_branch_ids(self):
        """
        Performs generation and setting of ids
        of branches for all MV and underlying LV grids.

        See Also
        --------
        ding0.core.network.grids.MVGridDing0.set_branch_ids
        """

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.set_branch_ids()

        logger.info('=====> Branch IDs set')

    def set_circuit_breakers(self, debug=False):
        """
        Calculates the optimal position of the existing circuit breakers
        and relocates them within the graph for all MV grids.

        Parameters
        ----------
        debug: :obj:`bool`, defaults to False
            If True, information is printed during process

        See Also
        --------
        ding0.grid.mv_grid.tools.set_circuit_breakers

        """

        for grid_district in self.mv_grid_districts():
            grid_district.mv_grid.set_circuit_breakers(debug=debug)

        logger.info('=====> MV Circuit Breakers relocated')

    def control_circuit_breakers(self, mode=None):
        """
        Opens or closes all circuit breakers of all MV grids.

        Parameters
        ---------
        mode: :obj:`str`
            Set mode='open' to open, mode='close' to close
        """

        for grid_district in self.mv_grid_districts():
            if mode == 'open':
                grid_district.mv_grid.open_circuit_breakers()
            elif mode == 'close':
                grid_district.mv_grid.close_circuit_breakers()
            else:
                raise ValueError('\'mode\' is invalid.')

        if mode == 'open':
            logger.info('=====> MV Circuit Breakers opened')
        elif mode == 'close':
            logger.info('=====> MV Circuit Breakers closed')

    def run_powerflow(self, session, method='onthefly', export_pypsa=False, debug=False):
        """
        Performs power flow calculation for all MV grids

        Parameters
        ----------
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session

        method: :obj:`str`
            Specify export method
            If method='db' grid data will be exported to database

            If method='onthefly' grid data will be passed to PyPSA directly (default)

        export_pypsa: :obj:`bool`
            If True PyPSA networks will be exported as csv to output/debug/grid/<MV-GRID_NAME>/

        debug: :obj:`bool`, defaults to False
            If True, information is printed during process
        """

        if method == 'db':
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

        elif method == 'onthefly':
            for grid_district in self.mv_grid_districts():
                if export_pypsa:
                    export_pypsa_dir = repr(grid_district.mv_grid)
                else:
                    export_pypsa_dir = None
                grid_district.mv_grid.run_powerflow(session,
                                                    method='onthefly',
                                                    export_pypsa_dir=export_pypsa_dir,
                                                    debug=debug)

    def reinforce_grid(self):
        """
        Performs grid reinforcement measures for all MV and LV grids
        """
        # TODO: Finish method and enable LV case

        for grid_district in self.mv_grid_districts():

            # reinforce MV grid
            grid_district.mv_grid.reinforce_grid()

            # reinforce LV grids
            for lv_load_area in grid_district.lv_load_areas():
                if not lv_load_area.is_aggregated:
                    for lv_grid_district in lv_load_area.lv_grid_districts():
                        lv_grid_district.lv_grid.reinforce_grid()

    @property
    def metadata(self, run_id=None):
        """Provide metadata on a Ding0 run

        Parameters
        ----------
        run_id: :obj:`str`, (defaults to current date)
            Distinguish multiple versions of Ding0 data by a `run_id`. If not
            set it defaults to current date in the format YYYYMMDDhhmmss

        Returns
        -------
        :obj:`dict`
            Metadata
        """

        # Get latest version and/or git commit hash
        try:
            version = subprocess.check_output(
                ["git", "describe", "--tags", "--always"]).decode('utf8')
        except:
            version = None

        # Collect names of database table used to run Ding0 and data version
        if self.config['input_data_source']['input_data'] == 'versioned':
            data_version = self.config['versioned']['version']
            database_tables = self.config['versioned']
        elif self.config['input_data_source']['input_data'] == 'model_draft':
            data_version = 'model_draft'
            database_tables = self.config['model_draft']
        else:
            data_version = 'unknown'
            database_tables = 'unknown'

        # Collect assumptions
        assumptions = {}
        assumptions.update(self.config['assumptions'])
        assumptions.update(self.config['mv_connect'])
        assumptions.update(self.config['mv_routing'])
        assumptions.update(self.config['mv_routing_tech_constraints'])

        # Determine run_id if not set
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d%H%M%S")

        # Set instance attribute run_id
        if not self._run_id:
            self._run_id = run_id

        # Assing data to dict
        metadata = dict(
            version=version,
            mv_grid_districts=[int(_.id_db) for _ in self._mv_grid_districts],
            database_tables=database_tables,
            data_version=data_version,
            assumptions=assumptions,
            run_id=self._run_id
        )

        return metadata


    def __repr__(self):
        """"
        A repr string representation of the NetworkDing0 object.
        This prints out only the name attribute.
        """
        return str(self.name)

    def list_generators(self, session):
        """
        List renewable (res) and conventional (conv) generators

        Parameters
        ----------
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            A table containing the generator data, the columns being:
            - subst_id,
            - la_id (load area id),
            - mv_lv_subst_id (id of the mv lv substation),
            - electrical_capacity
            - generation_type
            - generation_subtype
            - voltage_level
            - geospatial coordinates as :shapely:`Shapely Point object<point>`
        """
        srid = str(int(cfg_ding0.get('geo', 'srid')))

        # build dicts to map MV grid district and Load Area ids to related objects
        mv_grid_districts_dict,\
        lv_load_areas_dict,\
        lv_grid_districts_dict,\
        lv_stations_dict = self.get_mvgd_lvla_lvgd_obj_from_id()

        # import renewable generators
        # build query
        generators_sqla = session.query(
                self.orm['orm_re_generators'].columns.id,
                self.orm['orm_re_generators'].columns.subst_id,
                self.orm['orm_re_generators'].columns.la_id,
                self.orm['orm_re_generators'].columns.mvlv_subst_id,
                self.orm['orm_re_generators'].columns.electrical_capacity,
                self.orm['orm_re_generators'].columns.generation_type,
                self.orm['orm_re_generators'].columns.generation_subtype,
                self.orm['orm_re_generators'].columns.voltage_level,
                func.ST_AsText(func.ST_Transform(
                   self.orm['orm_re_generators'].columns.rea_geom_new, srid)).label('geom_new'),
                func.ST_AsText(func.ST_Transform(
                   self.orm['orm_re_generators'].columns.geom, srid)).label('geom')
                ).filter(
                self.orm['orm_re_generators'].columns.subst_id.in_(list(mv_grid_districts_dict))). \
                filter(self.orm['orm_re_generators'].columns.voltage_level.in_([4, 5, 6, 7])). \
                filter(self.orm['version_condition_re'])

        # read data from db
        generators_res = pd.read_sql_query(generators_sqla.statement,
                                        session.bind,
                                        index_col='id')

        generators_res.columns = ['GenCap' if c=='electrical_capacity' else
                                  'type' if c=='generation_type' else
                                  'subtype' if c=='generation_subtype' else
                                  'v_level' if c=='voltage_level' else
                                  c for c in generators_res.columns]
        ###########################
        # Imports conventional (conv) generators
        # build query
        generators_sqla = session.query(
                self.orm['orm_conv_generators'].columns.id,
                self.orm['orm_conv_generators'].columns.subst_id,
                self.orm['orm_conv_generators'].columns.name,
                self.orm['orm_conv_generators'].columns.capacity,
                self.orm['orm_conv_generators'].columns.fuel,
                self.orm['orm_conv_generators'].columns.voltage_level,
                func.ST_AsText(func.ST_Transform(
                   self.orm['orm_conv_generators'].columns.geom, srid)).label('geom')
                ).filter(
                self.orm['orm_conv_generators'].columns.subst_id.in_(list(mv_grid_districts_dict))). \
                filter(self.orm['orm_conv_generators'].columns.voltage_level.in_([4, 5, 6])). \
                filter(self.orm['version_condition_conv'])

        # read data from db
        generators_conv = pd.read_sql_query(generators_sqla.statement,
                                           session.bind,
                                           index_col='id')

        generators_conv.columns = ['GenCap' if c=='capacity' else
                                   'type' if c=='fuel' else
                                   'v_level' if c=='voltage_level' else
                                   c for c in generators_conv.columns]
        ###########################
        generators = pd.concat([generators_conv, generators_res], axis=0)
        generators = generators.fillna('other')
        return generators

    def list_load_areas(self, session, mv_districts):
        """list load_areas (load areas) peak load from database for a single MV grid_district

        Parameters
        ----------
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session
        mv_districts: :obj:`list` of
            :class:`~.ding0.core.structure.regions.MVGridDistrictDing0` objects
        """

        # threshold: load area peak load, if peak load < threshold => disregard
        # load area
        lv_loads_threshold = cfg_ding0.get('mv_routing', 'load_area_threshold')
        #lv_loads_threshold = 0

        gw2kw = 10 ** 6  # load in database is in GW -> scale to kW

        #filter list for only desired MV districts
        stations_list = [d.mv_grid._station.id_db for d in mv_districts]

        # build SQL query
        lv_load_areas_sqla = session.query(
            self.orm['orm_lv_load_areas'].id.label('id_db'),
            (self.orm['orm_lv_load_areas'].sector_peakload_residential * gw2kw).\
                label('peak_load_residential'),
            (self.orm['orm_lv_load_areas'].sector_peakload_retail * gw2kw).\
                label('peak_load_retail'),
            (self.orm['orm_lv_load_areas'].sector_peakload_industrial * gw2kw).\
                label('peak_load_industrial'),
            (self.orm['orm_lv_load_areas'].sector_peakload_agricultural * gw2kw).\
                label('peak_load_agricultural'),
            #self.orm['orm_lv_load_areas'].subst_id
            ). \
            filter(self.orm['orm_lv_load_areas'].subst_id.in_(stations_list)).\
            filter(((self.orm['orm_lv_load_areas'].sector_peakload_residential  # only pick load areas with peak load > lv_loads_threshold
                     + self.orm['orm_lv_load_areas'].sector_peakload_retail
                     + self.orm['orm_lv_load_areas'].sector_peakload_industrial
                     + self.orm['orm_lv_load_areas'].sector_peakload_agricultural)
                       * gw2kw) > lv_loads_threshold). \
            filter(self.orm['version_condition_la'])

        # read data from db
        lv_load_areas = pd.read_sql_query(lv_load_areas_sqla.statement,
                                          session.bind,
                                          index_col='id_db')

        return lv_load_areas

    def list_lv_grid_districts(self, session, lv_stations):
        """Imports all lv grid districts within given load area

        Parameters
        ----------
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session
        lv_stations: :obj:`list`
            List required LV_stations==LV districts.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Pandas Data Frame
            Table of lv_grid_districts
        """
        gw2kw = 10 ** 6  # load in database is in GW -> scale to kW

        # 1. filter grid districts of relevant load area
        lv_grid_districs_sqla = session.query(
            self.orm['orm_lv_grid_district'].mvlv_subst_id,
            (self.orm[
                 'orm_lv_grid_district'].sector_peakload_residential * gw2kw).
                label('peak_load_residential'),
            (self.orm['orm_lv_grid_district'].sector_peakload_retail * gw2kw).
                label('peak_load_retail'),
            (self.orm[
                 'orm_lv_grid_district'].sector_peakload_industrial * gw2kw).
                label('peak_load_industrial'),
            (self.orm[
                 'orm_lv_grid_district'].sector_peakload_agricultural * gw2kw).
                label('peak_load_agricultural'),
            ). \
            filter(self.orm['orm_lv_grid_district'].mvlv_subst_id.in_(
            lv_stations)). \
            filter(self.orm['version_condition_lvgd'])

        # read data from db
        lv_grid_districts = pd.read_sql_query(lv_grid_districs_sqla.statement,
                                              session.bind,
                                              index_col='mvlv_subst_id')

        return lv_grid_districts
