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


import os
from . import RegionDing0
from ding0.tools import config as cfg_ding0
# from ding0.core.network.loads import MVLoadDing0

if not 'READTHEDOCS' in os.environ:
    from shapely.wkt import loads as wkt_loads


class MVGridDistrictDing0(RegionDing0):
        # TODO: check docstring
    """Defines a MV-grid_district in DfINGO
    
    Attributes
    ----------
    mv_grid: :obj:`int`
       Descr
    geo_data : :shapely:`Shapely Polygon object<polygons>`
        The geo-spatial Polygon in the coordinate reference
        system with the SRID:4326 or epsg:4326, this
        is the project used by the ellipsoid WGS 84.
    peak_load: :obj:`float`
       Descr
    peak_load_satellites: :obj:`float`
       Descr
    peak_load_aggregated: :obj:`float`
       Descr
    """

    def __init__(self, **kwargs):
        #inherit branch parameters from Region
        super().__init__(**kwargs)

        #more params
        self.mv_grid = kwargs.get('mv_grid', None)
        self._lv_load_areas = []
        self._lv_load_area_groups = []
        self.geo_data = kwargs.get('geo_data', None)

        # peak load sum in kVA
        self.peak_load = kwargs.get('peak_load', 0)

        # peak load (satellites only) in kVA
        self.peak_load_satellites = kwargs.get('peak_load_satellites', 0)

        # peak load (aggregated only) in kVA
        self.peak_load_aggregated = kwargs.get('peak_load_aggregated', 0)

    @property
    def network(self):
        return self.mv_grid.network

    def lv_load_areas(self):
        """Returns a generator for iterating over load_areas
        
        Yields
        ------
        int
            generator for iterating over load_areas
        """
        for load_area in sorted(self._lv_load_areas, key=lambda _: repr(_)):
            yield load_area

    def add_lv_load_area(self, lv_load_area):
        """ Adds a Load Area `lv_load_area` to _lv_load_areas if not already existing
        
        Additionally, adds the associated centre object to MV grid's _graph as node.

        Parameters
        ----------
        lv_load_area: LVLoadAreaDing0
            instance of class LVLoadAreaDing0
        """
        if lv_load_area not in self.lv_load_areas() and isinstance(lv_load_area, LVLoadAreaDing0):
            self._lv_load_areas.append(lv_load_area)
            self.mv_grid.graph_add_node(lv_load_area.lv_load_area_centre)

    def lv_load_area_groups(self):
        """Returns a generator for iterating over LV load_area groups.
        
        Yields
        ------
        int
            generator for iterating over LV load_areas
        """
        for lv_load_area_group in self._lv_load_area_groups:
            yield lv_load_area_group

    def lv_load_area_groups_count(self):
        """Returns the count of LV load_area groups in MV region
        
        Returns
        -------
        int
            Number of LV load_area groups in MV region.
        """
        return len(self._lv_load_area_groups)

    def add_lv_load_area_group(self, lv_load_area_group):
        """Adds a LV load_area to _lv_load_areas if not already existing.
        """
        if lv_load_area_group not in self.lv_load_area_groups():
            self._lv_load_area_groups.append(lv_load_area_group)

    def add_peak_demand(self):
        """Summarizes peak loads of underlying load_areas in kVA.
        
        (peak load sum and peak load of satellites)
        """
        peak_load = peak_load_satellites = 0
        
        for lv_load_area in self.lv_load_areas():
            peak_load += lv_load_area.peak_load
            if lv_load_area.is_satellite:
                peak_load_satellites += lv_load_area.peak_load
        # PAUL new add mv load to peak load of mvgd    
        for mv_load in self.mv_grid._loads:
            peak_load += mv_load.peak_load
            
        self.peak_load = peak_load
        self.peak_load_satellites = peak_load_satellites

    def add_aggregated_peak_demand(self):
        """Summarizes peak loads of underlying aggregated load_areas"""
        peak_load_aggregated = 0
        for lv_load_area in self.lv_load_areas():
            if lv_load_area.is_aggregated:
                peak_load_aggregated += lv_load_area.peak_load
        self.peak_load_aggregated = peak_load_aggregated

    def __repr__(self):
        return 'mv_grid_district_' + str(self.id_db)


class LVLoadAreaDing0(RegionDing0):
        # TODO: check docstring
    """Defines a LV-load_area in DINGO

    Attributes
    ----------
    ring: :obj:`int`
       Descr
    mv_grid_district : :shapely:`Shapely Polygon object<polygons>`
       Descr
    lv_load_area_centre: :shapely:`Shapely Point object<points>`
       Descr
    lv_load_area_group: :shapely:`Shapely Polygon object<polygons>`
       Descr
    is_satellite: :obj:`bool`
       Descr
    is_aggregated: :obj:`bool`
       Descr
    db_data: :pandas:`pandas.DatetimeIndex<datetimeindex>`
       Descr

    # new osm approach
    load_area_graph: networkx.MultiDiGraph
        contains all streets in load_area
    MVLoads: list
        list containing ding0.MVLoads in lvla
    """

    def __init__(self, **kwargs):
        # inherit branch parameters from Region
        super().__init__(**kwargs)

        # more params
        self._lv_grid_districts = []
        self._mv_loads = []
        self.ring = kwargs.get('ring', None)
        self.mv_grid_district = kwargs.get('mv_grid_district', None)
        self.lv_load_area_centre = kwargs.get('lv_load_area_centre', None)
        self.lv_load_area_group = kwargs.get('lv_load_area_group', None)
        self.is_satellite = kwargs.get('is_satellite', False)
        self.is_aggregated = kwargs.get('is_aggregated', False)

        # threshold: load area peak load, if peak load < threshold => treat load area as satellite
        load_area_sat_load_threshold = cfg_ding0.get('mv_connect', 'load_area_sat_load_threshold')
        # TODO: Value is read from file every time a LV load_area is created -> move to associated NetworkDing0 class?

        db_data = kwargs.get('db_data', None)

        # dangerous: attributes are created for any passed argument in `db_data`
        # load values into attributes
        if db_data is not None:
            for attribute in list(db_data.keys()):
                setattr(self, attribute, db_data[attribute])

        # convert geo attributes to to shapely objects
        if hasattr(self, 'geo_area'):
            self.geo_area = wkt_loads(self.geo_area)
        if hasattr(self, 'geo_centre'):
            self.geo_centre = wkt_loads(self.geo_centre)

        # convert load values (rounded floats) to int
        if hasattr(self, 'peak_load_residential'):
            self.peak_load_residential = self.peak_load_residential
        if hasattr(self, 'peak_load_retail'):
            self.peak_load_retail = self.peak_load_retail
        if hasattr(self, 'peak_load_industrial'):
            self.peak_load_industrial = self.peak_load_industrial
        if hasattr(self, 'peak_load_agricultural'):
            self.peak_load_agricultural = self.peak_load_agricultural
        
        if hasattr(self, 'peak_load'):
            self.peak_load = self.peak_load
            
        self.peak_load = kwargs.get('peak_load', None)
        self.load_area_graph = kwargs.get('load_area_graph', False)

        # if load area has got a peak load less than load_area_sat_threshold, it's a satellite
        if self.peak_load < load_area_sat_load_threshold:
            self.is_satellite = True

    @property
    def network(self):
        return self.mv_grid_district.network

    def lv_grid_districts(self):
        """Returns a generator for iterating over LV grid districts
        
        Yields
        ------
        int
            generator for iterating over LV grid districts
        """
        for lv_grid_district in sorted(self._lv_grid_districts, key=lambda _: repr(_)):
            yield lv_grid_district

    def lv_grid_districts_count(self):
        """Returns the count of LV grid districts
        
        Returns
        -------
        int
            Number of LV grid districts.
        """
        return len(self._lv_grid_districts)

    def mv_loads_count(self):
        """Returns the count of MV loads

        Returns
        -------
        int
            Number of MV loads.
        """
        return len(self._mv_loads)

    def add_mv_load(self, mv_load):
        """Adds a MVLoad to _lv_grid_districts if not already existing
        
        Parameters
        ----------
        mv_load: ding0.core.network.MVLoadDing0
        """

        if mv_load not in self._mv_loads:
        # TODO: CHECK IF ITS AN MV LOAD
        # isinstance(mv_load, MVLoadDing0):
            self._mv_loads.append(mv_load)

    def add_lv_grid_district(self, lv_grid_district):
        """Adds a LV grid district to _lv_grid_districts if not already existing
        
        Parameters
        ----------
        lv_grid_district: :shapely:`Shapely Polygon object<polygons>`
            Descr
        """

        if lv_grid_district not in self._lv_grid_districts and \
                isinstance(lv_grid_district, LVGridDistrictDing0):
            self._lv_grid_districts.append(lv_grid_district)

    @property
    def peak_generation(self):
        """Cumulative peak generation of generators connected to LV grids of 
        underlying LVGDs
        """
        cum_peak_generation = 0

        for lv_grid_district in self._lv_grid_districts:
            cum_peak_generation += lv_grid_district.lv_grid.station().peak_generation

        return cum_peak_generation

    def __repr__(self):
        return 'lv_load_area_' + str(self.id_db)


class LVLoadAreaCentreDing0:
    # TODO: check docstring
    """
    Defines a region centre in Ding0.
    
    The centres are used in the MV routing as nodes.
    
    Note
    -----
    Centre is a point within a region's polygon that is located most central 
    (e.g. in a simple region shape like a circle it's the geometric center).

    Parameters
    ----------
    id_db: :obj:`int`
        unique ID in database (=id of associated load area)
    grid: :obj:`int`
        Descr
    geo_data: :shapely:`Shapely Point object<points>`
        The geo-spatial point in the coordinate reference
        system with the SRID:4326 or epsg:4326, this
        is the project used by the ellipsoid WGS 84.
    lv_load_area: :class:`~.ding0.core.network.regions.LVLoadAreaDing0`
        Descr
    """
    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.grid = kwargs.get('grid', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.lv_load_area = kwargs.get('lv_load_area', None)
        # PAUL new
        self.osm_id_node = kwargs.get('osm_id_node', None)
        

    @property
    def network(self):
        return self.lv_load_area.network

    @property
    def pypsa_bus_id(self):
        """Todo: Remove
        Returns specific ID for representing bus in pypsa network.

        Returns
        -------
        :obj:`str`:
            Representative of pypsa bus
        """
        return '_'.join(['Bus','mvgd', str(self.grid.id_db), 'lac', str(self.id_db)])

    def __repr__(self):
        return '_'.join(['LVLoadAreaCentre',  'mvgd', str(
                self.grid.id_db), str(self.id_db)])

class LVGridDistrictDing0(RegionDing0):
    # TODO: check docstring
    """Describes region that is covered by a single LV grid

    Parameters
    ----------
    geo_data: :shapely:`Shapely Polygon object<polygons>`
        The geo-spatial polygon in the coordinate reference
        system with the SRID:4326 or epsg:4326, this
        is the project used by the ellipsoid WGS 84.
    lv_load_area : :shapely:`Shapely Polygon object<polygons>`
       Descr
    lv_grid: :shapely:`Shapely Polygon object<polygons>`
       Descr
    population: :obj:`float`
       Descr
    peak_load_residential: :obj:`float`
       Descr
    peak_load_retail: :obj:`float`
       Descr
    peak_load_industrial: :obj:`float`
       Descr
    peak_load_agricultural: :obj:`float`
       Descr
    peak_load: :obj:`float`
       Descr
    sector_count_residential: :obj:`int`
       Descr
    sector_count_retail: :obj:`int`
       Descr
    sector_count_industrial: :obj:`int`
       Descr
    sector_count_agricultural: :obj:`int`
       Descr
       
       
    TODO UPDATE DESCR FOR GRAPH AND BUILDINGS
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.geo_data = kwargs.get('geo_data', None)
        self.lv_load_area = kwargs.get('lv_load_area', None)
        self.lv_grid = kwargs.get('lv_grid', None)

        self.population = kwargs.get('population', None)

        self.peak_load_residential = kwargs.get('peak_load_residential', None)
        self.peak_load_retail = kwargs.get('peak_load_retail', None)
        self.peak_load_industrial = kwargs.get('peak_load_industrial', None)
        self.peak_load_agricultural = kwargs.get('peak_load_agricultural', None)
        self.peak_load = kwargs.get('peak_load', None)

        self.sector_count_residential = kwargs.get('sector_count_residential',
                                                   None)
        self.sector_count_retail = kwargs.get('sector_count_retail', None)
        self.sector_count_industrial = kwargs.get('sector_count_industrial',
                                                  None)
        self.sector_count_agricultural = kwargs.get('sector_count_agricultural',
                                                    None)

        self.sector_consumption_residential = kwargs.get(
            'sector_consumption_residential',
            None)
        self.sector_consumption_retail = kwargs.get('sector_consumption_retail',
                                                    None)
        self.sector_consumption_industrial = kwargs.get(
            'sector_consumption_industrial',
            None)
        self.sector_consumption_agricultural = kwargs.get(
            'sector_consumption_agricultural',
            None)
        
        # todo: do doc
        self.mvlv_subst_id = kwargs.get('mvlv_subst_id', None)
        self.graph_district = kwargs.get('graph_district', None)
        self.buildings = kwargs.get('buildings_district', None)
        self.peak_load = kwargs.get('peak_load', None)
        

    @property
    def network(self):
        return self.lv_load_area.network

    def __repr__(self):
        return 'lv_grid_district_' + str(self.id_db)
