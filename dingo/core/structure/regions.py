from . import RegionDingo
from dingo.tools import config as cfg_dingo

from shapely.wkt import loads as wkt_loads


class MVGridDistrictDingo(RegionDingo):
    """
    Defines a MV-grid_district in DINGO
    ----------------------------

    """
    # TODO: add method remove_lv_load_area()

    def __init__(self, **kwargs):
        #inherit branch parameters from Region
        super().__init__(**kwargs)

        #more params
        self.mv_grid = kwargs.get('mv_grid', None)
        self._lv_load_areas = []
        self._lv_load_area_groups = []
        self.geo_data = kwargs.get('geo_data', None)

        # INSERT LOAD PARAMS
        self.peak_load = kwargs.get('peak_load', None)  # in kVA
        self.peak_load_satellites = kwargs.get('peak_load_satellites', None)  # in kVA
        self.peak_load_aggregated = kwargs.get('peak_load_aggregated', None)  # in kVA

    def lv_load_areas(self):
        """Returns a generator for iterating over load_areas"""
        for load_area in sorted(self._lv_load_areas, key=lambda _: repr(_)):
            yield load_area

    def add_lv_load_area(self, lv_load_area):
        """ Adds a Load Area `lv_load_area` to _lv_load_areas if not already existing, and adds the associated centre
            object to MV grid's _graph as node.

        Args:
            lv_load_area: instance of class LVLoadAreaDingo

        Returns:
            nothing
        """
        if lv_load_area not in self.lv_load_areas() and isinstance(lv_load_area, LVLoadAreaDingo):
            self._lv_load_areas.append(lv_load_area)
            self.mv_grid.graph_add_node(lv_load_area.lv_load_area_centre)

    def lv_load_area_groups(self):
        """Returns a generator for iterating over LV load_area groups"""
        for lv_load_area_group in self._lv_load_area_groups:
            yield lv_load_area_group

    def lv_load_area_groups_count(self):
        """Returns the count of LV load_area groups in MV region"""
        return len(self._lv_load_area_groups)

    def add_lv_load_area_group(self, lv_load_area_group):
        """Adds a LV load_area to _lv_load_areas if not already existing"""
        if lv_load_area_group not in self.lv_load_area_groups():
            self._lv_load_area_groups.append(lv_load_area_group)

    def add_peak_demand(self):
        """Summarizes peak loads of underlying load_areas in kVA (peak load sum and peak load of satellites)"""
        peak_load = peak_load_satellites = 0
        for lv_load_area in self.lv_load_areas():
            peak_load += lv_load_area.peak_load
            if lv_load_area.is_satellite:
                peak_load_satellites += lv_load_area.peak_load
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


class LVLoadAreaDingo(RegionDingo):
    """
    Defines a LV-load_area in DINGO
    ----------------------------

    """

    def __init__(self, **kwargs):
        # inherit branch parameters from Region
        super().__init__(**kwargs)

        # more params
        self._lv_grid_districts = []
        self.ring = kwargs.get('ring', None)
        self.mv_grid_district = kwargs.get('mv_grid_district', None)
        self.lv_load_area_centre = kwargs.get('lv_load_area_centre', None)
        self.lv_load_area_group = kwargs.get('lv_load_area_group', None)
        self.is_satellite = kwargs.get('is_satellite', False)
        self.is_aggregated = kwargs.get('is_aggregated', False)

        # threshold: load area peak load, if peak load < threshold => treat load area as satellite
        load_area_sat_load_threshold = cfg_dingo.get('mv_connect', 'load_area_sat_load_threshold')
        # TODO: Value is read from file every time a LV load_area is created -> move to associated NetworkDingo class?

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

            # if load area has got a peak load less than load_area_sat_threshold, it's a satellite
            if self.peak_load < load_area_sat_load_threshold:
                self.is_satellite = True

    def lv_grid_districts(self):
        """Returns a generator for iterating over LV grid districts"""
        for lv_grid_district in sorted(self._lv_grid_districts, key=lambda _: repr(_)):
            yield lv_grid_district

    def lv_grid_districts_count(self):
        """Returns the count of LV grid districts"""
        return len(self._lv_grid_districts)

    def add_lv_grid_district(self, lv_grid_district):
        """Adds a LV grid district to _lv_grid_districts if not already existing"""

        if lv_grid_district not in self._lv_grid_districts and \
                isinstance(lv_grid_district, LVGridDistrictDingo):
            self._lv_grid_districts.append(lv_grid_district)

    @property
    def peak_generation(self):
        """
        Cumulative peak generation of generators connected to LV grids of underlying LVGDs
        """
        cum_peak_generation = 0

        for lv_grid_district in self._lv_grid_districts:
            cum_peak_generation += lv_grid_district.lv_grid.station().peak_generation

        return cum_peak_generation

    def __repr__(self):
        return 'lv_load_area_' + str(self.id_db)


class LVLoadAreaCentreDingo:
    """
    Defines a region centre in Dingo
    --------------------------------
    The centres are used in the MV routing as nodes.
    Note: Centre is a point within a region's polygon that is located most central (e.g. in a simple region shape like a
    circle it's the geometric center).

    Parameters
    ----------
    id_db: unique ID in database (=id of associated load area)
    """
    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.grid = kwargs.get('grid', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.lv_load_area = kwargs.get('lv_load_area', None)

    @property
    def pypsa_id(self):
        return '_'.join(['MV', str(self.grid.id_db), 'lac', str(self.id_db)])

    def __repr__(self):
        return 'lv_load_area_centre_' + str(self.id_db)


class LVGridDistrictDingo(RegionDingo):
    """
    Describes region that is covered by a single LV grid

    Parameters
    ----------
    RegionDingo: class
        Dingo's region base class
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

    def __repr__(self):
        return 'lv_grid_district_' + str(self.id_db)