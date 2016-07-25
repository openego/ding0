from . import RegionDingo
#from dingo.core.network.grids import LVGridDingo
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

    def lv_load_areas(self):
        """Returns a generator for iterating over load_areas"""
        for load_area in self._lv_load_areas:
            yield load_area

    def add_lv_load_area(self, lv_load_area):
        """ Adds a LV load area `lv_load_area` to _lv_load_areas if not already existing, and adds the associated centre
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
        """Summarizes peak loads of underlying load_areas in kVA"""
        peak_load = 0
        for lv_load_area in self.lv_load_areas():
            peak_load += lv_load_area.peak_load_sum
        self.peak_load = peak_load

    def __repr__(self):
        return 'mv_grid_district_' + str(self.id_db)


class LVLoadAreaDingo(RegionDingo):
    """
    Defines a LV-load_area in DINGO
    ----------------------------

    """
    # TODO: add method remove_lv_grid()

    def __init__(self, **kwargs):
        # inherit branch parameters from Region
        super().__init__(**kwargs)

        # more params
        self._lv_grids = []     # TODO: add setter
        self.mv_grid_district = kwargs.get('mv_grid_district', None)
        self.lv_load_area_centre = kwargs.get('lv_load_area_centre', None)
        self.lv_load_area_group = kwargs.get('lv_load_area_group', None)
        self.is_satellite = kwargs.get('is_satellite', False)

        # threshold: load area peak load, if peak load < threshold => treat load area as satellite
        load_area_sat_load_threshold = cfg_dingo.get('mv_connect', 'load_area_sat_load_threshold')
        # TODO: Value is read from file every time a LV load_area is created -> move to associated NetworkDingo class?

        db_data = kwargs.get('db_data', None)

        # TODO: Choose good argument handling (add any given attribute (OPTION 1) vs. list of args (OPTION 2), see below)

        # OPTION 1
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
            self.peak_load_residential = int(self.peak_load_residential)
        if hasattr(self, 'peak_load_retail'):
            self.peak_load_retail = int(self.peak_load_retail)
        if hasattr(self, 'peak_load_industrial'):
            self.peak_load_industrial = int(self.peak_load_industrial)
        if hasattr(self, 'peak_load_agricultural'):
            self.peak_load_agricultural = int(self.peak_load_agricultural)
        if hasattr(self, 'peak_load_sum'):
            self.peak_load_sum = int(self.peak_load_sum)

            # if load area has got a peak load less than load_area_sat_threshold, it's a satellite
            if self.peak_load_sum < load_area_sat_load_threshold:
                self.is_satellite = True

        # Alternative to version above:
        # many params, use better structure (dict? classes from demand-lib?)


        # for attribute in ['geo_area',
        #                   'geo_centre',
        #                   'area',
        #                   'nuts_code',
        #                   'zensus_sum',
        #                   'zensus_cnt',
        #                   'ioer_sum',
        #                   'ioer_cnt',
        #                   'sector_area_residential',
        #                   'sector_area_retail',
        #                   'sector_area_industrial',
        #                   'sector_area_agricultural',
        #                   'sector_share_residential',
        #                   'sector_share_retail',
        #                   'sector_share_industrial',
        #                   'sector_share_agricultural',
        #                   'sector_count_residential',
        #                   'sector_count_retail',
        #                   'sector_count_industrial',
        #                   'sector_count_agricultural']:
        #     setattr(self, attribute, kwargs.get(attribute, None))

        # self.geo_area = kwargs.get('geo_area', None)
        # self.geo_centre= kwargs.get('geo_centroid', None)
        #
        # self.area = kwargs.get('area', None)
        # self.nuts_code = kwargs.get('nuts_code', None)
        #
        # self.zensus_sum = kwargs.get('zensus_sum', None)
        # self.zensus_cnt = kwargs.get('zensus_cnt', None)
        # self.ioer_sum = kwargs.get('ioer_sum', None)
        # self.ioer_cnt = kwargs.get('ioer_cnt', None)

        # self.sector_area_residential =
        # self.sector_area_retail =
        # self.sector_area_industrial =
        # self.sector_area_agricultural =
        # self.sector_share_residential =
        # self.sector_share_retail =
        # self.sector_share_industrial =
        # self.sector_share_agricultural =
        # self.sector_count_residential =
        # self.sector_count_retail =
        # self.sector_count_industrial =
        # self.sector_count_agricultural =

        #self.sector_consumption_residential =
        #self.sector_consumption_retail =
        #self.sector_consumption_industrial =
        #self.sector_consumption_agricultural =

    def lv_grids(self):
        """Returns a generator for iterating over LV grids"""
        for grid in self._lv_grids:
            yield grid

    def add_lv_grid(self, lv_grid):
        """Adds a LV grid to _lv_grids if not already existing"""
        if lv_grid not in self.lv_grids():  # and isinstance(lv_grid, LVGridDingo):
            self._lv_grids.append(lv_grid)

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
        self.geo_data = kwargs.get('geo_data', None)
        self.lv_load_area = kwargs.get('lv_load_area', None)

    def __repr__(self):
        return 'lv_load_area_centre_' + str(self.id_db)
