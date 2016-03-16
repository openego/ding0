from . import RegionDingo
from dingo.core.network.grids import LVGridDingo

from shapely.wkt import loads as wkt_loads


class MVRegionDingo(RegionDingo):
    """
    Defines a MV-region in DINGO
    ----------------------------

    """
    # TODO: add method remove_lv_region()

    def __init__(self, **kwargs):
        #inherit branch parameters from Region
        super().__init__(**kwargs)

        #more params
        self.mv_grid = kwargs.get('mv_grid', None)
        self._lv_regions = []
        self.geo_data = kwargs.get('geo_data', None)

        # INSERT LOAD PARAMS
        self.peak_load = kwargs.get('peak_load', None)

    def lv_regions(self):
        """Returns a generator for iterating over LV regions"""
        for region in self._lv_regions:
            yield region

    def add_lv_region(self, lv_region):
        """Adds a LV region to _lv_regions if not already existing"""
        if lv_region not in self.lv_regions() and isinstance(lv_region, LVRegionDingo):
            self._lv_regions.append(lv_region)

    def __repr__(self):
        return 'mvregion_' + str(self.id_db)

class LVRegionDingo(RegionDingo):
    """
    Defines a LV-region in DINGO
    ----------------------------

    """
    # TODO: add method remove_lv_grid()

    def __init__(self, **kwargs):
        # inherit branch parameters from Region
        super().__init__(**kwargs)

        # more params
        self._lv_grids = [] # TODO: add setter
        self.mv_region = kwargs.get('mv_region', None)

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
        if hasattr(self, 'geo_centroid'):
            self.geo_centroid = wkt_loads(self.geo_centroid)
        if hasattr(self, 'geo_surfacepnt'):
            self.geo_surfacepnt = wkt_loads(self.geo_surfacepnt)

        # Alternative to version above:
        # many params, use better structure (dict? classes from demand-lib?)


        # for attribute in ['geo_area',
        #                   'geo_centroid',
        #                   'geo_surfacepnt',
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
        # self.geo_centroid = kwargs.get('geo_centroid', None)
        # self.geo_surfacepnt = kwargs.get('geo_surfacepnt', None)
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
        if lv_grid not in self.lv_grids() and isinstance(lv_grid, LVGridDingo):
            self._lv_grids.append(lv_grid)

    def __repr__(self):
        return 'lvregion_' + str(self.id_db)