from . import RegionDingo

class MVRegionDingo(RegionDingo):
    """
    Defines a MV-region in DINGO
    ----------------------------

    """

    def __init__(self, **kwargs):
        #inherit branch parameters from Region
        super().__init__(**kwargs)

        #more params
        self.geo_data = kwargs.get('geo_data', None)
        self.lv_regions = kwargs.get('lv_regions', None) # method instead? (iterate over lv regions)

        # INSERT LOAD PARAMS
        self.peak_load = kwargs.get('peak_load', None)

    def db_import(self):
        print('blabla')

class LVRegionDingo(RegionDingo):
    """
    Defines a LV-region in DINGO
    ----------------------------

    """

    def __init__(self, **kwargs):
        #inherit branch parameters from Region
        super().__init__(**kwargs)

        #more params
        self.geo_data = kwargs.get('geo_data', None)
        self.mv_region = kwargs.get('mv_region', None)

        # INSERT PARAMS FROM LUI

    def db_import(self):
        print('blabla')