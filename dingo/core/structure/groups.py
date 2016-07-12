from dingo.tools import config as cfg_dingo


class LVRegionGroupDingo:
    """ Container for small LV regions / load areas (satellites) = a group of stations which are within the same
        satellite string. It is required to check whether a satellite string has got more load than allowed, hence new
        nodes cannot be added to it.
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self._lv_regions = []
        self.peak_load_sum = 0
        self.branch_length_sum = 0
        # threshold: max. allowed peak load of satellite string
        self.peak_load_max = cfg_dingo.get('mv_connect', 'load_area_sat_string_load_threshold')
        self.branch_length_max = cfg_dingo.get('mv_connect', 'load_area_sat_string_length_threshold')
        # TODO: Value is read from file every time a LV region is created -> move to associated NetworkDingo class?

    def lv_regions(self):
        """Returns a generator for iterating over LV regions"""
        for region in self._lv_regions:
            yield region

    def add_lv_region(self, lv_region, branch_length):
        """Adds a LV region to _lv_regions if not already existing"""
        self._lv_regions.append(lv_region)
        self.peak_load_sum += lv_region.peak_load_sum
        self.branch_length_sum += branch_length

    def can_add_lv_region(self, lv_region, branch_length):
        """Sums up peak load of LV stations = total peak load for satellite string"""
        if lv_region not in self.lv_regions():  # and isinstance(lv_region, LVRegionDingo):
            if (((lv_region.peak_load_sum + self.peak_load_sum) <= self.peak_load_max) and
                    ((branch_length + self.branch_length_sum) <= self.branch_length_max)):
                return True
            else:
                return False

    def __repr__(self):
        return 'lvregiongroup_' + str(self.id_db)
