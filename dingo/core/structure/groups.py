from dingo.core.network import CableDistributorDingo
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
        self.root_node = kwargs.get('root_node', None)  # root node (Dingo object) = start of string on MV main route
        # TODO: Value is read from file every time a LV region is created -> move to associated NetworkDingo class?

    def lv_regions(self):
        """Returns a generator for iterating over LV regions"""
        for region in self._lv_regions:
            yield region

    def add_lv_region(self, lv_region):
        """Adds a LV region to _lv_regions if not already existing"""
        self._lv_regions.append(lv_region)
        if not isinstance(lv_region, CableDistributorDingo):
            self.peak_load_sum += lv_region.peak_load_sum

    def can_add_lv_region(self, node):
        """Sums up peak load of LV stations = total peak load for satellite string"""
        lv_region = node.grid.region
        if lv_region not in self.lv_regions():  # and isinstance(lv_region, LVRegionDingo):
            path_length_to_root = lv_region.mv_region.mv_grid.graph_path_length(self.root_node, node)
            if ((path_length_to_root <= self.branch_length_max) and
                (lv_region.peak_load_sum + self.peak_load_sum) <= self.peak_load_max):
                return True
            else:
                return False

    def __repr__(self):
        return 'lvregiongroup_' + str(self.id_db)
