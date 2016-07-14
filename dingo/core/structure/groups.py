from dingo.core.network import CableDistributorDingo
from dingo.tools import config as cfg_dingo


class LVRegionGroupDingo:
    """ Container for small LV regions / load areas (satellites) = a group of stations which are within the same
        satellite string. It is required to check whether a satellite string has got more load than allowed, hence new
        nodes cannot be added to it.
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self._lv_load_areas = []
        self.peak_load_sum = 0
        self.branch_length_sum = 0
        # threshold: max. allowed peak load of satellite string
        self.peak_load_max = cfg_dingo.get('mv_connect', 'load_area_sat_string_load_threshold')
        self.branch_length_max = cfg_dingo.get('mv_connect', 'load_area_sat_string_length_threshold')
        self.root_node = kwargs.get('root_node', None)  # root node (Dingo object) = start of string on MV main route
        # TODO: Value is read from file every time a LV region is created -> move to associated NetworkDingo class?

    def lv_load_areas(self):
        """Returns a generator for iterating over LV regions"""
        for region in self._lv_load_areas:
            yield region

    def add_lv_load_area(self, lv_load_area):
        """Adds a LV region to _lv_load_areas if not already existing"""
        self._lv_load_areas.append(lv_load_area)
        if not isinstance(lv_load_area, CableDistributorDingo):
            self.peak_load_sum += lv_load_area.peak_load_sum

    def can_add_lv_load_area(self, node):
        """Sums up peak load of LV stations = total peak load for satellite string"""
        lv_load_area = node.grid.region
        if lv_load_area not in self.lv_load_areas():  # and isinstance(lv_load_area, LVLoadAreaDingo):
            path_length_to_root = lv_load_area.mv_grid_district.mv_grid.graph_path_length(self.root_node, node)
            if ((path_length_to_root <= self.branch_length_max) and
                (lv_load_area.peak_load_sum + self.peak_load_sum) <= self.peak_load_max):
                return True
            else:
                return False

    def __repr__(self):
        return 'lvregiongroup_' + str(self.id_db)
