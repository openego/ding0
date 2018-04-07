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


from ding0.core import MVCableDistributorDing0
from ding0.tools import config as cfg_ding0


class LoadAreaGroupDing0:
        # TODO: check docstring
    """ Container for small load_areas / load areas (satellites).
    
    A group of stations which are within the same satellite string. It is 
    required to check whether a satellite string has got more load or string 
    length than allowed, hence new nodes cannot be added to it.
    
    Attributes
    ----------
    id_db: :obj:`int`
        Descr
    mv_grid_district : :shapely:`Shapely Polygon object<polygons>`
        Desc
    
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.mv_grid_district = kwargs.get('mv_grid_district', None)
        self._lv_load_areas = []
        self.peak_load = 0
        self.branch_length_sum = 0
        # threshold: max. allowed peak load of satellite string
        self.peak_load_max = cfg_ding0.get('mv_connect', 'load_area_sat_string_load_threshold')
        self.branch_length_max = cfg_ding0.get('mv_connect', 'load_area_sat_string_length_threshold')
        self.root_node = kwargs.get('root_node', None)  # root node (Ding0 object) = start of string on MV main route
        # TODO: Value is read from file every time a LV load_area is created -> move to associated NetworkDing0 class?

        # get id from count of load area groups in associated MV grid district
        self.id_db = self.mv_grid_district.lv_load_area_groups_count() + 1

    @property
    def network(self):
        return self.mv_grid_district.network

    def lv_load_areas(self):
        # TODO: check docstring
        """Returns a generator for iterating over load_areas
        
        Yields
        ------
        int
            generator for iterating over load_areas
        """
        for load_area in self._lv_load_areas:
            yield load_area

    def add_lv_load_area(self, lv_load_area):
        # TODO: check docstring
        """Adds a LV load_area to _lv_load_areas if not already existing
        
        Args
        ----
        lv_load_area: :shapely:`Shapely Polygon object<polygons>`
            Descr
        """
        self._lv_load_areas.append(lv_load_area)
        if not isinstance(lv_load_area, MVCableDistributorDing0):
            self.peak_load += lv_load_area.peak_load

    def can_add_lv_load_area(self, node):
        # TODO: check docstring
        """Sums up peak load of LV stations 
        
        That is, total peak load for satellite string
        
        Args
        ----
        node: GridDing0
            Descr
        
        Returns
        -------
        bool
            True if ????
        
        """
        # get power factor for loads
        cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')

        lv_load_area = node.lv_load_area
        if lv_load_area not in self.lv_load_areas():  # and isinstance(lv_load_area, LVLoadAreaDing0):
            path_length_to_root = lv_load_area.mv_grid_district.mv_grid.graph_path_length(self.root_node, node)
            if ((path_length_to_root <= self.branch_length_max) and
                (lv_load_area.peak_load + self.peak_load) / cos_phi_load <= self.peak_load_max):
                return True
            else:
                return False

    def __repr__(self):
        return 'load_area_group_' + str(self.id_db)
