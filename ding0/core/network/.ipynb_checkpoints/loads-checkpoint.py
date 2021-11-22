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


from . import LoadDing0


class MVLoadDing0(LoadDing0):
    """
    Load in MV grids

    Note
    -----
    Currently not used, check later if still required
    """
    # TODO: Currently not used, check later if still required

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id_db = self.grid.mv_grid.loads_count() + 1
        self.osmid_building = kwargs.get('osmid_building', None)
        self.osmid_nn = kwargs.get('osmid_nn', None)
        self.nn_coords = kwargs.get('nn_coords', None)
        self.lv_load_area = kwargs.get('lv_load_area', None)

    def __repr__(self):
        """
        The Representative of the
        :class:`~.ding0.core.network.CircuitBreakerDing0` object.

        Returns
        -------
        :obj:`str`
        """
        return '_'.join(['Load', 'mvgd', str(self.grid.id_db),
                         str(self.id_db)])


class LVLoadDing0(LoadDing0):
    """
    Load in LV grids

    Note
    -----
    Current attributes to fulfill requirements of typified model grids.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.string_id = kwargs.get('string_id', None)
        self.branch_no = kwargs.get('branch_no', None)
        self.load_no = kwargs.get('load_no', None)
        self.id_db = self.grid.loads_count() + 1

    def __repr__(self):
        """
        The Representative of the
        :class:`~.ding0.core.network.CircuitBreakerDing0` object.

        Returns
        -------
        :obj:`str`
        """
        return '_'.join(['Load', 'mvgd', str(
            self.grid.grid_district.lv_load_area.mv_grid_district.mv_grid.\
            id_db), 'lvgd', str(self.grid.id_db), str(self.id_db)])
