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

    def __repr__(self):
        return 'mv_load_' + str(self.id_db)


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

    def __repr__(self):
        return 'lv_load_' + str(self.id_db)
