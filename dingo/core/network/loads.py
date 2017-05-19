"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DINGO lives at github: https://github.com/openego/dingo/
The documentation is available on RTD: http://dingo.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/dingo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from . import LoadDingo


class MVLoadDingo(LoadDingo):
    """
    Load in MV grids

    """
    # TODO: Currently not used, check later if still required

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return 'mv_load_' + str(self.id_db)


class LVLoadDingo(LoadDingo):
    """
    Load in LV grids

    Notes
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