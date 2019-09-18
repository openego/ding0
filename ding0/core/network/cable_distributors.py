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


from . import CableDistributorDing0


class MVCableDistributorDing0(CableDistributorDing0):
    """ MV Cable distributor (connection point) 
    
    Attributes
    ----------
    lv_load_area_group : 
        Description #TODO
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lv_load_area_group = kwargs.get('lv_load_area_group', None)
        self.id_db = self.grid.cable_distributors_count() + 1

    @property
    def pypsa_id(self):
        """ :obj:`str`: Returns ...#TODO        
        """
        return '_'.join(['MV', str(self.grid.id_db),
                                  'cld', str(self.id_db)])

    def __repr__(self):
        return 'mv_cable_dist_' + repr(self.grid) + '_' + str(self.id_db)


class LVCableDistributorDing0(CableDistributorDing0):
    """ LV Cable distributor (connection point) 
    
    Attributes
    ----------
    string_id : 
        Description #TODO
    branch_no : 
        Description #TODO
    load_no : 
        Description #TODO
    in_building : 
        Description #TODO
    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.string_id = kwargs.get('string_id', None)
        self.branch_no = kwargs.get('branch_no', None)
        self.load_no = kwargs.get('load_no', None)
        self.id_db = self.grid.cable_distributors_count() + 1
        self.in_building = kwargs.get('in_building', False)

    def __repr__(self):
        return 'lv_cable_dist_' + repr(self.grid) + '_' + str(self.id_db)
