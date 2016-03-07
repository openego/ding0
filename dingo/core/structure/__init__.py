from oemof.core.energy_system import Region

class MvRegionDingo(Region):
    """
    Defines a MV-region in DINGO
    ----------------------------
    
    """


    def __init__(self, **kwargs):
        #inherit branch parameters from oemof's Region
        super().__init__(**kwargs)
        #more params
        self.geo_data = kwargs.get('geo_data', None)
        self.equip_trans_id = kwargs.get('equip_trans_id', None)
        self.v_level = kwargs.get('v_level', None)
        self.s_max_a = kwargs.get('s_max_a', None)
        self.s_max_b = kwargs.get('s_max_b', None)
        self.s_max_c = kwargs.get('s_max_c', None)
        self.phase_angle = kwargs.get('phase_angle', None)
        self.tap_ratio = kwargs.get('tap_ratio', None)
    def db_import(self):
        print('blabla')
        
class LvRegionDingo(Region):
    """
    Defines a LV-region in DINGO
    ----------------------------
    
    """


    def __init__(self, **kwargs):
        #inherit branch parameters from oemof's Region
        super().__init__(**kwargs)
        #more params

    def db_import(self):
        print('blabla')