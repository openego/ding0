from oemof.core.energy_system import Region


class RegionDingo(Region):
    """
    Defines a region in DINGO, derived from oemof
    ---------------------------------------------

    """
    def __init__(self, **kwargs):
        #inherit parameters from oemof's Region
        super().__init__(**kwargs)

        #more params
        self.id_db = kwargs.get('id_db', None)
