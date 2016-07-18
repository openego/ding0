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


class RegionCentreDingo:
    """
    Defines a region centre in Dingo
    --------------------------------
    The centres are used in the MV routing as nodes.
    Note: Centre is a point within a region's polygon that is located most central (e.g. in a simple region shape like a
    circle it's the geometric center).

    Parameters
    ----------
    id_db: unique ID in database (=id of associated load area)
    """
    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)

    def __repr__(self):
        return 'regioncentre_' + str(self.id_db)