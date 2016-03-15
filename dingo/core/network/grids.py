from . import GridDingo

class MVGridDingo(GridDingo):
    """ DINGO medium voltage grid
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #more params
        self.station = kwargs.get('station', None)
        self.graph.add_node(self.station)

    def __repr__(self):
        return 'mvgrid_' + str(self.id_db)

class LVGridDingo(GridDingo):
    """ DINGO low voltage grid
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #more params
        self.stations = kwargs.get('stations', [])

    def __repr__(self):
        return 'lvgrid_' + str(self.id_db)