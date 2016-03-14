from . import GridDingo

class MVGridDingo(GridDingo):
    """ DINGO medium voltage grid
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #more params
        self.station = kwargs.get('station', None)
        self.graph.add_node(self.station)
        #self.id_db = kwargs.get('id_db', None)

class LVGridDingo(GridDingo):
    """ DINGO low voltage grid
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #more params
        self.stations = kwargs.get('stations', [])

        #self.id_db = kwargs.get('id_db', None)