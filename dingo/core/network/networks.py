from . import NetworkDingo

class MVNetworkDingo(NetworkDingo):
    """ DINGO medium voltage network
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #more params
        self.station = kwargs.get('station', None)
        #self.id_db = kwargs.get('id_db', None)

class LVNetworkDingo(NetworkDingo):
    """ DINGO low voltage network
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #more params
        self.stations = kwargs.get('stations', None)

        #self.id_db = kwargs.get('id_db', None)