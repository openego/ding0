from . import GridDingo
from dingo.core.network.stations import *

class MVGridDingo(GridDingo):
    """ DINGO medium voltage grid
    """
    # TODO: Add method to join MV graph with LV graphs to have one graph that covers whole grid (MV and LV)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #more params
        self._station = None

        self.add_station(kwargs.get('station', None))

        self.graph.add_node(self.station)

    def station(self):
        """Returns MV station"""
        return self._station

    def add_station(self, mv_station, force=False):
        """Adds MV station if not already existing

        mv_station: MVStationDingo object
        force: bool. If True, MV Station is set even though it's not empty (override)
        """
        # TODO: Use this exception handling as template for similar methods in other classes
        if not isinstance(mv_station, MVStationDingo):
            raise Exception('Given MV station is not a MVStationDingo object.')
        if self._station is None:
            self._station = mv_station
        else:
            if force:
                self._station = mv_station
            else:
                raise Exception('MV Station already set, use argument `force=True` to override.')

    def graph_build(self):
        """Builds/fills graph with objects (stations, ..)"""
        # TODO: Add edges, loads etc. later on

        # add MV station
        self.graph_add_node(self._station)

        # add LV stations
        for lv_station in LV_STATIONS <-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            self.graph_add_node(lv_station)

    def __repr__(self):
        return 'mvgrid_' + str(self.id_db)

class LVGridDingo(GridDingo):
    """ DINGO low voltage grid
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #more params
        self._stations = []

    def stations(self):
        """Returns a generator for iterating over LV stations"""
        for station in self._stations:
            yield station

    def add_station(self, lv_station):
        """Adds a LV station to _stations if not already existing"""
        if lv_station not in self.stations() and isinstance(lv_station, LVStationDingo):
            self._stations.append(lv_station)

    def graph_build(self):
        """Builds/fills graph with objects (stations, ..)"""
        # TODO: Add edges, loads etc. later on

        # add LV stations
        for lv_station in LV_STATIONS <-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            self.graph_add_node(lv_station)

        # TODO: add more nodes (loads etc.) here

    def __repr__(self):
        return 'lvgrid_' + str(self.id_db)