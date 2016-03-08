from . import StationDingo

class MVStationDingo(StationDingo):
    """
    Defines a MV station in DINGO
    -----------------------------
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.busbar = None

    def db_import(self, con):
        # populate objects

class LVStationDingo(StationDingo):
    """
    Defines a LV station in DINGO
    -----------------------------
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def db_import(self, con):
        # populate objects