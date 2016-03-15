from . import StationDingo

class MVStationDingo(StationDingo):
    """
    Defines a MV station in DINGO
    -----------------------------
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return 'mvstation_' + str(self.id_db)


class LVStationDingo(StationDingo):
    """
    Defines a LV station in DINGO
    -----------------------------
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return 'lvstation_' + str(self.id_db)