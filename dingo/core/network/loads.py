from . import LoadDingo


class MVLoadDingo(LoadDingo):
    """
    Load in MV grids

    """
    # TODO: Currently not used, check later if still required

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.id_db = self.grid.loads_count() + 1

    def __repr__(self):
        return 'mv_load_' + str(self.id_db)


class LVLoadDingo(LoadDingo):
    """
    Load in LV grids

    Notes
    -----
    Current attributes to fulfill requirements of typified model grids.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.string_id = kwargs.get('string_id', None)
        self.branch_no = kwargs.get('branch_no', None)
        self.load_no = kwargs.get('load_no', None)
        self.id_db = self.grid.loads_count() + 1

    def __repr__(self):
        return 'lv_load_' + str(self.id_db)