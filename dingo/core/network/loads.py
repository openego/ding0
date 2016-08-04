from . import LoadDingo


class LVLoadDingo(LoadDingo):
    """
    Load in LV grids

    Notes
    -----
    Current attributes and __repr__ is designed to fulfill requirements of
    typified model grids.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.string_id = kwargs.get('string_id', None)
        self.branch_no = kwargs.get('branch_no', None)
        self.load_no = kwargs.get('load_no', None)

    def __repr__(self):
        return ('lv_load_' + str(self.id_db) + '_' + str(self.string_id) + '-'
            + str(self.branch_no) + '_' + str(self.load_no))