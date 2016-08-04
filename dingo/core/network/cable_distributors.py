from . import CableDistributorDingo


class MVCableDistributorDingo(CableDistributorDingo):
    """ Cable distributor (connection point) """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lv_load_area_group = kwargs.get('lv_load_area_group', None)

        # build id from associated grid district id and the count of cable distributors in grid, use 10^4
        # (because max. count of MV grid districts is 4*10^3) as factor to separate both ids
        # (allow later distinction between these two parts)
        self.id_db = self.grid.grid_district.id_db * 10**4 + self.grid.cable_distributors_count() + 1

    def __repr__(self):
        return 'mv_cable_dist_' + str(self.id_db)


class LVCableDistributorDingo(CableDistributorDingo):
    """LV Cable distributor (connection point) """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.string_id = kwargs.get('string_id', None)
        self.branch_no = kwargs.get('branch_no', None)
        self.load_no = kwargs.get('load_no', None)

        # build id from associated grid district id and the count of cable distributors in grid, use 10^8
        # (because max. count of LV grid districts is 5*10^6 and an additional 10^1 is required to achieve same length
        # as MV grid branches' ids) as factor to separate both ids (allow later distinction between these two parts)
        self.id_db = self.grid.grid_district.id_db * 10**8 + self.grid.cable_distributors_count() + 1

    def __repr__(self):
        return 'lv_cable_dist_' + str(self.id_db)
