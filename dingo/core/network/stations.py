from . import StationDingo
from dingo.core.network import TransformerDingo
from dingo.tools import config as cfg_dingo

from itertools import compress

class MVStationDingo(StationDingo):
    """
    Defines a MV station in DINGO
    -----------------------------
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_operation_voltage_level(self):

        mv_station_v_level_operation = float(cfg_dingo.get('mv_routing_tech_constraints',
                                                           'mv_station_v_level_operation'))

        self.v_level_operation = mv_station_v_level_operation * self.grid.v_level

    def choose_transformers(self):
        """Chooses appropriate transformers for the MV sub-station

        Choice bases on voltage level (depends on load density), apparent power
        and general planning principles for MV distribution grids.

        Parameters
        ----------
        transformers : dict
            Contains technical information of p hv/mv transformers
        **kwargs : dict
            Should contain a value behind the key 'peak_load'

        Notes
        -----
        Potential hv-mv-transformers are chosen according to [2]_.
        Parametrization of transformers bases on [1]_.

        References
        ----------
        .. [1] Deutsche Energie-Agentur GmbH (dena), "dena-Verteilnetzstudie.
            Ausbau- und Innovationsbedarf der Stromverteilnetze in Deutschland
            bis 2030.", 2012
        .. [2] X. Tao, "Automatisierte Grundsatzplanung von
            Mittelspannungsnetzen", Dissertation, 2006

        """

        # Parameters of possible transformers
        # TODO: move to database of config file
        transformers = {
            20000: {
                'voltage_level': 20,
                'apparent_power': 20000},
            31500: {
                'voltage_level': 10,
                'apparent_power': 31500},
            40000: {
                'voltage_level': 10,
                'apparent_power': 40000}}

        load_factor_transformer_normal = float(cfg_dingo.get('assumptions',
                                                             'load_factor_transformer_normal'))

        # step 1: identify possible transformers by voltage level
        apparent_power = self.grid.grid_district.peak_load  # kW
        possible_transformers = []  # keys of above dict

        # put all keys of suitable transformers (based on voltage) to list
        for trans in transformers:
            if self.grid.v_level == transformers[trans]['voltage_level']:
                possible_transformers.append(trans)

        # step 2: determine number and size of required transformers
        residual_apparent_power = apparent_power

        apparent_power_list = [transformers[_]['apparent_power']
                               for _ in possible_transformers]

        while residual_apparent_power > 0:
            if residual_apparent_power > load_factor_transformer_normal * max(possible_transformers):
                selected_app_power = max(possible_transformers)
            else:
                selected_app_power = min(list(compress(possible_transformers,
                    [residual_apparent_power <= k
                     for k in possible_transformers])))

            # add transformer on determined size with according parameters
            self.add_transformer(TransformerDingo(**{'v_level': self.grid.v_level,
                's_max_longterm': selected_app_power}))
            residual_apparent_power -= (load_factor_transformer_normal *
                                        selected_app_power)

        # add redundant transformer of the size of the largest transformer
        s_max_max = max((o.s_max_a for o in self._transformers))
        int_kwargs = {'v_level': self.grid.v_level,
                      's_max_longterm': s_max_max}

        self.add_transformer(TransformerDingo(**int_kwargs))

    def __repr__(self):
        return 'mv_station_' + str(self.id_db)

class LVStationDingo(StationDingo):
    """
    Defines a LV station in DINGO
    -----------------------------
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return 'lv_station_' + str(self.id_db)
