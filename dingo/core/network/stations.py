from . import StationDingo
from dingo.core.network import TransformerDingo
from dingo.tools import config as cfg_dingo

from itertools import compress
import numpy as np


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
        Potential HV-MV transformers are chosen according to [2]_.
        Parametrization of transformers bases on [1]_.

        References
        ----------
        .. [1] Deutsche Energie-Agentur GmbH (dena), "dena-Verteilnetzstudie.
            Ausbau- und Innovationsbedarf der Stromverteilnetze in Deutschland
            bis 2030.", 2012
        .. [2] X. Tao, "Automatisierte Grundsatzplanung von
            Mittelspannungsnetzen", Dissertation, 2006

        """

        load_factor_mv_trans_lc_normal = float(cfg_dingo.get('assumptions',
                                                             'load_factor_mv_trans_lc_normal'))

        # get equipment parameters of MV transformers
        trafo_parameters = self.grid.network.static_data['MV_trafos']

        # determine number and size of required transformers
        kw2mw = 1e3
        residual_apparent_power = self.grid.grid_district.peak_load / kw2mw

        # get max. trafo
        transformer_max = trafo_parameters.iloc[trafo_parameters['S_max'].idxmax()]

        while residual_apparent_power > 0:
            if residual_apparent_power > load_factor_mv_trans_lc_normal * transformer_max['S_max']:
                transformer = transformer_max
            else:
                # choose trafo
                transformer = trafo_parameters.iloc[
                    trafo_parameters[trafo_parameters['S_max'] >
                                     residual_apparent_power]['S_max'].idxmin()]

            # add transformer on determined size with according parameters
            self.add_transformer(TransformerDingo(**{'v_level': self.grid.v_level,
                                                     's_max_longterm': transformer['S_max']}))
            # calc residual load
            residual_apparent_power -= (load_factor_mv_trans_lc_normal *
                                        transformer['S_max'])

        # if no transformer was selected (no load in grid district), use smallest one
        if len(self._transformers) == 0:
            transformer = trafo_parameters.iloc[trafo_parameters['S_max'].idxmin()]

            self.add_transformer(TransformerDingo(**{'v_level': self.grid.v_level,
                                                     's_max_longterm': transformer['S_max']}))

        # add redundant transformer of the size of the largest transformer
        s_max_max = max((o.s_max_a for o in self._transformers))
        self.add_transformer(TransformerDingo(**{'v_level': self.grid.v_level,
                                                 's_max_longterm': s_max_max}))

    @property
    def pypsa_id(self):
        return '_'.join(['HV', str(self.grid.id_db), 'trd'])

    def __repr__(self):
        return 'mv_station_' + str(self.id_db)


class LVStationDingo(StationDingo):
    """
    Defines a LV station in DINGO
    -----------------------------
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lv_load_area = kwargs.get('lv_load_area', None)

    @property
    def pypsa_id(self):
        return '_'.join(['MV', str(
            self.grid.grid_district.lv_load_area.mv_grid_district.mv_grid.\
                id_db), 'tru', str(self.id_db)])

    def __repr__(self):
        return 'lv_station_' + str(self.id_db)
