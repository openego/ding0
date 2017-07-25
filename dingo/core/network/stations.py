"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DINGO lives at github: https://github.com/openego/dingo/
The documentation is available on RTD: http://dingo.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/dingo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


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

    def peak_generation(self, mode):
        """
        Calculates cumulative peak generation of generators connected to
        underlying MV grid or MV+LV grids (controlled by parameter `mode`).
        This is done instantaneously using bottom-up approach.

        Parameters
        ----------
        mode: String
            determines which generators are included:

            'MV':   Only generation capacities of MV level are considered.
            'MVLV': Generation capacities of MV and LV are considered
                    (= cumulative generation capacities in entire MVGD).

        Returns
        -------
        capacity: Float
            Cumulative peak generation
        """

        if mode == 'MV':
            return sum([_.capacity for _ in self.grid.generators()])

        elif mode == 'MVLV':
            # calc MV geno capacities
            cum_mv_peak_generation = sum([_.capacity for _ in self.grid.generators()])
            # calc LV geno capacities
            cum_lv_peak_generation = 0
            for load_area in self.grid.grid_district.lv_load_areas():
                cum_lv_peak_generation += load_area.peak_generation

            return cum_mv_peak_generation + cum_lv_peak_generation

        else:
            raise ValueError('parameter \'mode\' is invalid!')

    def set_operation_voltage_level(self):

        mv_station_v_level_operation = float(cfg_dingo.get('mv_routing_tech_constraints',
                                                           'mv_station_v_level_operation'))

        self.v_level_operation = mv_station_v_level_operation * self.grid.v_level

    def select_transformers(self):
        """ Selects appropriate transformers for the HV-MV substation.

        The transformers are chosen according to max. of load case and feedin-case
        considering load factors.
        The HV-MV transformer with the next higher available nominal apparent power is
        chosen. If one trafo is not sufficient, multiple trafos are used. Additionally,
        in a second step an redundant trafo is installed with max. capacity of the
        selected trafos of the first step according to general planning principles for
        MV distribution grids (n-1).

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

        # get power factor for loads and generators
        cos_phi_load = cfg_dingo.get('assumptions', 'cos_phi_load')
        cos_phi_feedin = cfg_dingo.get('assumptions', 'cos_phi_gen')

        # get trafo load factors
        load_factor_mv_trans_lc_normal = float(cfg_dingo.get('assumptions',
                                                             'load_factor_mv_trans_lc_normal'))
        load_factor_mv_trans_fc_normal = float(cfg_dingo.get('assumptions',
                                                             'load_factor_mv_trans_fc_normal'))

        # get equipment parameters of MV transformers
        trafo_parameters = self.grid.network.static_data['MV_trafos']

        # get peak load and peak generation
        cum_peak_load = self.peak_load / cos_phi_load
        cum_peak_generation = self.peak_generation(mode='MVLV') / cos_phi_feedin

        kw2mw = 1e-3

        # check if load or generation is greater respecting corresponding load factor
        if (cum_peak_load / load_factor_mv_trans_lc_normal) > \
           (cum_peak_generation / load_factor_mv_trans_fc_normal):
            # use peak load and load factor from load case
            load_factor_mv_trans = load_factor_mv_trans_lc_normal
            residual_apparent_power = cum_peak_load * kw2mw
        else:
            # use peak generation and load factor for feedin case
            load_factor_mv_trans = load_factor_mv_trans_fc_normal
            residual_apparent_power = cum_peak_generation * kw2mw

        # determine number and size of required transformers

        # get max. trafo
        transformer_max = trafo_parameters.iloc[trafo_parameters['S_max'].idxmax()]

        while residual_apparent_power > 0:
            if residual_apparent_power > load_factor_mv_trans * transformer_max['S_max']:
                transformer = transformer_max
            else:
                # choose trafo
                transformer = trafo_parameters.iloc[
                    trafo_parameters[trafo_parameters['S_max'] * load_factor_mv_trans >
                                     residual_apparent_power]['S_max'].idxmin()]

            # add transformer on determined size with according parameters
            self.add_transformer(TransformerDingo(**{'grid': self.grid,
                                                     'v_level': self.grid.v_level,
                                                     's_max_longterm': transformer['S_max']}))
            # calc residual load
            residual_apparent_power -= (load_factor_mv_trans *
                                        transformer['S_max'])

        # if no transformer was selected (no load in grid district), use smallest one
        if len(self._transformers) == 0:
            transformer = trafo_parameters.iloc[trafo_parameters['S_max'].idxmin()]

            self.add_transformer(TransformerDingo(**{'grid': self.grid,
                                                     'v_level': self.grid.v_level,
                                                     's_max_longterm': transformer['S_max']}))

        # add redundant transformer of the size of the largest transformer
        s_max_max = max((o.s_max_a for o in self._transformers))
        self.add_transformer(TransformerDingo(**{'grid': self.grid,
                                                 'v_level': self.grid.v_level,
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
    def peak_generation(self):
        """
        Calculates cumulative peak generation of generators connected to
        underlying LV grid. This is done instantaneously using bottom-up
        approach.

        Returns
        -------
        capacity: Float
            Cumulative peak generation
        """

        return sum([_.capacity for _ in self.grid.generators()])

    @property
    def pypsa_id(self):
        return '_'.join(['MV', str(
            self.grid.grid_district.lv_load_area.mv_grid_district.mv_grid.\
                id_db), 'tru', str(self.id_db)])

    def __repr__(self):
        return 'lv_station_' + str(self.id_db)
