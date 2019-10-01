"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from . import StationDing0
from ding0.core.network import TransformerDing0
from ding0.tools import config as cfg_ding0

from itertools import compress
import numpy as np


class MVStationDing0(StationDing0):
    """
    Defines a MV station in DINGO
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def peak_generation(self, mode):
        """Calculates cumulative peak generation of generators connected to underlying grids
        
        This is done instantaneously using bottom-up approach.

        Parameters
        ----------
        mode: :obj:`str`
            determines which generators are included::

            'MV':   Only generation capacities of MV level are considered.
            
            'MVLV': Generation capacities of MV and LV are considered
                    (= cumulative generation capacities in entire MVGD).

        Returns
        -------
        float
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
        """Set operation voltage level
    
        """

        mv_station_v_level_operation = float(cfg_ding0.get('mv_routing_tech_constraints',
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


        Note
        -----
        Parametrization of transformers bases on [#]_.
        
        Potential hv-mv-transformers are chosen according to [#]_.
        

        References
        ----------
        .. [#] Deutsche Energie-Agentur GmbH (dena), "dena-Verteilnetzstudie.
            Ausbau- und Innovationsbedarf der Stromverteilnetze in Deutschland
            bis 2030.", 2012
        .. [#] X. Tao, "Automatisierte Grundsatzplanung von
            Mittelspannungsnetzen", Dissertation, 2006

        """

        # get power factor for loads and generators
        cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
        cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')

        # get trafo load factors
        load_factor_mv_trans_lc_normal = float(cfg_ding0.get('assumptions',
                                                             'load_factor_mv_trans_lc_normal'))
        load_factor_mv_trans_fc_normal = float(cfg_ding0.get('assumptions',
                                                             'load_factor_mv_trans_fc_normal'))

        # get equipment parameters of MV transformers
        trafo_parameters = self.grid.network.static_data['MV_trafos']

        # get peak load and peak generation
        cum_peak_load = self.peak_load / cos_phi_load
        cum_peak_generation = self.peak_generation(mode='MVLV') / cos_phi_feedin

        # check if load or generation is greater respecting corresponding load factor
        if (cum_peak_load / load_factor_mv_trans_lc_normal) > \
           (cum_peak_generation / load_factor_mv_trans_fc_normal):
            # use peak load and load factor from load case
            load_factor_mv_trans = load_factor_mv_trans_lc_normal
            residual_apparent_power = cum_peak_load
        else:
            # use peak generation and load factor for feedin case
            load_factor_mv_trans = load_factor_mv_trans_fc_normal
            residual_apparent_power = cum_peak_generation

        # determine number and size of required transformers

        # get max. trafo
        transformer_max = trafo_parameters.iloc[trafo_parameters['S_nom'].idxmax()]

        while residual_apparent_power > 0:
            if residual_apparent_power > load_factor_mv_trans * transformer_max['S_nom']:
                transformer = transformer_max
            else:
                # choose trafo
                transformer = trafo_parameters.iloc[
                    trafo_parameters[trafo_parameters['S_nom'] * load_factor_mv_trans >
                                     residual_apparent_power]['S_nom'].idxmin()]

            # add transformer on determined size with according parameters
            self.add_transformer(TransformerDing0(**{'grid': self.grid,
                                                     'v_level': self.grid.v_level,
                                                     's_max_longterm': transformer['S_nom'],
                                                     'id_db': len(list(self.transformers())) + 1}))
            # calc residual load
            residual_apparent_power -= (load_factor_mv_trans *
                                        transformer['S_nom'])

        # if no transformer was selected (no load in grid district), use smallest one
        if len(self._transformers) == 0:
            transformer = trafo_parameters.iloc[trafo_parameters['S_nom'].idxmin()]

            self.add_transformer(
                TransformerDing0(grid=self.grid,
                                 v_level=self.grid.v_level,
                                 s_max_longterm=transformer['S_nom'],
                                 id_db=1))

        # add redundant transformer of the size of the largest transformer
        s_max_max = max((o.s_max_a for o in self._transformers))
        self.add_transformer(TransformerDing0(**{'grid': self.grid,
                                                 'v_level': self.grid.v_level,
                                                 's_max_longterm': s_max_max,
                                                 'id_db': len(list(self.transformers())) + 1}))

    @property
    def pypsa_bus_id(self):
        """ 
        Returns specific ID for representing bus in pypsa network.
        
        Returns
        -------
        :obj:`str`: 
            Representative of pypsa bus
        """
        return '_'.join(['Busbar', 'mvgd'+str(self.grid.id_db), 'MV' + str(self.id_db)])

    @property
    def pypsa_bus0_id(self):
        """
        Returns specific ID for representing bus in pypsa network. Representative node at high voltage side (also used
        for transformer)

        Returns
        -------
        :obj:`str`:
            Representative of pypsa bus
        """
        return '_'.join(['Busbar', 'mvgd' + str(self.grid.id_db), 'HV' + str(self.id_db)])


    def __repr__(self):
        return '_'.join(['MVStation', 'mvgd' + str(self.grid.id_db), str(self.id_db)])

    


class LVStationDing0(StationDing0):
    """
    Defines a LV station in DINGO
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lv_load_area = kwargs.get('lv_load_area', None)

    @property
    def peak_generation(self):
        """Calculates cumulative peak generation of generators connected to underlying LV grid.
        
        This is done instantaneously using bottom-up approach.

        Returns
        -------
        float
            Cumulative peak generation
        """

        return sum([_.capacity for _ in self.grid.generators()])

    @property
    def pypsa_bus_id(self):
        """ 
        Returns specific ID for representing bus in pypsa network.
        
        Returns
        -------
        :obj:`str`: 
            Representative of pypsa bus
        """
        return '_'.join(['BusBar', 'mvgd'+str(
            self.grid.grid_district.lv_load_area.mv_grid_district.mv_grid.\
                id_db), 'lvgd'+str(self.grid.id_db), 'LV'+ str(self.id_db)])

    @property
    def pypsa_bus0_id(self):
        """
        Returns specific ID for representing bus in pypsa network. Representative node at medium voltage side (also used
        for transformer)

        Returns
        -------
        :obj:`str`:
            Representative of pypsa bus
        """
        return '_'.join(['BusBar', 'mvgd' + str(
            self.grid.grid_district.lv_load_area.mv_grid_district.mv_grid. \
                id_db), 'lvgd' + str(self.grid.id_db), 'MV' + str(self.id_db)])

    def __repr__(self):
        return '_'.join(['LVStation', 'mvgd'+str(
            self.grid.grid_district.lv_load_area.mv_grid_district.mv_grid.\
                id_db), 'lvgd'+str(self.grid.id_db), str(self.id_db)])
