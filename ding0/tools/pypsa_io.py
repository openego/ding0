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


import ding0

from ding0.tools import config as cfg_ding0
from ding0.tools.tools import merge_two_dicts
from ding0.core.network.stations import LVStationDing0, MVStationDing0
from ding0.core.network import LoadDing0, CircuitBreakerDing0, GeneratorDing0, GeneratorFluctuatingDing0
from ding0.core import MVCableDistributorDing0
from ding0.core.structure.regions import LVLoadAreaCentreDing0, LVGridDistrictDing0, LVLoadAreaDing0
from ding0.core.powerflow import q_sign
from ding0.core.network.cable_distributors import LVCableDistributorDing0
from ding0.core import network as ding0_nw
from ding0.tools.tools import merge_two_dicts_of_dataframes

from geoalchemy2.shape import from_shape
from math import tan, acos, pi, sqrt
from pandas import Series, DataFrame, DatetimeIndex
from pypsa.io import import_series_from_dataframe
from pypsa import Network
from shapely.geometry import Point

from datetime import datetime
import sys
import os
import logging
import pandas as pd
import numpy as np

if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString

logger = logging.getLogger('ding0')


def export_to_dir(network, export_dir):
    """
    Exports PyPSA network as CSV files to directory

    Parameters
    ----------
        network: :pypsa:pypsa.Network
        export_dir: :obj:`str`
            Sub-directory in output/debug/grid/
            where csv Files of PyPSA network are exported to.
    """

    package_path = ding0.__path__[0]

    network.export_to_csv_folder(os.path.join(package_path,
                                              'output',
                                              'debug',
                                              'grid',
                                              export_dir))


def nodes_to_dict_of_dataframes(grid, nodes, lv_transformer=True):
    """
    Creates dictionary of dataframes containing grid

    Parameters
    ----------
    grid: ding0.MVGridDing0
    nodes: :obj:`list` of ding0 grid components objects
        Nodes of the grid graph
    lv_transformer: bool, True
        Toggle transformer representation in power flow analysis

    Returns:
    components: dict of :pandas:`pandas.DataFrame<dataframe>`
        DataFrames contain components attributes. Dict is keyed by components
        type
    components_data: dict of :pandas:`pandas.DataFrame<dataframe>`
        DataFrame containing components time-varying data
    """
    generator_instances = [MVStationDing0, GeneratorDing0]
    # TODO: MVStationDing0 has a slack generator

    cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
    cos_phi_load_mode = cfg_ding0.get('assumptions', 'cos_phi_load_mode')
    cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
    cos_phi_feedin_mode = cfg_ding0.get('assumptions', 'cos_phi_gen_mode')
    srid = int(cfg_ding0.get('geo', 'srid'))

    load_in_generation_case = cfg_ding0.get('assumptions',
                                            'load_in_generation_case')
    generation_in_load_case = cfg_ding0.get('assumptions',
                                            'generation_in_load_case')

    Q_factor_load = q_sign(cos_phi_load_mode, 'load') * tan(acos(cos_phi_load))
    Q_factor_generation = q_sign(cos_phi_feedin_mode, 'generator') * tan(acos(cos_phi_feedin))

    voltage_set_slack = cfg_ding0.get("mv_routing_tech_constraints",
                                      "mv_station_v_level_operation")

    kw2mw = 1e-3

    # define dictionaries
    buses = {'bus_id': [], 'v_nom': [], 'geom': [], 'grid_id': []}
    bus_v_mag_set = {'bus_id': [], 'temp_id': [], 'v_mag_pu_set': [],
                     'grid_id': []}
    generator = {'generator_id': [], 'bus': [], 'control': [], 'grid_id': [],
                 'p_nom': []}
    generator_pq_set = {'generator_id': [], 'temp_id': [], 'p_set': [],
                        'grid_id': [], 'q_set': []}
    load = {'load_id': [], 'bus': [], 'grid_id': []}
    load_pq_set = {'load_id': [], 'temp_id': [], 'p_set': [],
                   'grid_id': [], 'q_set': []}

    # # TODO: consider other implications of `lv_transformer is True`
    # if lv_transformer is True:
    #     bus_instances.append(Transformer)

    for node in nodes:
        if node not in grid.graph_isolated_nodes():
            # buses only
            if isinstance(node, MVCableDistributorDing0):
                buses['bus_id'].append(node.pypsa_bus_id)
                buses['v_nom'].append(grid.v_level)
                buses['geom'].append(from_shape(node.geo_data, srid=srid))
                buses['grid_id'].append(grid.id_db)

                bus_v_mag_set['bus_id'].append(node.pypsa_bus_id)
                bus_v_mag_set['temp_id'].append(1)
                bus_v_mag_set['v_mag_pu_set'].append([1, 1])
                bus_v_mag_set['grid_id'].append(grid.id_db)

            # bus + generator
            elif isinstance(node, tuple(generator_instances)):
                # slack generator
                if isinstance(node, MVStationDing0):
                    logger.info('Only MV side bus of MVStation will be added.')
                    generator['generator_id'].append(
                        '_'.join(['MV', str(grid.id_db), 'slack']))
                    generator['control'].append('Slack')
                    generator['p_nom'].append(0)
                    bus_v_mag_set['v_mag_pu_set'].append(
                        [voltage_set_slack, voltage_set_slack])

                # other generators
                if isinstance(node, GeneratorDing0):
                    generator['generator_id'].append('_'.join(
                        ['MV', str(grid.id_db), 'gen', str(node.id_db)]))
                    generator['control'].append('PQ')
                    generator['p_nom'].append(node.capacity * node.capacity_factor)

                    generator_pq_set['generator_id'].append('_'.join(
                        ['MV', str(grid.id_db), 'gen', str(node.id_db)]))
                    generator_pq_set['temp_id'].append(1)
                    generator_pq_set['p_set'].append(
                        [node.capacity * node.capacity_factor * kw2mw * generation_in_load_case,
                         node.capacity * node.capacity_factor * kw2mw])
                    generator_pq_set['q_set'].append(
                        [node.capacity * node.capacity_factor * kw2mw * Q_factor_generation * generation_in_load_case,
                         node.capacity * node.capacity_factor * kw2mw * Q_factor_generation])
                    generator_pq_set['grid_id'].append(grid.id_db)
                    bus_v_mag_set['v_mag_pu_set'].append([1, 1])

                buses['bus_id'].append(node.pypsa_bus_id)
                buses['v_nom'].append(grid.v_level)
                buses['geom'].append(from_shape(node.geo_data, srid=srid))
                buses['grid_id'].append(grid.id_db)

                bus_v_mag_set['bus_id'].append(node.pypsa_bus_id)
                bus_v_mag_set['temp_id'].append(1)
                bus_v_mag_set['grid_id'].append(grid.id_db)

                generator['grid_id'].append(grid.id_db)
                generator['bus'].append(node.pypsa_bus_id)


            # aggregated load at hv/mv substation
            elif isinstance(node, LVLoadAreaCentreDing0):
                load['load_id'].append(node.pypsa_bus_id)
                load['bus'].append('_'.join(['HV', str(grid.id_db), 'trd']))
                load['grid_id'].append(grid.id_db)

                load_pq_set['load_id'].append(node.pypsa_bus_id)
                load_pq_set['temp_id'].append(1)
                load_pq_set['p_set'].append(
                    [node.lv_load_area.peak_load * kw2mw,
                     node.lv_load_area.peak_load * kw2mw * load_in_generation_case])
                load_pq_set['q_set'].append(
                    [node.lv_load_area.peak_load * kw2mw * Q_factor_load,
                     node.lv_load_area.peak_load * kw2mw * Q_factor_load * load_in_generation_case])
                load_pq_set['grid_id'].append(grid.id_db)

                # generator representing generation capacity of aggregate LA
                # analogously to load, generation is connected directly to
                # HV-MV substation
                generator['generator_id'].append('_'.join(
                    ['MV', str(grid.id_db), 'lcg', str(node.id_db)]))
                generator['control'].append('PQ')
                generator['p_nom'].append(node.lv_load_area.peak_generation)
                generator['grid_id'].append(grid.id_db)
                generator['bus'].append('_'.join(['HV', str(grid.id_db), 'trd']))

                generator_pq_set['generator_id'].append('_'.join(
                    ['MV', str(grid.id_db), 'lcg', str(node.id_db)]))
                generator_pq_set['temp_id'].append(1)
                generator_pq_set['p_set'].append(
                    [node.lv_load_area.peak_generation * kw2mw * generation_in_load_case,
                     node.lv_load_area.peak_generation * kw2mw])
                generator_pq_set['q_set'].append(
                    [node.lv_load_area.peak_generation * kw2mw * Q_factor_generation * generation_in_load_case,
                     node.lv_load_area.peak_generation * kw2mw * Q_factor_generation])
                generator_pq_set['grid_id'].append(grid.id_db)

            # bus + aggregate load of lv grids (at mv/ls substation)
            elif isinstance(node, LVStationDing0):
                # Aggregated load representing load in LV grid
                load['load_id'].append(
                    '_'.join(['MV', str(grid.id_db), 'loa', str(node.id_db)]))
                load['bus'].append(node.pypsa_bus_id)
                load['grid_id'].append(grid.id_db)

                load_pq_set['load_id'].append(
                    '_'.join(['MV', str(grid.id_db), 'loa', str(node.id_db)]))
                load_pq_set['temp_id'].append(1)
                load_pq_set['p_set'].append(
                    [node.peak_load * kw2mw,
                     node.peak_load * kw2mw * load_in_generation_case])
                load_pq_set['q_set'].append(
                    [node.peak_load * kw2mw * Q_factor_load,
                     node.peak_load * kw2mw * Q_factor_load * load_in_generation_case])
                load_pq_set['grid_id'].append(grid.id_db)

                # bus at primary MV-LV transformer side
                buses['bus_id'].append(node.pypsa_bus_id)
                buses['v_nom'].append(grid.v_level)
                buses['geom'].append(from_shape(node.geo_data, srid=srid))
                buses['grid_id'].append(grid.id_db)

                bus_v_mag_set['bus_id'].append(node.pypsa_bus_id)
                bus_v_mag_set['temp_id'].append(1)
                bus_v_mag_set['v_mag_pu_set'].append([1, 1])
                bus_v_mag_set['grid_id'].append(grid.id_db)

                # generator representing generation capacity in LV grid
                generator['generator_id'].append('_'.join(
                    ['MV', str(grid.id_db), 'gen', str(node.id_db)]))
                generator['control'].append('PQ')
                generator['p_nom'].append(node.peak_generation)
                generator['grid_id'].append(grid.id_db)
                generator['bus'].append(node.pypsa_bus_id)

                generator_pq_set['generator_id'].append('_'.join(
                    ['MV', str(grid.id_db), 'gen', str(node.id_db)]))
                generator_pq_set['temp_id'].append(1)
                generator_pq_set['p_set'].append(
                    [node.peak_generation * kw2mw * generation_in_load_case,
                     node.peak_generation * kw2mw])
                generator_pq_set['q_set'].append(
                    [node.peak_generation * kw2mw * Q_factor_generation * generation_in_load_case,
                     node.peak_generation * kw2mw * Q_factor_generation])
                generator_pq_set['grid_id'].append(grid.id_db)

            elif isinstance(node, CircuitBreakerDing0):
                # TODO: remove this elif-case if CircuitBreaker are removed from graph
                continue
            else:
                raise TypeError("Node of type", node, "cannot be handled here")
        else:
            if not isinstance(node, CircuitBreakerDing0):
                add_info =  "LA is aggr. {0}".format(
                    node.lv_load_area.is_aggregated)
            else:
                add_info = ""
            logger.warning("Node {0} is not connected to the graph and will " \
                  "be omitted in power flow analysis. {1}".format(
                node, add_info))

    components = {'Bus': DataFrame(buses).set_index('bus_id'),
                  'Generator': DataFrame(generator).set_index('generator_id'),
                  'Load': DataFrame(load).set_index('load_id')}

    components_data = {'Bus': DataFrame(bus_v_mag_set).set_index('bus_id'),
                       'Generator': DataFrame(generator_pq_set).set_index(
                           'generator_id'),
                       'Load': DataFrame(load_pq_set).set_index('load_id')}

    # with open('/home/guido/ding0_debug/number_of_nodes_buses.csv', 'a') as csvfile:
    #     csvfile.write(','.join(['\n', str(len(nodes)), str(len(grid.graph_isolated_nodes())), str(len(components['Bus']))]))

    return components, components_data

def initialize_component_dataframes():
    cols = {'buses_columns': ['name', 'geom', 'mv_grid_id', 'lv_grid_id', 'v_nom', 'in_building'],
            'lines_columns': ['name', 'bus0', 'bus1', 'length', 'r', 'x', 's_nom', 'num_parallel', 'type_info'],
            'transformer_columns': ['name', 'bus0', 'bus1', 's_nom', 'r', 'x', 'type'],
            'generators_columns': ['name', 'bus', 'control', 'p_nom', 'type', 'weather_cell_id', 'subtype'],
            'loads_columns': ['name', 'bus', 'peak_load', 'annual_consumption', 'sector']}
    # initialize dataframes
    buses_df = pd.DataFrame(columns=cols['buses_columns'])
    lines_df = pd.DataFrame(columns=cols['lines_columns'])
    transformer_df = pd.DataFrame(columns=cols['transformer_columns'])
    generators_df = pd.DataFrame(columns=cols['generators_columns'])
    loads_df = pd.DataFrame(columns=cols['loads_columns'])
    return buses_df, generators_df, lines_df, loads_df, transformer_df

def fill_mvgd_component_dataframes(grid_district, buses_df, generators_df, lines_df, loads_df, transformer_df, only_export_mv = False,
                                    return_time_varying_data = False):
    srid = str(int(cfg_ding0.get('geo', 'srid')))
    # fill dataframes
    network_df = pd.DataFrame(
        {'name': grid_district.id_db, 'srid': srid, 'mv_grid_district_geom': grid_district.geo_data,
         'mv_grid_district_population': 0}).set_index('name')
    # add mv grid components
    mv_grid = grid_district.mv_grid
    mv_components, mv_component_data = fill_component_dataframes(mv_grid, buses_df, lines_df, transformer_df, generators_df, loads_df,
                                              only_export_mv, return_time_varying_data)
    if not only_export_mv:
        # add lv grid components
        for lv_load_area in grid_district.lv_load_areas():
            for lv_grid_district in lv_load_area.lv_grid_districts():
                lv_grid = lv_grid_district.lv_grid
                lv_components_tmp, lv_component_data = fill_component_dataframes(lv_grid, buses_df, lines_df, transformer_df,
                                                              generators_df, loads_df, return_time_varying_data)
                mv_components = merge_two_dicts_of_dataframes(mv_components, lv_components_tmp)
                mv_component_data = merge_two_dicts_of_dataframes(mv_component_data,lv_component_data)
    return mv_components, network_df, mv_component_data

def fill_component_dataframes(grid, buses_df, lines_df, transformer_df, generators_df, loads_df, only_export_mv = False,
                              return_time_varying_data = False):
    '''
    Parameters
    ----------
    grid: GridDing0
        Grid that is exported
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name,v_nom,geom,mv_grid_id,lv_grid_id,in_building
    lines_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of lines with entries name,bus0,bus1,length,x,r,s_nom,num_parallel,type
    transformer_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of trafos with entries name,bus0,bus1,x,r,s_nom,type
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name,bus,control,p_nom,type,weather_cell_id,subtype
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name,bus,peak_load,sector
    Returns
    -------
    :obj:`dict`
        Dictionary of component Dataframes 'Bus', 'Generator', 'Line', 'Load', 'Transformer'
    '''
    nodes = grid._graph.nodes()

    edges = [edge for edge in list(grid.graph_edges())
             if (edge['adj_nodes'][0] in nodes and not isinstance(
            edge['adj_nodes'][0], LVLoadAreaCentreDing0))
             and (edge['adj_nodes'][1] in nodes and not isinstance(
            edge['adj_nodes'][1], LVLoadAreaCentreDing0))]


    for trafo in grid.station()._transformers:
        trafo_type = str(int(trafo.s_max_a/1e3))+ ' MVA 110/10 kV'
        transformer_df = append_transformers_df(transformer_df, trafo, trafo_type)

    node_components, component_data = nodes_to_dict_of_dataframes_for_csv_export(grid, nodes, buses_df, generators_df,
                                                                 loads_df, transformer_df, only_export_mv, return_time_varying_data)
    branch_components = edges_to_dict_of_dataframes_for_csv_export(edges, lines_df)
    components = merge_two_dicts(node_components, branch_components)
    return components, component_data

def nodes_to_dict_of_dataframes_for_csv_export(grid, nodes, buses_df, generators_df, loads_df, transformer_df, only_export_mv = False,
                                               return_time_varying_data = False):
    """
    Creates dictionary of dataframes containing grid

    Parameters
    ----------
    grid: ding0.Network
    nodes: :obj:`list` of ding0 grid components objects
        Nodes of the grid graph
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
            Dataframe of buses with entries name,v_nom,geom,mv_grid_id,lv_grid_id,in_building
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name,bus,control,p_nom,type,weather_cell_id,subtype
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name,bus,peak_load,sector
    only_export_mv: bool
        Bool that indicates whether only mv grid should be exported, per default lv grids are exported too

    Returns:
    components: dict of :pandas:`pandas.DataFrame<dataframe>`
        DataFrames contain components attributes. Dict is keyed by components
        type
    """

    srid = int(cfg_ding0.get('geo', 'srid'))
    # check if there are islanded nodes which do not belong to aggregated load area
    for isl_node in grid.graph_isolated_nodes():
        if isinstance(isl_node, CircuitBreakerDing0):
            continue
        elif isl_node.lv_load_area.is_aggregated:
            continue
        else:
            raise Exception("{} is isolated node. Please check.".format(repr(isl_node)))

    # initialise DataFrames for time varying elements, load all necessary values
    components_data = {}
    if return_time_varying_data:
        conf = {}
        conf['kw2mw'] = 1e-3
        cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
        cos_phi_load_mode = cfg_ding0.get('assumptions', 'cos_phi_load_mode')
        cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
        cos_phi_feedin_mode = cfg_ding0.get('assumptions', 'cos_phi_gen_mode')

        conf['load_in_generation_case'] = cfg_ding0.get('assumptions',
                                                'load_in_generation_case')
        conf['generation_in_load_case'] = cfg_ding0.get('assumptions',
                                                'generation_in_load_case')

        conf['Q_factor_load'] = q_sign(cos_phi_load_mode, 'load') * tan(acos(cos_phi_load))
        conf['Q_factor_generation'] = q_sign(cos_phi_feedin_mode, 'generator') * tan(acos(cos_phi_feedin))

        voltage_set_slack = cfg_ding0.get("mv_routing_tech_constraints",
                                          "mv_station_v_level_operation")
        bus_v_mag_set_df = pd.DataFrame(columns=['name', 'temp_id', 'v_mag_pu_set'])
        generator_pq_set_df = pd.DataFrame(columns=['name', 'temp_id', 'p_set', 'q_set'])
        load_pq_set_df = pd.DataFrame(columns=['name', 'temp_id', 'p_set', 'q_set'])

    for node in nodes:
        if node not in grid.graph_isolated_nodes():
            # buses only
            if (isinstance(node, MVCableDistributorDing0) or isinstance(node, LVCableDistributorDing0)):
                buses_df = append_buses_df(buses_df, grid, node, srid)
                # add time varying elements
                if return_time_varying_data:
                    bus_v_mag_set_df = append_bus_v_mag_set_df(bus_v_mag_set_df, node)
      
            # slack generator
            elif isinstance(node, MVStationDing0):
                # add dummy generator
                slack = pd.Series({'name':('_'.join(['MV', str(grid.id_db), 'slack'])),
                                   'bus':node.pypsa_bus_id, 'control':'Slack', 'p_nom':0, 'type': 'station',
                                   'subtype':'mv_station'})
                generators_df = generators_df.append(slack, ignore_index=True)
                # add HV side bus
                bus_HV = pd.Series({'name':node.pypsa_bus0_id, 'v_nom':110,
                                    'geom':from_shape(node.geo_data, srid=srid),'mv_grid_id': grid.id_db, 
                                    'in_building': False})
                buses_df = buses_df.append(bus_HV,ignore_index=True)
                # add MV side bus
                buses_df = append_buses_df(buses_df, grid, node, srid)
                # add time varying elements
                if return_time_varying_data:
                    slack_v_mag = pd.Series({'name':node.pypsa_bus0_id, 'temp_id':1,
                                             'v_mag_pu_set': [voltage_set_slack, voltage_set_slack]})
                    bus_v_mag_set_df = bus_v_mag_set_df.append(slack_v_mag, ignore_index=True)
                    slack_v_mag['name'] = node.pypsa_bus_id
                    bus_v_mag_set_df = bus_v_mag_set_df.append(slack_v_mag, ignore_index=True)

            # other generators
            elif isinstance(node, GeneratorDing0):
                generators_df = append_generators_df(generators_df, node)
                buses_df = append_buses_df(buses_df, grid, node, srid)
                # add time varying elements
                if return_time_varying_data:
                    bus_v_mag_set_df = append_bus_v_mag_set_df(bus_v_mag_set_df, node)
                    generator_pq_set_df = append_generator_pq_set_df(conf, generator_pq_set_df, node)

            elif isinstance(node, LoadDing0):
                # choose sector with highest consumption and assign sector accordingly #Todo: replace when loads are seperated in a cleaner way (retail, industrial)
                sorted_consumption = [(value, key) for key, value in node.consumption.items()]
                sector = max(sorted_consumption)[1]
                # add load
                load = pd.Series({'name': repr(node), 'bus': node.pypsa_bus_id,
                                  'peak_load': node.peak_load, 'sector': sector})
                loads_df = loads_df.append(load, ignore_index=True)
                buses_df = append_buses_df(buses_df,grid,node,srid)
                # add time varying elements
                if return_time_varying_data:
                    bus_v_mag_set_df = append_bus_v_mag_set_df(bus_v_mag_set_df, node)
                    load_pq_set_df = append_load_pq_set_df(conf, load_pq_set_df, node)

            # aggregated load at hv/mv substation
            elif isinstance(node, LVLoadAreaCentreDing0):
                if (node.lv_load_area.peak_load!=0):
                    if return_time_varying_data:
                        loads_df, generators_df, load_pq_set_df, generator_pq_set_df = append_load_areas_to_df(loads_df, generators_df, node, return_time_varying_data, conf=conf,
                                                                                                               load_pq_set_df=load_pq_set_df, generator_pq_set_df = generator_pq_set_df)
                    else:
                        loads_df, generators_df = append_load_areas_to_df(loads_df, generators_df, node)

            # bus + aggregate load of lv grids (at mv/ls substation)
            elif isinstance(node, LVStationDing0):
                if isinstance(grid, ding0_nw.grids.MVGridDing0): #Todo: remove ding0_nw.grids when functions are thinned out
                    # Aggregated load representing load in LV grid, only needed when LV_grids are not exported
                    if only_export_mv:
                        if return_time_varying_data:
                            loads_df, generators_df, load_pq_set_df, generator_pq_set_df = append_load_areas_to_df(
                                loads_df, generators_df, node, return_time_varying_data, conf=conf,
                                load_pq_set_df=load_pq_set_df, generator_pq_set_df=generator_pq_set_df)
                            bus_v_mag_set_df = append_bus_v_mag_set_df(bus_v_mag_set_df,node, node.pypsa_bus0_id)
                        else:
                            loads_df, generators_df = append_load_areas_to_df(loads_df, generators_df, node)
                        for trafo in node.transformers():
                            transformer_df = append_transformers_df(transformer_df,trafo)
                        # bus at secondary MV-LV transformer side
                        buses_df = append_buses_df(buses_df, grid, node, srid, node.pypsa_bus0_id)
                    # bus at primary MV-LV transformer side
                    buses_df = append_buses_df(buses_df, grid, node, srid)
                    if return_time_varying_data:
                        bus_v_mag_set_df = append_bus_v_mag_set_df(bus_v_mag_set_df,node)
                elif isinstance(grid, ding0_nw.grids.LVGridDing0):
                    # bus at secondary MV-LV transformer side
                    buses_df = append_buses_df(buses_df, grid, node, srid,node.pypsa_bus0_id)
                    if return_time_varying_data:
                        bus_v_mag_set_df = append_bus_v_mag_set_df(bus_v_mag_set_df,node, node.pypsa_bus0_id)
                else: 
                    raise TypeError('Something went wrong. Only LVGridDing0 or MVGridDing0 can be handled as grid.')
            elif isinstance(node, CircuitBreakerDing0):
                # TODO: remove this elif-case if CircuitBreaker are removed from graph
                continue
            else:
                raise TypeError("Node of type", node, "cannot be handled here")
        else:
            continue

    nodal_components = {'Bus': buses_df.set_index('name'),
                        'Generator': generators_df.set_index('name'),
                        'Load': loads_df.set_index('name'),
                        'Transformer': transformer_df.set_index('name')}

    if return_time_varying_data:
        components_data = {'Bus': bus_v_mag_set_df.set_index('name'),
                           'Generator': generator_pq_set_df.set_index('name'),
                           'Load': load_pq_set_df.set_index('name')}
    else:
        components_data = {}

    return nodal_components, components_data


def append_generator_pq_set_df(conf, generator_pq_set_df, node):
    # active and reactive power of generator in load and generation case
    p_set = [node.capacity * node.capacity_factor * conf['kw2mw'] * conf['generation_in_load_case'],
             node.capacity * node.capacity_factor * conf['kw2mw']]
    q_set = [node.capacity * node.capacity_factor * conf['kw2mw'] * conf['Q_factor_generation'] * conf[
        'generation_in_load_case'],
             node.capacity * node.capacity_factor * conf['kw2mw'] * conf['Q_factor_generation']]
    generator_pq_set_df = generator_pq_set_df.append(
        pd.Series({'name': repr(node), 'temp_id': 1, 'p_set': p_set, 'q_set': q_set}), ignore_index=True)
    return generator_pq_set_df


def append_load_pq_set_df(conf, load_pq_set_df, node, node_name = None, peak_load = None):
    if node_name is None:
        node_name = repr(node)
    if peak_load is None:
        peak_load = node.peak_load
    # active and reactive power of load in load and generation case
    p_set = [peak_load * conf['kw2mw'], peak_load * conf['kw2mw'] * conf['load_in_generation_case']]
    q_set = [peak_load * conf['kw2mw'] * conf['Q_factor_load'],
             peak_load * conf['kw2mw'] * conf['Q_factor_load'] * conf['load_in_generation_case']]
    load_pq_set_df = load_pq_set_df.append(
        pd.Series({'name': node_name, 'temp_id': 1, 'p_set': p_set, 'q_set': q_set}), ignore_index=True)
    return load_pq_set_df


def append_bus_v_mag_set_df(bus_v_mag_set_df, node, node_name = None):
    if node_name is None:
        node_name = node.pypsa_bus_id
    bus_v_mag_set_df = bus_v_mag_set_df.append(pd.Series({'name': node_name, 'temp_id': 1,
                                                          'v_mag_pu_set': [1, 1]}),
                                               ignore_index=True)
    return bus_v_mag_set_df


def append_load_areas_to_df(loads_df, generators_df, node, return_time_varying_data = False, **kwargs):
    '''
    Appends lv load area (or single lv grid district) to dataframe of nodes. Each sector (agricultural, industrial, residential, retail)
    is represented by own entry of load. Each generator in underlying grid districts is added as own entry.
    Generators and load are connected to BusBar of the respective grid (LVGridDing0 for LVStationDing0 and MVGridDing0 for LVLoadAreaCentreDing0)

    Parameters
    ----------
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name,bus,peak_load,sector
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name,bus,control,p_nom,type,weather_cell_id,subtype
    node: :obj: ding0 grid components object
        Node, which is either LVStationDing0 or LVLoadAreaCentreDing0

    Returns:
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name,bus,peak_load,sector
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name,bus,control,p_nom,type,weather_cell_id,subtype
    '''
    # set name of bus, name of load and handles load area and grid districts
    if isinstance(node,LVStationDing0):
        name_bus = node.pypsa_bus_id
        name_load = '_'.join(['Load','mvgd' + str(node.grid.grid_district.lv_load_area.mv_grid_district.id_db), 'lac' + str(node.id_db)])
        load_area = node.grid.grid_district
        grid_districts = [load_area]
    elif isinstance(node,LVLoadAreaCentreDing0):
        name_bus = node.grid.station().pypsa_bus_id
        name_load = '_'.join(['Load','mvgd' + str(node.grid.id_db), 'lac' + str(node.id_db)])
        load_area = node.lv_load_area
        grid_districts = load_area.lv_grid_districts()
    else:
        raise TypeError("Only LVStationDing0 or LVLoadAreaCentreDing0 can be inserted into function append_load_areas_to_df.")

    # unpack time varying elements
    if return_time_varying_data:
        conf = kwargs.get('conf', None)
        load_pq_set_df = kwargs.get('load_pq_set_df',None)
        generator_pq_set_df = kwargs.get('generator_pq_set_df', None)

    # Handling of generators
    for lvgd in grid_districts:
        for gen in lvgd.lv_grid.generators():
            generators_df = append_generators_df(generators_df, gen, name_bus=name_bus)
            # add time varying elements
            if return_time_varying_data:
                generator_pq_set_df = append_generator_pq_set_df(conf, generator_pq_set_df, gen)


    # Handling of loads
    sectors = ['agricultural', 'industrial', 'residential', 'retail']
    for sector in sectors:
        if (getattr(load_area, '_'.join(['peak_load', sector]))!= 0):
            if return_time_varying_data:
                loads_df, load_pq_set_df = append_load_area_to_load_df(sector, load_area, loads_df, name_bus, name_load,
                                                       return_time_varying_data, conf= conf, load_pq_set_df=load_pq_set_df)
            else:
                loads_df = append_load_area_to_load_df(sector, load_area, loads_df, name_bus, name_load)

    if return_time_varying_data:
        return loads_df, generators_df, load_pq_set_df, generator_pq_set_df
    else:
        return loads_df, generators_df


def append_load_area_to_load_df(sector, load_area, loads_df, name_bus, name_load, return_time_varying_data = False, **kwargs):
    '''
    Appends LVLoadArea or LVGridDistrict to dataframe of loads in pypsa format.

    Parameters
    ----------
    sector: str
        load sector: 'agricultural', 'industrial', 'residential' or 'retail'
    load_are: :obj: ding0 region
        LVGridDistrictDing0 or LVLoadAreaDing0, load area of which load is to be aggregated and added
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, bus, peak_load, annual_consumption and sector
    name_bus: str
        name of bus to which load is connected
    name_load: str
        name of load

    Returns:
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, bus, peak_load, annual_consumption and sector
    '''
    # get annual consumption
    if isinstance(load_area, LVGridDistrictDing0):
        consumption = getattr(load_area, '_'.join(['sector_consumption', sector]))
    elif isinstance(load_area,LVLoadAreaDing0):
        consumption = 0
        for lv_grid_district in load_area.lv_grid_districts():
            consumption += getattr(lv_grid_district, '_'.join(['sector_consumption', sector]))
    # create and append load to df
    name_load = '_'.join([name_load, sector])
    peak_load = getattr(load_area, '_'.join(['peak_load', sector]))
    load = pd.Series(
        {'name': name_load, 'bus': name_bus,
         'peak_load': peak_load,
         'annual_consumption': consumption, 'sector': sector})
    loads_df = loads_df.append(load, ignore_index=True)
    # handle time varying data
    if return_time_varying_data:
        conf = kwargs.get('conf', None)
        load_pq_set_df = kwargs.get('load_pq_set_df', None)
        load_pq_set_df = append_load_pq_set_df(conf, load_pq_set_df, None, name_load, peak_load)
        return loads_df,load_pq_set_df
    else:
        return loads_df


def append_generators_df(generators_df, node, name_bus = None):
    '''
    Appends generator to dataframe of generators in pypsa format.

    Parameters
    ----------
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name,bus,control,p_nom,type,weather_cell_id,subtype
    node: :obj: ding0 grid components object
        GeneratorDing0

    Returns:
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name,bus,control,p_nom,type,weather_cell_id,subtype
    '''
    if isinstance(node,GeneratorFluctuatingDing0):
        weather_cell_id = node.weather_cell_id
    else:
        weather_cell_id = np.NaN
    if name_bus is None:
        name_bus = node.pypsa_bus_id
    generator = pd.Series({'name':repr(node),
                           'bus': name_bus, 'control':'PQ', 'p_nom':(node.capacity * node.capacity_factor),
                           'type':node.type, 'subtype':node.subtype, 'weather_cell_id':weather_cell_id})
    generators_df = generators_df.append(generator, ignore_index=True)
    return generators_df


def append_buses_df(buses_df, grid, node, srid, node_name =''):
    '''
    Appends buses to dataframe of buses in pypsa format.

    Parameters
    ----------
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name,v_nom,geom,mv_grid_id,lv_grid_id,in_building
    grid: ding0.Network
    node: :obj: ding0 grid components object
    srid: int
    node_name: str
        name of node, per default is set to node.pypsa_bus_id

    Returns:
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name,v_nom,geom,mv_grid_id,lv_grid_id,in_building
    '''
    # set default name of node
    if node_name == '':
        node_name = node.pypsa_bus_id
    # check if node is in building
    if isinstance(node, LVCableDistributorDing0):
        in_building = node.in_building
    else:
        in_building = False
    # set geodata, if existing
    geo = np.NaN
    if isinstance(node.geo_data,Point):
        geo = from_shape(node.geo_data, srid=srid)
    #set grid_ids
    if isinstance(grid, ding0_nw.grids.MVGridDing0):
        mv_grid_id = grid.id_db
        lv_grid_id = np.NaN
    elif isinstance(grid, ding0_nw.grids.LVGridDing0):
        mv_grid_id = grid.grid_district.lv_load_area.mv_grid_district.mv_grid.id_db
        lv_grid_id = grid.id_db
    else:
        raise TypeError('Something went wrong, only MVGridDing0 and LVGridDing0 should be inserted as grid.')
    # create bus dataframe
    bus = pd.Series({'name': node_name,'v_nom':grid.v_level, 'geom':geo,
                     'mv_grid_id':mv_grid_id,'lv_grid_id':lv_grid_id, 'in_building': in_building})
    buses_df = buses_df.append(bus, ignore_index=True)
    return buses_df

def append_transformers_df(transformers_df, trafo, type = np.NaN):
    '''
    Appends transformer to dataframe of buses in pypsa format.

    Parameters
    ----------
    transformers_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of trafos with entries name,bus0,bus1,x,r,s_nom,type
    trafo: :obj:TransformerDing0
        Transformer to be added
    name_trafo: str
        Name of transformer
    name_bus1: str
        name of secondary bus

    Returns:
    transformers_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of trafos with entries name,bus0,bus1,x,r,s_nom,type
    '''
    trafo_tmp = pd.Series({'name': repr(trafo), 'bus0':trafo.grid.station().pypsa_bus0_id,
                           'bus1':trafo.grid.station().pypsa_bus_id, 'x':trafo.x_pu, 'r':trafo.r_pu,
                           's_nom':trafo.s_max_a, 'type': type})
    transformers_df = transformers_df.append(trafo_tmp,ignore_index=True)
    return transformers_df

def edges_to_dict_of_dataframes(grid, edges):
    """
    Export edges to DataFrame

    Parameters
    ----------
    grid: ding0.Network
    edges: :obj:`list`
        Edges of Ding0.Network graph

    Returns
    -------
    edges_dict: dict
    """
    freq = cfg_ding0.get('assumptions', 'frequency')
    omega = 2 * pi * freq
    srid = int(cfg_ding0.get('geo', 'srid'))

    lines = {'line_id': [], 'bus0': [], 'bus1': [], 'x': [], 'r': [],
             's_nom': [], 'length': [], 'cables': [], 'geom': [],
             'grid_id': []}

    # iterate over edges and add them one by one
    for edge in edges:

        line_name = repr(edge['branch'])

        # TODO: find the real cause for being L, C, I_th_max type of Series
        if (isinstance(edge['branch'].type['L_per_km'], Series) or#warum wird hier c abgefragt?
                isinstance(edge['branch'].type['C_per_km'], Series)):
            x_per_km = omega * edge['branch'].type['L_per_km'].values[0] * 1e-3
        else:

            x_per_km = omega * edge['branch'].type['L_per_km'] * 1e-3

        if isinstance(edge['branch'].type['R_per_km'], Series):
            r_per_km = edge['branch'].type['R_per_km'].values[0]
        else:
            r_per_km = edge['branch'].type['R_per_km']

        if (isinstance(edge['branch'].type['I_max_th'], Series) or
                isinstance(edge['branch'].type['U_n'], Series)):
            s_nom = sqrt(3) * edge['branch'].type['I_max_th'].values[0] * \
                    edge['branch'].type['U_n'].values[0]
        else:
            s_nom = sqrt(3) * edge['branch'].type['I_max_th'] * \
                    edge['branch'].type['U_n']

        # get lengths of line
        l = edge['branch'].length / 1e3

        lines['line_id'].append(line_name)
        lines['bus0'].append(edge['adj_nodes'][0].pypsa_bus_id)
        lines['bus1'].append(edge['adj_nodes'][1].pypsa_bus_id)
        lines['x'].append(x_per_km * l)
        lines['r'].append(r_per_km * l)
        lines['s_nom'].append(s_nom)
        lines['length'].append(l)
        lines['cables'].append(3)
        lines['geom'].append(from_shape(
            LineString([edge['adj_nodes'][0].geo_data,
                        edge['adj_nodes'][1].geo_data]),
            srid=srid))
        lines['grid_id'].append(grid.id_db)

    return {'Line': DataFrame(lines).set_index('line_id')}

def edges_to_dict_of_dataframes_for_csv_export(edges, lines_df):
    """
    Export edges to DataFrame

    Parameters
    ----------
    edges: :obj:`list`
        Edges of Ding0.Network graph
    lines_df: :pandas:`pandas.DataFrame<dataframe>`
            Dataframe of lines with entries name,bus0,bus1,length,x,r,s_nom,num_parallel,type
        

    Returns
    -------
    edges_dict: dict
    """

    # iterate over edges and add them one by one
    for edge in edges:
        if not edge['branch'].connects_aggregated:
            lines_df = append_lines_df(edge, lines_df)
        else:
            node = edge['adj_nodes']

        if isinstance(edge['adj_nodes'][0], LVLoadAreaCentreDing0) or isinstance(edge['adj_nodes'][1],LVLoadAreaCentreDing0):
            print()

    return {'Line': lines_df.set_index('name')}


def append_lines_df(edge, lines_df):
    freq = cfg_ding0.get('assumptions', 'frequency')
    omega = 2 * pi * freq
    # TODO: find the real cause for being L, C, I_th_max type of Series
    if (isinstance(edge['branch'].type['L_per_km'], Series)):
        x_per_km = omega * edge['branch'].type['L_per_km'].values[0] * 1e-3
    else:

        x_per_km = omega * edge['branch'].type['L_per_km'] * 1e-3
    if isinstance(edge['branch'].type['R_per_km'], Series):
        r_per_km = edge['branch'].type['R_per_km'].values[0]
    else:
        r_per_km = edge['branch'].type['R_per_km']
    if (isinstance(edge['branch'].type['I_max_th'], Series) or
            isinstance(edge['branch'].type['U_n'], Series)):
        s_nom = sqrt(3) * edge['branch'].type['I_max_th'].values[0] * \
                edge['branch'].type['U_n'].values[0]
    else:
        s_nom = sqrt(3) * edge['branch'].type['I_max_th'] * \
                edge['branch'].type['U_n']
    # get lengths of line
    length = edge['branch'].length / 1e3
    #Todo: change into same format
    if 'name' in edge['branch'].type:
        type = edge['branch'].type['name']
    else:
        type = edge['branch'].type.name

    line = pd.Series({'name':repr(edge['branch']),'bus0':edge['adj_nodes'][0].pypsa_bus_id, 'bus1':edge['adj_nodes'][1].pypsa_bus_id,
                      'x':x_per_km * length, 'r':r_per_km * length, 's_nom':s_nom, 'length':length, 
                      'num_parallel':1, 'type_info':type})
    lines_df = lines_df.append(line, ignore_index=True)
    return lines_df


def run_powerflow_onthefly(components, components_data, grid, export_pypsa_dir=None, debug=False, export_result_dir = None):
    """
    Run powerflow to test grid stability

    Two cases are defined to be tested here:
     i) load case
     ii) feed-in case

    Parameters
    ----------
    components: dict of :pandas:`pandas.DataFrame<dataframe>`
    components_data: dict of :pandas:`pandas.DataFrame<dataframe>`
    export_pypsa_dir: :obj:`str`
        Sub-directory in output/debug/grid/ where csv Files of PyPSA network are exported to.
        Export is omitted if argument is empty.
    """

    scenario = cfg_ding0.get("powerflow", "test_grid_stability_scenario")
    start_hour = cfg_ding0.get("powerflow", "start_hour")
    end_hour = cfg_ding0.get("powerflow", "end_hour")

    # choose temp_id
    temp_id_set = 1
    timesteps = 2
    start_time = datetime(1970, 1, 1, 00, 00, 0)
    resolution = 'H'

    # inspect grid data for integrity
    if debug:
        data_integrity(components, components_data)

    # define investigated time range
    timerange = DatetimeIndex(freq=resolution,
                              periods=timesteps,
                              start=start_time)

    # TODO: Instead of hard coding PF config, values from class PFConfigDing0 can be used here.
    #Todo: REMOVE ONLY FOR DEBUGGING
    components['Bus'] = components['Bus'].drop('Busbar_mvgd460_HV')
    components_data['Bus'] = components_data['Bus'].drop('Busbar_mvgd460_HV')
    components['Transformer'] = components['Transformer'].drop(['Transformer_mv_grid_460_1','Transformer_mv_grid_460_2'])
    # create PyPSA powerflow problem
    network, snapshots = create_powerflow_problem(timerange, components)

    # import pq-sets
    for key in ['Load', 'Generator']:
        for attr in ['p_set', 'q_set']:
            # catch MV grid districts without generators
            if not components_data[key].empty:
                series = transform_timeseries4pypsa(components_data[key][
                                                        attr].to_frame(),
                                                    timerange,
                                                    column=attr)
                import_series_from_dataframe(network,
                                             series,
                                             key,
                                             attr)
    series = transform_timeseries4pypsa(components_data['Bus']
                                        ['v_mag_pu_set'].to_frame(),
                                        timerange,
                                        column='v_mag_pu_set')

    import_series_from_dataframe(network,
                                 series,
                                 'Bus',
                                 'v_mag_pu_set')

    # add coordinates to network nodes and make ready for map plotting
    # network = add_coordinates(network)

    # start powerflow calculations
    network.pf(snapshots)

    # # make a line loading plot
    # # TODO: make this optional
    # plot_line_loading(network, timestep=0,
    #                   filename='Line_loading_load_case.png')
    # plot_line_loading(network, timestep=1,
    #                   filename='Line_loading_feed-in_case.png')

    # process results
    bus_data, line_data = process_pf_results(network)
    if export_result_dir:
        bus_data.to_csv(os.path.join(export_result_dir,'bus_data.csv'))
        line_data.to_csv(os.path.join(export_result_dir, 'line_data.csv'))
    # assign results data to graph
    assign_bus_results(grid, bus_data)
    assign_line_results(grid, line_data)

    # export network if directory is specified
    if export_pypsa_dir:
        export_to_dir(network, export_dir=export_pypsa_dir)


def data_integrity(components, components_data):
    """
    Check grid data for integrity

    Parameters
    ----------
    components: dict
        Grid components
    components_data: dict
        Grid component data (such as p,q and v set points)

    Returns
    -------
    """

    data_check = {}

    for comp in ['Bus', 'Load']:  # list(components_data.keys()):
        data_check[comp] = {}
        data_check[comp]['length_diff'] = len(components[comp]) - len(
            components_data[comp])

    # print short report to user and exit program if not integer
    for comp in list(data_check.keys()):
        if data_check[comp]['length_diff'] != 0:
            logger.exception("{comp} data is invalid. You supplied {no_comp} {comp} "
                  "objects and {no_data} datasets. Check you grid data "
                  "and try again".format(comp=comp,
                                         no_comp=len(components[comp]),
                                         no_data=len(components_data[comp])))
            sys.exit(1)

def process_pf_results(network):
    """

    Parameters
    ----------
    network: pypsa.Network

    Returns
    -------
    bus_data: :pandas:`pandas.DataFrame<dataframe>`
        Voltage level results at buses
    line_data: :pandas:`pandas.DataFrame<dataframe>`
        Resulting apparent power at lines
    """

    bus_data = {'bus_id': [], 'v_mag_pu': []}
    line_data = {'line_id': [], 'p0': [], 'p1': [], 'q0': [], 'q1': []}

    # create dictionary of bus results data
    for col in list(network.buses_t.v_mag_pu.columns):
        bus_data['bus_id'].append(col)
        bus_data['v_mag_pu'].append(network.buses_t.v_mag_pu[col].tolist())

    # create dictionary of line results data
    for col in list(network.lines_t.p0.columns):
        line_data['line_id'].append(col)
        line_data['p0'].append(network.lines_t.p0[col].tolist())
        line_data['p1'].append(network.lines_t.p1[col].tolist())
        line_data['q0'].append(network.lines_t.q0[col].tolist())
        line_data['q1'].append(network.lines_t.q1[col].tolist())

    return DataFrame(bus_data).set_index('bus_id'), \
           DataFrame(line_data).set_index('line_id')


def assign_bus_results(grid, bus_data):
    """
    Write results obtained from PF to graph

    Parameters
    ----------
    grid: ding0.network
    bus_data: :pandas:`pandas.DataFrame<dataframe>`
        DataFrame containing voltage levels obtained from PF analysis
    """

    # iterate of nodes and assign voltage obtained from power flow analysis
    for node in grid._graph.nodes():
        # check if node is connected to graph
        if (node not in grid.graph_isolated_nodes()
            and not isinstance(node,
                               LVLoadAreaCentreDing0)):
            if isinstance(node, LVStationDing0):
                node.voltage_res = bus_data.loc[node.pypsa_bus_id, 'v_mag_pu']
            elif isinstance(node, (LVStationDing0, LVLoadAreaCentreDing0)):
                if node.lv_load_area.is_aggregated:
                    node.voltage_res = bus_data.loc[node.pypsa_bus_id, 'v_mag_pu']
            elif not isinstance(node, CircuitBreakerDing0):
                node.voltage_res = bus_data.loc[node.pypsa_bus_id, 'v_mag_pu']
            else:
                logger.warning("Object {} has been skipped while importing "
                               "results!")


def assign_line_results(grid, line_data):
    """
    Write results obtained from PF to graph

    Parameters
    -----------
    grid: ding0.network
    line_data: :pandas:`pandas.DataFrame<dataframe>`
        DataFrame containing active/reactive at nodes obtained from PF analysis
    """

    package_path = ding0.__path__[0]

    edges = [edge for edge in grid.graph_edges()
             if (edge['adj_nodes'][0] in grid._graph.nodes() and not isinstance(
            edge['adj_nodes'][0], LVLoadAreaCentreDing0))
             and (
             edge['adj_nodes'][1] in grid._graph.nodes() and not isinstance(
                 edge['adj_nodes'][1], LVLoadAreaCentreDing0))]

    decimal_places = 6
    for edge in edges:
        s_res = [
            round(sqrt(
                max(abs(line_data.loc[repr(edge['branch']), 'p0'][0]),
                    abs(line_data.loc[repr(edge['branch']), 'p1'][0])) ** 2 +
                max(abs(line_data.loc[repr(edge['branch']), 'q0'][0]),
                    abs(line_data.loc[repr(edge['branch']), 'q1'][0])) ** 2),decimal_places),
            round(sqrt(
                max(abs(line_data.loc[repr(edge['branch']), 'p0'][1]),
                    abs(line_data.loc[repr(edge['branch']), 'p1'][1])) ** 2 +
                max(abs(line_data.loc[repr(edge['branch']), 'q0'][1]),
                    abs(line_data.loc[repr(edge['branch']), 'q1'][1])) ** 2),decimal_places)]

        edge['branch'].s_res = s_res


def init_pypsa_network(time_range_lim):
    """
    Instantiate PyPSA network
    Parameters
    ----------
    time_range_lim:
    Returns
    -------
    network: PyPSA network object
        Contains powerflow problem
    snapshots: iterable
        Contains snapshots to be analyzed by powerplow calculation
    """
    network = Network()
    network.set_snapshots(time_range_lim)
    snapshots = network.snapshots

    return network, snapshots


def transform_timeseries4pypsa(timeseries, timerange, column=None):
    """
    Transform pq-set timeseries to PyPSA compatible format
    Parameters
    ----------
    timeseries: Pandas DataFrame
        Containing timeseries
    Returns
    -------
    pypsa_timeseries: Pandas DataFrame
        Reformated pq-set timeseries
    """
    timeseries.index = [str(i) for i in timeseries.index]

    if column is None:
        pypsa_timeseries = timeseries.apply(
            Series).transpose().set_index(timerange)
    else:
        pypsa_timeseries = timeseries[column].apply(
            Series).transpose().set_index(timerange)

    return pypsa_timeseries


def create_powerflow_problem(timerange, components):
    """
    Create PyPSA network object and fill with data
    Parameters
    ----------
    timerange: Pandas DatetimeIndex
        Time range to be analyzed by PF
    components: dict
    Returns
    -------
    network: PyPSA powerflow problem object
    """

    # initialize powerflow problem
    network, snapshots = init_pypsa_network(timerange)

    # add components to network
    for component in components.keys():
        network.import_components_from_dataframe(components[component],
                                                 component)

    return network, snapshots
