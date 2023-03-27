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
from ding0.core.network.loads import LVLoadDing0, MVLoadDing0
from ding0.core.network import LoadDing0, CircuitBreakerDing0, \
    GeneratorDing0, GeneratorFluctuatingDing0, TransformerDing0, CableDistributorDing0
from ding0.core.structure.regions import LVLoadAreaCentreDing0, \
    LVGridDistrictDing0, LVLoadAreaDing0
from ding0.core.powerflow import q_sign
from ding0.core.network.cable_distributors import LVCableDistributorDing0, MVCableDistributorDing0
from ding0.core import network as ding0_nw
from ding0.tools.tools import merge_two_dicts_of_dataframes
from ding0.grid.lv_grid.build_grid import select_transformers

from geoalchemy2.shape import from_shape
from math import tan, acos, pi, sqrt
from pandas import Series, DataFrame, date_range
from pypsa.io import import_series_from_dataframe
from pypsa import Network
from shapely.geometry import Point

from datetime import datetime
import sys
import os
import logging
import pandas as pd
import numpy as np
from networkx import connected_components


if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString

logger = logging.getLogger(__name__)

# parameter to enable sector-specific consumption data
# for egon data buildings set to False, because
# dataset do not contain consumptional data
consider_consumption = False

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


def initialize_component_dataframes():
    """
    Initializes and returns empty component dataframes

    Returns
    -------
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name, v_nom, geom, mv_grid_id,
        lv_grid_id, in_building
    lines_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of lines with entries name, bus0, bus1, length, x, r, s_nom,
        num_parallel, type, geometry (LineString)
    transformer_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of trafos with entries name, bus0, bus1, x, r, s_nom, type
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name, bus, control, p_nom, type,
        weather_cell_id, subtype
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, bus, p_set,
        annual_consumption, sector
    """
    cols = {'buses_columns': ['name', 'x', 'y', 'mv_grid_id',
                              'lv_grid_id', 'v_nom', 'in_building'],
            'lines_columns': ['name', 'bus0', 'bus1', 'length', 'r', 'x',
                              's_nom', 'num_parallel', 'kind', 'type_info', 'geometry'],
            'transformer_columns': ['name', 'bus0', 'bus1', 's_nom', 'r',
                                    'x', 'type', 'type_info'],
            'generators_columns': ['name', 'bus', 'control', 'p_nom', 'type',
                                   'weather_cell_id', 'subtype'],
            'loads_columns': ['name', 'bus', 'p_set', 'building_id',
                              'annual_consumption', 'sector']}
    # initialize dataframes
    buses_df = pd.DataFrame(columns=cols['buses_columns'])
    lines_df = pd.DataFrame(columns=cols['lines_columns'])
    transformer_df = pd.DataFrame(columns=cols['transformer_columns'])
    generators_df = pd.DataFrame(columns=cols['generators_columns'])
    loads_df = pd.DataFrame(columns=cols['loads_columns'])
    return buses_df, generators_df, lines_df, loads_df, transformer_df


def fill_mvgd_component_dataframes(
        mv_grid_district,
        buses_df,
        generators_df,
        lines_df,
        loads_df,
        transformer_df,
        only_export_mv=False,
        return_time_varying_data=False
):
    """
    Returns component and if necessary time varying data for power flow
    or csv export of inserted mv grid district

    Parameters
    ----------
    mv_grid_district: :class:`~.ding0.core.structure.regions.MVGridDistrictDing0`
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name, v_nom, geom, mv_grid_id,
        lv_grid_id, in_building
    lines_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of lines with entries name, bus0, bus1, length, x, r, s_nom,
        num_parallel, type
    transformer_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of trafos with entries name, bus0, bus1, x, r, s_nom, type
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name, bus, control, p_nom, type,
        weather_cell_id, subtype
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, bus, p_set, building_id,
        annual_consumption, sector
    only_export_mv: :obj:`bool`
        Bool that determines export modes for grid district,
        if True only mv grids are exported with lv grids aggregated at
        respective station,
        if False lv grids are fully exported
    return_time_varying_data: :obj:`bool`
        States whether time varying data needed for power flow calculations
        are constructed as well. Set to True to run power flow, set to False
        to export network to csv.

    Returns
    -------
    mv_components: :obj:`dict`
        Dictionary of component Dataframes 'Bus', 'Generator', 'Line',
        'Load', 'Transformer', 'Switch'
    network_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of network containing name, srid, geom and population
    mv_component_data: :obj:`dict`
        Dictionary of component Dataframes 'Bus', 'Generator', 'Load',
        needed for power flow calculations
    """
    srid = str(int(cfg_ding0.get('geo', 'srid')))
    # fill dataframes

    # fill grids_df with population and geoms of grids
    grids_df = pd.DataFrame(
        columns=["name", "grid_district_population", "srid", "area", "grid_district_geom"]
    ).set_index("name")
    grids_df.loc[f"mvgd_{mv_grid_district.id_db}"] = (0, srid, mv_grid_district.geo_data.area, mv_grid_district.geo_data)
    mvgd_population = 0
    for lv_load_area in mv_grid_district.lv_load_areas():
        if pd.isna(lv_load_area.population):
            lv_load_area_population = 0
            logger.warning(f"Population of {lv_load_area} is NaN")
        else:
            lv_load_area_population = lv_load_area.population
        grids_df.loc[f"lvgd_{lv_load_area.id_db}"] = (lv_load_area.population, srid, lv_load_area.geo_area.area, lv_load_area.geo_area)
        mvgd_population += lv_load_area_population
    grids_df.loc[f"mvgd_{mv_grid_district.id_db}", "grid_district_population"] = mvgd_population


    network_df = pd.DataFrame(
        {'name': [mv_grid_district.id_db], 'srid': [srid],
         'mv_grid_district_geom': [mv_grid_district.geo_data],
         'mv_grid_district_population': [int(mvgd_population)]}).set_index('name')

    # add mv grid components
    mv_grid = mv_grid_district.mv_grid
    mv_components, mv_component_data = fill_component_dataframes(
        mv_grid,
        buses_df,
        lines_df,
        transformer_df,
        generators_df,
        loads_df,
        only_export_mv,
        return_time_varying_data
    )
    # move hvmv transformer to own entry to handle differently and ignore in 
    # power flow
    hvmv_transformers = mv_components['Transformer'].loc[
        ~mv_components['Transformer'].bus0.isin(mv_components['Bus'].index)
    ]
    mv_components['HVMV_Transformer'] = hvmv_transformers
    mv_components['Transformer'].drop(hvmv_transformers.index, inplace=True)
    logger.info('MV grid {} exported to pypsa format.'.format(
        str(mv_grid_district.id_db))
    )
    if not only_export_mv:
        # add lv grid components
        for lv_load_area in mv_grid_district.lv_load_areas():
            for lv_grid_district in lv_load_area.lv_grid_districts():
                lv_grid = lv_grid_district.lv_grid
                lv_components_tmp, lv_component_data = fill_component_dataframes(
                    lv_grid,
                    buses_df,
                    lines_df,
                    transformer_df,
                    generators_df,
                    loads_df,
                    only_export_mv,
                    return_time_varying_data
                )
                mv_components = merge_two_dicts_of_dataframes(
                    mv_components,
                    lv_components_tmp
                )
                mv_component_data = merge_two_dicts_of_dataframes(
                    mv_component_data,
                    lv_component_data
                )
                logger.info('LV grid {} exported to pypsa format.'.format(
                    str(lv_grid.id_db))
                )
    return mv_components, network_df, grids_df, mv_component_data


def fill_component_dataframes(
        grid,
        buses_df,
        lines_df,
        transformer_df,
        generators_df,
        loads_df,
        only_export_mv=False,
        return_time_varying_data=False
):
    """
    Returns component and if necessary time varying data for power flow
    or csv export of inserted mv or lv grid

    Parameters
    ----------
    grid: :class:`~.ding0.core.network.GridDing0`
        Grid that is exported
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name, v_nom, geom, mv_grid_id,
        lv_grid_id, in_building
    lines_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of lines with entries name, bus0, bus1, length, x, r, s_nom,
        num_parallel, type_info
    transformer_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of trafos with entries name, bus0, bus1, x, r, s_nom, type
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name, bus, control, p_nom, type,
        weather_cell_id, subtype
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, bus, p_set,
        annual_consumption, sector
    only_export_mv: :obj:`bool`
    return_time_varying_data: :obj:`bool`
        States whether time varying data needed for power flow calculations
        are constructed as well. Set to True to run power flow, set to False
        to export network to csv.

    Returns
    -------
    components: :obj:`dict`
        Dictionary of component Dataframes 'Bus', 'Generator', 'Line', 'Load',
        'Transformer', 'Switch'
    component_data: :obj:`dict`
        Dictionary of component Dataframes 'Bus', 'Generator', 'Load',
        needed for power flow calculations
    """

    # fill list of open circuit breakers
    open_circuit_breakers = []
    if hasattr(grid, '_circuit_breakers'):
        for circuit_breaker in grid.circuit_breakers():
            if circuit_breaker.status == 'open':
                open_circuit_breakers.append(repr(circuit_breaker))
                circuit_breaker.close()
    # get all grid nodes
    nodes = grid.graph.nodes()
    # get all grid edges
    edges = [
        edge for edge in list(grid.graph_edges())
        if (
                   edge['adj_nodes'][0] in nodes
                   and not isinstance(edge['adj_nodes'][0], LVLoadAreaCentreDing0)
           )
           and
           (
                   edge['adj_nodes'][1] in nodes
                   and not isinstance(edge['adj_nodes'][1], LVLoadAreaCentreDing0)
           )
    ]
    # add station transformers to respective dataframe
    for trafo in grid.station()._transformers:
        if trafo.x_pu == None:
            type = '{} MVA 110/{} kV'.format(int(trafo.s_max_a/1e3), grid.v_level)
            transformer_df = append_transformers_df(
                transformer_df,trafo, type
            )
        else:
            transformer_df = append_transformers_df(transformer_df, trafo)
    # handle all nodes and append to respective dataframes
    node_components, component_data = nodes_to_dict_of_dataframes(
        grid,
        nodes,
        buses_df,
        generators_df,
        loads_df,
        transformer_df,
        only_export_mv,
        return_time_varying_data
    )
    # handle all edges and append to respective dataframe
    branch_components = edges_to_dict_of_dataframes(
        edges, lines_df, node_components['Bus']
    )
    # merge node and edges
    components = merge_two_dicts(node_components, branch_components)
    components, component_data = circuit_breakers_to_df(
        grid, components, component_data, open_circuit_breakers, return_time_varying_data
    )
    return components, component_data


def nodes_to_dict_of_dataframes(grid, nodes, buses_df, generators_df, loads_df,
                                transformer_df, only_export_mv=False,
                                return_time_varying_data=False):
    """
    Creates dictionary of dataframes containing grid nodes and transformers

    Parameters
    ----------
    grid: :class:`~.ding0.core.network.GridDing0`
    nodes: :obj:`list` of ding0 grid components objects
        Nodes of the grid graph
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
            Dataframe of buses with entries name, v_nom, geom, mv_grid_id,
            lv_grid_id, in_building
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name, bus, control, p_nom, type,
        weather_cell_id, subtype
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name,bus,p_set, building_id,
        annual_consumption,sector
    transformer_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of trafos with entries name, bus0, bus1, x, r, s_nom, type
    only_export_mv: :obj:`bool`
        Bool that indicates whether only mv grid should be exported,
        per default lv grids are exported too
    return_time_varying_data: :obj:`bool`
        Set to True when running power flow. Then time varying data are
        returned as well.

    Returns
    -------
    components: dict of :pandas:`pandas.DataFrame<dataframe>`
        DataFrames contain components attributes. Dict is keyed by components
        type
    component_data: :obj:`dict`
        Dictionary of component Dataframes 'Bus', 'Generator', 'Load',
        needed for power flow calculations, only exported when
        return_time_varying_data is True empty dict otherwise.
    """

    def next_branch_is_helper_component(node):
        import networkx as nx
        if isinstance(node, LVLoadDing0):
            graph = node.grid.graph
        elif isinstance(node, MVLoadDing0):
            graph = node.grid.mv_grid.graph
        elif isinstance(node, GeneratorDing0):
            if node.lv_grid is not None:
                graph = node.lv_grid.graph
            else:
                graph = node.mv_grid.graph
        else:
            TypeError("No ding0 load!")
        adjacent_edges = list(nx.edges(graph, node))
        if len(adjacent_edges) == 1:
            branch = graph.get_edge_data(
                adjacent_edges[0][0], adjacent_edges[0][1]
            )["branch"]
            if branch.helper_component:
                return list(graph.neighbors(node))[0]
            else:
                return False
        else:
            ValueError("More than one adjacent branch!")

    srid = int(cfg_ding0.get('geo', 'srid'))
    # check if there are islanded nodes which do not belong to aggregated
    # load area
    for isl_node in grid.graph_isolated_nodes():
        if isinstance(isl_node, CircuitBreakerDing0):
            continue
        elif isl_node.lv_load_area.is_aggregated: #TODO consider aggregated
            continue
        elif isinstance(isl_node, LVStationDing0):
            # Todo: This is only a workaround for empty grids (see
            #  build_grid.py line 800). Aim should be to solve the issue
            #  there and remove this exception here afterwards
            if not only_export_mv:
                buses_df = append_buses_df(buses_df, isl_node.grid,
                                           isl_node)
            else:
                continue
        else:
            raise Exception("{} is isolated node. Please check.".
                            format(repr(isl_node)))

    # initialise DataFrames for time varying elements, load all necessary
    # values
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

        conf['Q_factor_load'] = q_sign(cos_phi_load_mode, 'load') * \
                                tan(acos(cos_phi_load))
        conf['Q_factor_generation'] = q_sign(cos_phi_feedin_mode, 'generator')\
                                      * tan(acos(cos_phi_feedin))

        voltage_set_slack = cfg_ding0.get("mv_routing_tech_constraints",
                                          "mv_station_v_level_operation")
        bus_v_mag_set_df = pd.DataFrame(columns=['name', 'temp_id',
                                                 'v_mag_pu_set'])
        generator_pq_set_df = pd.DataFrame(columns=['name', 'temp_id',
                                                    'p_set', 'q_set'])
        load_pq_set_df = pd.DataFrame(columns=['name', 'temp_id',
                                               'p_set', 'q_set'])

    for node in nodes:
        if node not in grid.graph_isolated_nodes():
            # buses only
            if isinstance(node, CableDistributorDing0):
                buses_df = append_buses_df(buses_df, grid, node)
                # add time varying elements
                if return_time_varying_data:
                    bus_v_mag_set_df = append_bus_v_mag_set_df(bus_v_mag_set_df, node)
      
            # slack generator
            elif isinstance(node, MVStationDing0):
                # add dummy generator
                slack = pd.Series({'name': ('_'.join(['MV', str(grid.id_db), 'slack'])),
                                   'bus': node.pypsa_bus_id,
                                   'control': 'Slack',
                                   'p_nom': 0,
                                   'type': 'station',
                                   'subtype': 'mv_station'})
                generators_df = generators_df.append(slack, ignore_index=True)
                # add MV side bus
                buses_df = append_buses_df(buses_df, grid, node)
                # add time varying elements
                if return_time_varying_data:
                    slack_v_mag = pd.Series({
                        'name':node.pypsa_bus0_id,
                        'temp_id':1,
                        'v_mag_pu_set': [voltage_set_slack,  voltage_set_slack]
                    })
                    bus_v_mag_set_df = bus_v_mag_set_df.append(
                        slack_v_mag, ignore_index=True
                    )
                    slack_v_mag['name'] = node.pypsa_bus_id
                    bus_v_mag_set_df = bus_v_mag_set_df.append(
                        slack_v_mag, ignore_index=True
                    )

            # other generators
            elif isinstance(node, GeneratorDing0):
                if node.lv_grid is not None:
                    branches = node.lv_grid.graph_branches_from_node(node)
                else:
                    branches = node.mv_grid.graph_branches_from_node(node)
                # check if generator is in building generator
                neighbor = next_branch_is_helper_component(node)
                if neighbor:
                    bus_name = neighbor.pypsa_bus_id
                else:
                    bus_name = node.pypsa_bus_id
                    buses_df = append_buses_df(buses_df, grid, node)
                    if return_time_varying_data:
                        bus_v_mag_set_df = append_bus_v_mag_set_df(
                            bus_v_mag_set_df, node, node_name=bus_name
                        )
                generators_df = append_generators_df(
                    generators_df,
                    node,
                    name_bus=bus_name
                )
                # add time varying elements
                if return_time_varying_data:
                    generator_pq_set_df = append_generator_pq_set_df(
                        conf, generator_pq_set_df, node
                    )

            elif isinstance(node, LoadDing0):
                # egon data buildings do not contain consumptional data
                # consumption is not considered (at the moment)
                if consider_consumption:
                    # choose sector with highest consumption and assign sector
                    # accordingly
                    # Todo: replace when loads are seperated in a cleaner way
                    #  (retail, industrial)
                    sorted_consumption = [
                        (value, key) for key, value in node.consumption.items()
                    ]
                    sector = max(sorted_consumption)[1]
                    annual_consumption = node.consumption[sector]/1e3
                else:
                    sector, annual_consumption = None, None
                # check whether load is in building

                neighbor = next_branch_is_helper_component(node)
                if neighbor:
                    bus_name = neighbor.pypsa_bus_id
                else:
                    bus_name = node.pypsa_bus_id
                    buses_df = append_buses_df(buses_df, grid, node)
                    if return_time_varying_data:
                        bus_v_mag_set_df = append_bus_v_mag_set_df(
                            bus_v_mag_set_df, node, node_name=bus_name
                        )

                # add load
                for sector in ["residential", "cts", "industrial"]:
                    if sector == "residential":
                        number_households = node.number_households
                    else:
                        number_households = 0
                    sector_peak_load = getattr(node, f"peak_load_{sector}")
                    if sector_peak_load:
                        load = pd.Series({'name': f"{repr(node)}_{sector}",
                                          'bus': bus_name,
                                          'p_set': sector_peak_load/1e3,
                                          'number_households': int(number_households),
                                          'building_id': node.building_id,
                                          'annual_consumption': annual_consumption,
                                          'sector': sector,
                                          'type': node.type})
                        loads_df = loads_df.append(load, ignore_index=True)
                        # add time varying elements
                        if return_time_varying_data:
                            load_pq_set_df = append_load_pq_set_df(
                                conf,
                                load_pq_set_df,
                                node,
                                node_name=f"{repr(node)}_{sector}"
                            )

            # aggregated load at hv/mv substation
            elif isinstance(node, LVLoadAreaCentreDing0):
                # Todo: change to list of generators and loads?
                if node.lv_load_area.peak_load != 0:
                    node_name = 'BusBar_lac_' + str(node.lv_load_area.id_db)
                    buses_df = append_buses_df(buses_df, node.lv_load_area, 
                                               node, node_name)
                    if return_time_varying_data:
                        bus_v_mag_set_df = \
                            append_bus_v_mag_set_df(bus_v_mag_set_df, node,
                                                    node_name)
                    transformer_df = \
                        select_and_append_load_area_trafos(node.lv_load_area,
                                                           node_name,
                                                           transformer_df)
                    if return_time_varying_data:
                        loads_df, generators_df, load_pq_set_df, \
                        generator_pq_set_df = append_load_areas_to_df(
                                loads_df, generators_df,
                                node,
                                return_time_varying_data,
                                conf=conf,
                                load_pq_set_df=load_pq_set_df,
                                generator_pq_set_df=generator_pq_set_df,
                        )
                    else:
                        loads_df, generators_df = \
                            append_load_areas_to_df(loads_df, generators_df,
                                                    node, node_name=node_name)

            # bus + aggregate load of lv grids (at mv/ls substation)
            elif isinstance(node, LVStationDing0):
                # Todo: remove ding0_nw.grids when functions are thinned out
                if isinstance(grid, ding0_nw.grids.MVGridDing0):
                    # Aggregated load representing load in LV grid,
                    # only needed when LV_grids are not exported
                    if only_export_mv:
                        if return_time_varying_data:
                            loads_df, generators_df, load_pq_set_df, \
                            generator_pq_set_df = append_load_areas_to_df(
                                loads_df, generators_df, node,
                                return_time_varying_data, conf=conf,
                                load_pq_set_df=load_pq_set_df,
                                generator_pq_set_df=generator_pq_set_df)
                            bus_v_mag_set_df = append_bus_v_mag_set_df(
                                bus_v_mag_set_df,node)
                        else:
                            loads_df, generators_df = \
                                append_load_areas_to_df(loads_df,
                                                        generators_df, node)
                        for trafo in node.transformers():
                            transformer_df = \
                                append_transformers_df(transformer_df, trafo)
                        # bus at secondary MV-LV transformer side
                        buses_df = append_buses_df(buses_df, node.grid, node)
                    # bus at primary MV-LV transformer side
                    buses_df = append_buses_df(buses_df, grid, node,
                                               node.pypsa_bus0_id)
                    if return_time_varying_data:
                        bus_v_mag_set_df = \
                            append_bus_v_mag_set_df(bus_v_mag_set_df, node,
                                                   node.pypsa_bus0_id)
                elif isinstance(grid, ding0_nw.grids.LVGridDing0):
                    # bus at secondary MV-LV transformer side
                    buses_df = append_buses_df(buses_df, grid, node)
                    if return_time_varying_data:
                        bus_v_mag_set_df = \
                            append_bus_v_mag_set_df(bus_v_mag_set_df,node)
                else: 
                    raise TypeError('Something went wrong. '
                                    'Only LVGridDing0 or MVGridDing0 can '
                                    'be handled as grid.')
            elif isinstance(node, CircuitBreakerDing0):
                # TODO: remove this elif-case if CircuitBreaker are
                #  removed from graph
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


def select_and_append_load_area_trafos(aggregated_load_area, node_name,
                                       transformer_df):
    """
    Selects the right trafos for aggregrated load areas and appends them to
    the transformer dataframe.

    Parameters
    ----------
    aggregated_load_area: LVLoadAreaDing0
        Aggregated load area to be appended
    node_name: str
        Name of LV side bus for appending LV load area
    transformer_df: :pandas:`pandas.DataFrame<dataframe>`
        Transformer dataframe of network

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Transformer dataframe of network with appended transformers
    """
    if aggregated_load_area.peak_generation > \
            aggregated_load_area.peak_load:
        s_max = {'s_max': aggregated_load_area.peak_generation,
                 'case': 'gen'}
    else:
        s_max = {'s_max': aggregated_load_area.peak_load,
                 'case': 'load'}
    trafo_type, num_trafo = \
        select_transformers(aggregated_load_area, s_max)
    for t in range(0, num_trafo):
        lv_transformer = TransformerDing0(
            grid=aggregated_load_area,
            id_db=t + 1,
            v_level=aggregated_load_area._lv_grid_districts[0].
                lv_grid.v_level,
            s_max_longterm=trafo_type['S_nom'],
            r_pu=trafo_type['r_pu'],
            x_pu=trafo_type['x_pu'])
        mv_station_bus = \
            aggregated_load_area.mv_grid_district.\
                mv_grid.station().pypsa_bus_id
        transformer_df = append_transformers_df(
            transformer_df, lv_transformer, bus0=mv_station_bus, bus1=node_name)
    return transformer_df


def append_generator_pq_set_df(conf, generator_pq_set_df, node):
    """
    Fills generator pq_set data needed for power flow calculation

    Parameters
    ----------
    conf: :obj:`dict`
        dictionary with technical constants
    generator_pq_set_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name, temp_id, p_set and q_set
    node: obj:node object of generator

    Returns
    -------
    generator_pq_set_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name, temp_id, p_set and q_set
    """
    # active and reactive power of generator in load and generation case
    p_set = [node.capacity * node.capacity_factor * conf['kw2mw'] *
             conf['generation_in_load_case'],
             node.capacity * node.capacity_factor * conf['kw2mw']]
    q_set = [node.capacity * node.capacity_factor * conf['kw2mw'] *
             conf['Q_factor_generation'] * conf['generation_in_load_case'],
             node.capacity * node.capacity_factor * conf['kw2mw'] *
             conf['Q_factor_generation']]
    generator_pq_set_df = generator_pq_set_df.append(
        pd.Series({'name': repr(node), 'temp_id': 1, 'p_set': p_set,
                   'q_set': q_set}), ignore_index=True)
    return generator_pq_set_df


def append_load_pq_set_df(conf, load_pq_set_df, node, node_name = None,
                          peak_load = None):
    """
    Fills load pq_set data needed for power flow calculation

    Parameters
    ----------
    conf: :obj:`dict`
        dictionary with technical constants
    load_pq_set_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, temp_id, p_set and q_set
    node: obj:node object of generator
    node_name: :obj:`str`
        Optional parameter for name of load
    p_set: :obj:`float`
        Optional parameter for peak_load

    Returns
    -------
    load_pq_set_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, temp_id, p_set and q_set
    """
    if node_name is None:
        node_name = repr(node)
    if peak_load is None:
        peak_load = node.peak_load
    # active and reactive power of load in load and generation case
    p_set = [peak_load * conf['kw2mw'],
             peak_load * conf['kw2mw'] * conf['load_in_generation_case']]
    q_set = [peak_load * conf['kw2mw'] * conf['Q_factor_load'],
             peak_load * conf['kw2mw'] * conf['Q_factor_load'] *
             conf['load_in_generation_case']]
    load_pq_set_df = load_pq_set_df.append(
        pd.Series({'name': node_name, 'temp_id': 1,
                   'p_set': p_set, 'q_set': q_set}), ignore_index=True)
    return load_pq_set_df


def append_bus_v_mag_set_df(bus_v_mag_set_df, node, node_name = None):
    """
    Fills bus v_mag_set data needed for power flow calculation

    Parameters
    ----------
    bus_v_mag_set_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name, temp_id, v_mag_pu_set
    node: obj:node object of generator
    node_name: :obj:`str`
        Optional parameter for name of bus

    Returns
    -------
    bus_v_mag_set_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name, temp_id, v_mag_pu_set
    """
    if node_name is None:
        node_name = node.pypsa_bus_id

    # due to lv loads may have same connection bus, just append once
    if node_name not in bus_v_mag_set_df['name'].tolist():
        bus_v_mag_set_df = bus_v_mag_set_df.append(pd.Series({'name': node_name,
                                                              'temp_id': 1,
                                                              'v_mag_pu_set':
                                                              [1, 1]}),
                                                   ignore_index=True)
    return bus_v_mag_set_df


def append_load_areas_to_df(loads_df, generators_df, node,
                            return_time_varying_data=False, **kwargs):
    """
    Appends lv load area (or single lv grid district) to dataframe of loads
    and generators. Also returns power flow time varying data if
    return_time_varying_data is True.
    Each sector (agricultural, industrial, residential, retail)
    is represented by own entry of load. Each generator in underlying
    grid districts is added as own entry.
    Generators and load are connected to BusBar of the respective grid
    (LVGridDing0 for LVStationDing0 and MVGridDing0 for LVLoadAreaCentreDing0)

    Parameters
    ----------
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, bus, p_set,
        annual_consumption, sector
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name, bus, control, p_nom, type,
        weather_cell_id, subtype
    node: :obj: ding0 grid components object
        Node, which is either LVStationDing0 or LVLoadAreaCentreDing0
    return_time_varying_data: :obj:`bool`
        Determines whether data for power flow calculation is exported as well
    kwargs: list of conf, load_pq_set_df, generator_pq_set_df
        All three arguments have to be inserted if return_time_varying_data is
        True.

    Returns
    -------
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, bus, p_set,
        annual_consumption, sector
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name, bus, control, p_nom, type,
        weather_cell_id, subtype
    load_pq_set_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, temp_id, p_set and q_set,
        only exported if return_time_varying_data is True
    generator_pq_set_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name, temp_id, p_set and q_set,
        only exported if return_time_varying_data is True
    """
    # set name of bus, name of load and handles load area and grid districts
    if isinstance(node,LVStationDing0):
        name_bus = node.pypsa_bus_id
        name_load = '_'.join(['Load','mvgd' +
                              str(node.grid.grid_district.lv_load_area.
                                  mv_grid_district.id_db),
                              'lvgd' + str(node.id_db)])
        load_area = node.grid.grid_district
        grid_districts = [load_area]
    elif isinstance(node, LVLoadAreaCentreDing0):
        name_bus = kwargs.get('node_name')
        name_load = '_'.join(['Load', 'mvgd',  str(node.grid.id_db),
                              'lac',  str(node.id_db)])
        load_area = node.lv_load_area
        grid_districts = load_area.lv_grid_districts()
    else:
        raise TypeError("Only LVStationDing0 or LVLoadAreaCentreDing0 can be "
                        "inserted into function append_load_areas_to_df.")

    # unpack time varying elements
    if return_time_varying_data:
        conf = kwargs.get('conf', None)
        load_pq_set_df = kwargs.get('load_pq_set_df',None)
        generator_pq_set_df = kwargs.get('generator_pq_set_df', None)

    # Handling of generators
    for lvgd in grid_districts:
        for gen in lvgd.lv_grid.generators():
            generators_df = append_generators_df(generators_df, gen,
                                                 name_bus=name_bus)
            # add time varying elements
            if return_time_varying_data:
                generator_pq_set_df = \
                    append_generator_pq_set_df(conf, generator_pq_set_df, gen)


    # Handling of loads
    # egon data buildings do not contain consumptional data
    # sector specific consumption is not considered (at the moment)
    if consider_consumption:
        sectors = ['agricultural', 'industrial', 'residential', 'retail']
    else:
        sectors = [None]

    for sector in sectors:
        if sector:
            if (getattr(load_area, '_'.join(['peak_load', sector]))!= 0):
                if return_time_varying_data:
                    loads_df, load_pq_set_df = \
                        append_load_area_to_load_df(sector, load_area, loads_df,
                                                    name_bus, name_load,
                                                    return_time_varying_data,
                                                    conf=conf,
                                                    load_pq_set_df=load_pq_set_df)
                else:
                    loads_df = \
                        append_load_area_to_load_df(sector, load_area,
                                                    loads_df, name_bus, name_load)
        # return data without sector specific resolution (egon data buildings)
        else:
            if return_time_varying_data:
                loads_df, load_pq_set_df = \
                    append_load_area_to_load_df(sector, load_area, loads_df,
                                                name_bus, name_load,
                                                return_time_varying_data,
                                                conf=conf,
                                                load_pq_set_df=load_pq_set_df)
            else:
                loads_df = \
                    append_load_area_to_load_df(sector, load_area,
                                                loads_df, name_bus, name_load)

    if return_time_varying_data:
        return loads_df, generators_df, load_pq_set_df, generator_pq_set_df
    else:
        return loads_df, generators_df


def append_load_area_to_load_df(sector, load_area, loads_df, name_bus,
                                name_load, return_time_varying_data = False,
                                **kwargs):
    """
    Appends LVLoadArea or LVGridDistrict to dataframe of loads in pypsa format.

    Parameters
    ----------
    sector: str
        load sector: 'agricultural', 'industrial', 'residential' or 'retail'
    load_area: :obj: ding0 region
        LVGridDistrictDing0 or LVLoadAreaDing0, load area of which load is to
        be aggregated and added
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, bus, p_set,
        annual_consumption and sector
    name_bus: :obj:`str`
        name of bus to which load is connected
    name_load: :obj:`str`
        name of load
    return_time_varying_data: :obj:`bool`
        Determines whether data for power flow calculation is exported as well
    kwargs: list of conf, load_pq_set_df
        Both arguments have to be inserted if return_time_varying_data is
        True.

    Returns
    -------
    loads_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, bus, p_set,
        annual_consumption and sector
    load_pq_set_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of loads with entries name, temp_id, p_set and q_set,
        only exported if return_time_varying_data is True
    """
    # if egon data buildings used, no sector-specific consumptional data are available
    # consumption is set to None
    if not sector:

        consumption = None
        peak_load = load_area.peak_load
        load = pd.Series(
            {'name': name_load, 'bus': name_bus,
             'p_set': peak_load / 1e3,
             'annual_consumption': consumption, 'sector': sector})
        loads_df = loads_df.append(load, ignore_index=True)
        # handle time varying data
        if return_time_varying_data:
            conf = kwargs.get('conf', None)
            load_pq_set_df = kwargs.get('load_pq_set_df', None)
            load_pq_set_df = append_load_pq_set_df(conf, load_pq_set_df,
                                                   None, name_load, peak_load)
            return loads_df, load_pq_set_df
        else:
            return loads_df

    else:
        # get annual consumption
        if isinstance(load_area, LVGridDistrictDing0):
            consumption = getattr(load_area, '_'.join(['sector_consumption',
                                                       sector]))
        elif isinstance(load_area,LVLoadAreaDing0):
            consumption = 0
            for lv_grid_district in load_area.lv_grid_districts():
                consumption += getattr(lv_grid_district,
                                       '_'.join(['sector_consumption', sector]))
        # create and append load to df
        name_load = '_'.join([name_load, sector])
        peak_load = getattr(load_area, '_'.join(['peak_load', sector])) # TODO change sctor specific load allocation
        load = pd.Series(
            {'name': name_load, 'bus': name_bus,
             'p_set': peak_load/1e3,
             'annual_consumption': consumption/1e3, 'sector': sector})
        loads_df = loads_df.append(load, ignore_index=True)
        # handle time varying data
        if return_time_varying_data:
            conf = kwargs.get('conf', None)
            load_pq_set_df = kwargs.get('load_pq_set_df', None)
            load_pq_set_df = append_load_pq_set_df(conf, load_pq_set_df,
                                                   None, name_load, peak_load)
            return loads_df, load_pq_set_df
        else:
            return loads_df


def append_generators_df(generators_df, node, name_bus = None):
    """
    Appends generator to dataframe of generators in pypsa format.

    Parameters
    ----------
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name, bus, control, p_nom, type,
        weather_cell_id, subtype
    node: :obj: ding0 grid components object
        GeneratorDing0
    name_bus: :obj:`str`
        Optional parameter for name of bus

    Returns
    -------
    generators_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of generators with entries name, bus, control, p_nom, type,
        weather_cell_id, subtype
    """
    if isinstance(node,GeneratorFluctuatingDing0):
        weather_cell_id = node.weather_cell_id
    else:
        weather_cell_id = np.NaN
    if name_bus is None:
        name_bus = node.pypsa_bus_id
    generator = pd.Series(
        {
            'name': repr(node),
            'bus': name_bus,
            'control': 'PQ',
            'p_nom': (node.capacity * node.capacity_factor)/1e3,
            'type': node.type,
            'subtype': node.subtype,
            'weather_cell_id': weather_cell_id,
            "gens_id": node.gens_id
        }
    )
    generators_df = generators_df.append(generator, ignore_index=True)
    return generators_df


def append_buses_df(buses_df, grid, node, node_name =''):
    """
    Appends buses to dataframe of buses in pypsa format.

    Parameters
    ----------
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name, v_nom, geom, mv_grid_id,
        lv_grid_id, in_building
    grid: :class:`~.ding0.core.network.GridDing0`
    node: :obj: ding0 grid components object
    node_name: :obj:`str`
        name of node, per default is set to node.pypsa_bus_id

    Returns
    -------
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name, v_nom, geom, mv_grid_id,
        lv_grid_id, in_building
    """
    # set default name of node
    if node_name == '':
        node_name = node.pypsa_bus_id
    # check if node is in building
    if isinstance(node, LVCableDistributorDing0):
        in_building = node.in_building
    else:
        in_building = False
    # set geodata, if existing
    [x,y] = [np.NaN, np.NaN]
    if isinstance(node.geo_data,Point):
        [x,y] = [node.geo_data.x, node.geo_data.y]
    #set grid_ids
    if isinstance(grid, ding0_nw.grids.MVGridDing0):
        mv_grid_id = grid.id_db
        lv_grid_id = np.NaN
        v_nom = grid.v_level
    elif isinstance(grid, ding0_nw.grids.LVGridDing0):
        mv_grid_id = \
            grid.grid_district.lv_load_area.mv_grid_district.mv_grid.id_db
        lv_grid_id = grid.id_db
        v_nom = grid.v_level/1e3
    elif isinstance(grid, LVLoadAreaDing0):
        mv_grid_id = \
            grid.mv_grid_district.mv_grid.id_db
        lv_grid_id = grid.id_db
        try:
            v_nom = grid._lv_grid_districts[0].lv_grid.v_level/1e3
        except:
            v_nom = 0.4
            logger.warning("V_nom of aggregated LVLoadArea{} in MVGrid {} "
                           "could not be set. Will be set to 0.4 as a default. "
                           "Please check.".format(lv_grid_id, mv_grid_id))
    else:
        raise TypeError('Something went wrong, only MVGridDing0 and '
                        'LVGridDing0 should be inserted as grid.')
    # create bus dataframe
    bus = pd.Series({'name': node_name,'v_nom':v_nom, 'x':x, 'y':y,
                     'mv_grid_id':mv_grid_id,'lv_grid_id':lv_grid_id,
                     'in_building': in_building})
    buses_df = buses_df.append(bus, ignore_index=True)
    return buses_df


def append_transformers_df(transformers_df, trafo, type = np.NaN,
                           bus0=None, bus1=None):
    """
    Appends transformer to dataframe of buses in pypsa format.

    Parameters
    ----------
    transformers_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of trafos with entries name, bus0, bus1, x, r, s_nom, type
    trafo: :obj:TransformerDing0
        Transformer to be added
    type: :obj:`str`
        Optional parameter for type of transformer
    bus0: :obj:`str`
        Name of primary side bus. Defaults to None and is set to primary side
        of transformer station by default.
    bus1: :obj:`str`
        Name of secondary side bus. Defaults to None and is set to secondary
        side of transformer station by default.

    Returns
    -------
    transformers_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of trafos with entries name, bus0, bus1, x, r, s_nom, type
    """
    if isinstance(type, str):
        type_info = type
    else:
        if isinstance(trafo.grid, ding0_nw.grids.MVGridDing0):
            voltage_upper = 110
            voltage_lower = trafo.grid.v_level
        elif isinstance(trafo.grid, ding0_nw.grids.LVGridDing0):
            voltage_upper = trafo.grid.grid_district.lv_load_area.mv_grid_district.mv_grid.v_level
            voltage_lower = trafo.grid.v_level/1e3
        elif isinstance(trafo.grid, LVLoadAreaDing0):
            voltage_upper = trafo.grid.mv_grid_district.mv_grid.v_level
            voltage_lower = trafo.v_level/1e3
        type_info = '{} MVA {}/{} kV'.format(trafo.s_max_a/1e3,
                                             voltage_upper, voltage_lower)

    if bus0 is None:
        bus0 = trafo.grid.station().pypsa_bus0_id
    if bus1 is None:
        bus1 = trafo.grid.station().pypsa_bus_id

    trafo_tmp = pd.Series({'name': repr(trafo),
                           'bus0': bus0,
                           'bus1': bus1,
                           'x': trafo.x_pu, 'r': trafo.r_pu,
                           's_nom': trafo.s_max_a/1e3,
                           'type': type,
                           'type_info': type_info})
    transformers_df = transformers_df.append(trafo_tmp, ignore_index=True)
    return transformers_df


def edges_to_dict_of_dataframes(edges, lines_df, buses_df):
    """
    Export edges to DataFrame

    Parameters
    ----------
    edges: :obj:`list`
        Edges of Ding0.Network graph
    lines_df: :pandas:`pandas.DataFrame<dataframe>`
            Dataframe of lines with entries name, bus0, bus1, length, x, r,
            s_nom, num_parallel, type
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name, v_nom, geom, mv_grid_id,
        lv_grid_id, in_building

    Returns
    -------
    edges_dict: dict
    """

    # iterate over edges and add them one by one
    for edge in edges:
        if not edge['branch'].connects_aggregated: #TODO
            lines_df = append_lines_df(edge, lines_df, buses_df)

    return {'Line': lines_df.set_index('name')}


def append_lines_df(edge, lines_df, buses_df):
    """
    Append edge to lines_df

    Parameters
    ----------
    edge:
        Edge of Ding0.Network graph
    lines_df: :pandas:`pandas.DataFrame<dataframe>`
            Dataframe of lines with entries name, bus0, bus1, length, x, r,
            s_nom, num_parallel, type, geometry
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name, v_nom, geom, mv_grid_id,
        lv_grid_id, in_building

    Returns
    -------
    lines_df: :pandas:`pandas.DataFrame<dataframe>`
            Dataframe of lines with entries name, bus0, bus1, length, x, r,
            s_nom, num_parallel, type
    """
    if edge["branch"].helper_component:
        return lines_df

    freq = cfg_ding0.get('assumptions', 'frequency')
    omega = 2 * pi * freq
    # set grid_ids
    if isinstance(edge['branch'].grid, ding0_nw.grids.MVGridDing0):
        unitconversion = 1e3
        is_mv = True
    elif isinstance(edge['branch'].grid, ding0_nw.grids.LVGridDing0):
        unitconversion = 1e6
        is_mv = False
    else:
        raise TypeError('Something went wrong, only MVGridDing0 and '
                        'LVGridDing0 should be inserted as grid.')
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
                edge['branch'].type['U_n'].values[0]/unitconversion
    else:
        s_nom = sqrt(3) * edge['branch'].type['I_max_th'] * \
                edge['branch'].type['U_n']/unitconversion
    # get lengths of line
    length = edge['branch'].length / 1e3
    # Add minimal length of 1m
    if length < 0.001:
        length = 0.001
    #Todo: change into same format
    if 'name' in edge['branch'].type:
        type = edge['branch'].type['name']
    else:
        type = edge['branch'].type.name
    
    # make sure right side of station is appended
    if isinstance(edge['adj_nodes'][0], LVStationDing0) and is_mv:
        name_bus0 = edge['adj_nodes'][0].pypsa_bus0_id
    else:
        name_bus0 = edge['adj_nodes'][0].pypsa_bus_id
    if isinstance(edge['adj_nodes'][1], LVStationDing0) and is_mv:
        name_bus1 = edge['adj_nodes'][1].pypsa_bus0_id
    else:
        name_bus1 = edge['adj_nodes'][1].pypsa_bus_id

    line = pd.Series({'name': repr(edge['branch']),
                      'bus0': name_bus0,
                      'bus1': name_bus1,
                      'x': x_per_km * length, 'r':r_per_km * length,
                      's_nom': s_nom, 'length': length,
                      'num_parallel': edge['branch'].num_parallel,
                      'kind': edge['branch'].kind,
                      'type_info': type,
                      'geometry': edge['branch'].geometry})
    lines_df = lines_df.append(line, ignore_index=True)

    return lines_df


def _check_branch_for_in_building_buses(buses_df, edge, name_bus0, name_bus1):
    """
    Checks if line is to be added to line_df. If line is artificial line that
    connects in building loads or generators to preceding branch tee, False is
    returned.

    Parameters
    ----------
    buses_df: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe of buses with entries name, v_nom, geom, mv_grid_id,
        lv_grid_id, in_building
    edge:
        Edge of Ding0.Network graph
    name_bus0: str
    name_bus1: str

    Returns
    -------
    bool
        Indicator whether line should be added to lines_df or not
    """
    # check if neighboring buses are in buses_df
    if name_bus0 not in buses_df.index:
        if (isinstance(edge['adj_nodes'][0], GeneratorDing0)
            or isinstance(edge['adj_nodes'][0], LoadDing0)) and \
                buses_df.loc[name_bus1, 'in_building'] and \
                edge['branch'].length == 1:
            return False
        else:
            raise ValueError('Bus0 of line {} not in buses_df.'.format(
                repr(edge['branch'])))
    if name_bus1 not in buses_df.index:
        if (isinstance(edge['adj_nodes'][1], GeneratorDing0)
            or isinstance(edge['adj_nodes'][1], LoadDing0)) and \
                buses_df.loc[name_bus0, 'in_building'] and \
                edge['branch'].length == 1:
            return False
        else:
            raise ValueError('Bus1 of line {} not in buses_df.'.format(
                repr(edge['branch'])))
    return True


def circuit_breakers_to_df(grid, components, component_data,
                           open_circuit_breakers,
                           return_time_varying_data=False):
    """
    Appends circuit breakers to component dicts. If circuit breakers are open
    a virtual bus is added to the respective dataframe and bus1 of the line 
    attached to the circuit breaker is set to the new virtual node. 

    Parameters
    ----------
    grid: :class:`~.ding0.core.network.GridDing0`
    components: components: :obj:`dict`
        Dictionary of component Dataframes 'Bus', 'Generator', 'Line', 'Load',
        'Transformer'   
    component_data: :obj:`dict`
        Dictionary of component Dataframes 'Bus', 'Generator', 'Load',
        needed for power flow calculations
    open_circuit_breakers: :obj:`dict`
        Dictionary containing names of open circuit breakers
    return_time_varying_data: :obj:`bool`
        States whether time varying data needed for power flow calculations
        are constructed as well. Set to True to run power flow, set to False
        to export network to csv.

    Returns
    -------
    components: :obj:`dict`
        Dictionary of component Dataframes 'Bus', 'Generator', 'Line', 'Load',
        'Transformer', 'Switch'
    component_data: :obj:`dict`
        Dictionary of component Dataframes 'Bus', 'Generator', 'Load',
        needed for power flow calculations    
    """
    if hasattr(grid, '_circuit_breakers'):
        # initialise dataframe for circuit breakers
        circuit_breakers_df = pd.DataFrame(columns=['name', 'bus_closed', 
                                                    'bus_open','branch', 
                                                    'type_info'])
        for circuit_breaker in grid.circuit_breakers():
            if circuit_breaker.switch_node is not None:
                if isinstance(circuit_breaker.switch_node, LVStationDing0):
                    name_bus_closed = circuit_breaker.switch_node.pypsa_bus0_id
                else:
                    name_bus_closed = circuit_breaker.switch_node.pypsa_bus_id
            else:
                # get secondary bus of opened branch
                name_bus_closed = \
                    components['Line'].T[repr(circuit_breaker.branch)].bus1
            # create virtual bus and append to components['Bus']
            name_bus_open = 'virtual_' + name_bus_closed
            # if circuit breaker was open, change bus1 of branch to new 
            # virtual node
            if repr(circuit_breaker) in open_circuit_breakers:
                if components['Line'].at[repr(circuit_breaker.branch),'bus1'] \
                        == name_bus_closed:
                    components['Line'].at[
                        repr(circuit_breaker.branch), 'bus1'] = \
                        name_bus_open
                elif components['Line'].at[repr(circuit_breaker.branch),'bus0'] \
                        == name_bus_closed:
                    components['Line'].at[
                        repr(circuit_breaker.branch), 'bus0'] = \
                        name_bus_open
                else:
                    raise Exception('Branch connected to circuit breaker {}'
                                    ' is not connected to node {}'. format(
                        repr(circuit_breaker), name_bus_closed
                    ))
                
                bus_open = components['Bus'].T[name_bus_closed]
                bus_open.name = name_bus_open
                components['Bus'] = components['Bus'].append(bus_open)
                if return_time_varying_data:
                    component_data['Bus'] = \
                        component_data['Bus'].append(
                            pd.DataFrame({'name': [name_bus_open], 
                                          'temp_id': [1],
                                          'v_mag_pu_set': [[1, 1]]}).
                            set_index('name'))
            # append circuit breaker to dataframe
            circuit_breakers_df = \
                circuit_breakers_df.append(
                    pd.Series({'name': repr(circuit_breaker), 
                               'bus_closed': name_bus_closed,
                               'bus_open': name_bus_open,
                               'branch': repr(circuit_breaker.branch), 
                               'type_info': 'Switch Disconnector'}),
                    ignore_index=True)
        # add switches to components
        components['Switch'] = circuit_breakers_df.set_index('name')
    return components, component_data


def run_powerflow_onthefly(components, components_data, grid, 
                           export_pypsa_dir=None, debug=False, 
                           export_result_dir=None):
    """
    Run powerflow to test grid stability

    Two cases are defined to be tested here:
     i) load case
     ii) feed-in case

    Parameters
    ----------
    components: dict of :pandas:`pandas.DataFrame<dataframe>`
    components_data: dict of :pandas:`pandas.DataFrame<dataframe>`
    grid: :class:`~.ding0.core.network.GridDing0`
    export_pypsa_dir: :obj:`str`
        Sub-directory in output/debug/grid/ where csv Files of PyPSA network 
        are exported to. Export is omitted if argument is empty.
    debug: :obj:`bool`
    export_result_dir: :obj:`str`
        Directory where csv Files of power flow results are exported to.
        Export is omitted if argument is empty.
    """

    # choose temp_id
    timesteps = 2
    start_time = datetime(1970, 1, 1, 00, 00, 0)
    resolution = 'H'

    # inspect grid data for integrity
    if debug:
        data_integrity(components, components_data)

    # define investigated time range
    timerange = date_range(freq=resolution,
                           periods=timesteps,
                           start=start_time)

    # TODO: Instead of hard coding PF config, values from class PFConfigDing0 
    #  can be used here.
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

    # check if network is created in a correct way
    _check_integrity_of_pypsa(network)

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

    for comp in ['Bus', 'Load', 'Generator']:  # list(components_data.keys()):
        data_check[comp] = {}
        if comp == "Bus":
            # HV Bus is not in components but in components_data
            offset = 1
        elif comp == "Generator":
            # Slack Generator is in components but not in components_data
            offset = -1
        else:
            offset = 0
        data_check[comp]['length_diff'] = (
                len(components[comp]) - len(components_data[comp]) + offset
        )

    # print short report to user and exit program if not integer
    for comp in list(data_check.keys()):
        if data_check[comp]['length_diff'] != 0:
            logger.error("{comp} data is invalid. You supplied {no_comp} "
                "{comp} objects and {no_data} datasets. Check you grid data "
                "and try again".format(comp=comp,
                                     no_comp=len(components[comp]),
                                     no_data=len(components_data[comp])))


def _check_integrity_of_pypsa(pypsa_network):
    """
    Checks if pypsa network fulfills certain requirements

    Parameters
    ----------
    pypsa_network: PyPSA powerflow problem object
    """

    # check for sub-networks
    subgraphs = list(pypsa_network.graph().subgraph(c) for c in
                     connected_components(pypsa_network.graph()))
    pypsa_network.determine_network_topology()

    if len(subgraphs) > 1 or len(pypsa_network.sub_networks) > 1:
        raise ValueError("The graph has isolated nodes or edges")

    # check consistency of topology and time series data
    generators_ts_p_missing = pypsa_network.generators.loc[
        ~pypsa_network.generators.index.isin(
            pypsa_network.generators_t['p_set'].columns.tolist())]
    generators_ts_q_missing = pypsa_network.generators.loc[
        ~pypsa_network.generators.index.isin(
            pypsa_network.generators_t['q_set'].columns.tolist())]
    loads_ts_p_missing = pypsa_network.loads.loc[
        ~pypsa_network.loads.index.isin(
            pypsa_network.loads_t['p_set'].columns.tolist())]
    loads_ts_q_missing = pypsa_network.loads.loc[
        ~pypsa_network.loads.index.isin(
            pypsa_network.loads_t['q_set'].columns.tolist())]
    bus_v_set_missing = pypsa_network.buses.loc[
        ~pypsa_network.buses.index.isin(
            pypsa_network.buses_t['v_mag_pu_set'].columns.tolist())]

    # Comparison of generators excludes slack generators (have no time series)
    if (not generators_ts_p_missing.empty and not all(
            generators_ts_p_missing['control'] == 'Slack')):
        raise ValueError("Following generators have no `p_set` time series "
                         "{generators}".format(
            generators=generators_ts_p_missing))

    if (not generators_ts_q_missing.empty and not all(
            generators_ts_q_missing['control'] == 'Slack')):
        raise ValueError("Following generators have no `q_set` time series "
                         "{generators}".format(
            generators=generators_ts_q_missing))

    if not loads_ts_p_missing.empty:
        raise ValueError("Following loads have no `p_set` time series "
                         "{loads}".format(
            loads=loads_ts_p_missing))

    if not loads_ts_q_missing.empty:
        raise ValueError("Following loads have no `q_set` time series "
                         "{loads}".format(
            loads=loads_ts_q_missing))

    if not bus_v_set_missing.empty:
        raise ValueError("Following loads have no `v_mag_pu_set` time series "
                         "{buses}".format(
            buses=bus_v_set_missing))

    # check for duplicate labels (of components)
    duplicated_labels = []
    if any(pypsa_network.buses.index.duplicated()):
        duplicated_labels.append(pypsa_network.buses.index[
                                 pypsa_network.buses.index.duplicated()])
    if any(pypsa_network.generators.index.duplicated()):
        duplicated_labels.append(pypsa_network.generators.index[
                                 pypsa_network.generators.index.duplicated()])
    if any(pypsa_network.loads.index.duplicated()):
        duplicated_labels.append(pypsa_network.loads.index[
                                 pypsa_network.loads.index.duplicated()])
    if any(pypsa_network.transformers.index.duplicated()):
        duplicated_labels.append(pypsa_network.transformers.index[
                                 pypsa_network.transformers.index.duplicated()])
    if any(pypsa_network.lines.index.duplicated()):
        duplicated_labels.append(pypsa_network.lines.index[
                                 pypsa_network.lines.index.duplicated()])
    if duplicated_labels:
        raise ValueError("{labels} have duplicate entry in "
                         "one of the components dataframes".format(
            labels=duplicated_labels))

    # duplicate p_sets and q_set
    duplicate_p_sets = []
    duplicate_q_sets = []
    if any(pypsa_network.loads_t['p_set'].columns.duplicated()):
        duplicate_p_sets.append(pypsa_network.loads_t['p_set'].columns[
                                    pypsa_network.loads_t[
                                        'p_set'].columns.duplicated()])
    if any(pypsa_network.loads_t['q_set'].columns.duplicated()):
        duplicate_q_sets.append(pypsa_network.loads_t['q_set'].columns[
                                    pypsa_network.loads_t[
                                        'q_set'].columns.duplicated()])

    if any(pypsa_network.generators_t['p_set'].columns.duplicated()):
        duplicate_p_sets.append(pypsa_network.generators_t['p_set'].columns[
                                    pypsa_network.generators_t[
                                        'p_set'].columns.duplicated()])
    if any(pypsa_network.generators_t['q_set'].columns.duplicated()):
        duplicate_q_sets.append(pypsa_network.generators_t['q_set'].columns[
                                    pypsa_network.generators_t[
                                        'q_set'].columns.duplicated()])

    if duplicate_p_sets:
        raise ValueError("{labels} have duplicate entry in "
                         "generators_t['p_set']"
                         " or loads_t['p_set']".format(
            labels=duplicate_p_sets))
    if duplicate_q_sets:
        raise ValueError("{labels} have duplicate entry in "
                         "generators_t['q_set']"
                         " or loads_t['q_set']".format(
            labels=duplicate_q_sets))

    # find duplicate v_mag_set entries
    duplicate_v_mag_set = []
    if any(pypsa_network.buses_t['v_mag_pu_set'].columns.duplicated()):
        duplicate_v_mag_set.append(pypsa_network.buses_t['v_mag_pu_set'].
                                   columns[pypsa_network.buses_t[
            'v_mag_pu_set'].columns.duplicated()])

    if duplicate_v_mag_set:
        raise ValueError("{labels} have duplicate entry in buses_t".format(
            labels=duplicate_v_mag_set))

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
    grid: :class:`~.ding0.core.network.GridDing0`
    bus_data: :pandas:`pandas.DataFrame<dataframe>`
        DataFrame containing voltage levels obtained from PF analysis
    """

    # iterate of nodes and assign voltage obtained from power flow analysis
    for node in grid.graph.nodes():
        # check if node is connected to graph
        if (node not in grid.graph_isolated_nodes()
            and not isinstance(node,
                               LVLoadAreaCentreDing0)):
            if isinstance(node, LVStationDing0):
                node.voltage_res = bus_data.loc[node.pypsa_bus_id, 'v_mag_pu']
            elif isinstance(node, (LVStationDing0, LVLoadAreaCentreDing0)):
                if node.lv_load_area.is_aggregated:
                    node.voltage_res = bus_data.loc[node.pypsa_bus_id, 
                                                    'v_mag_pu']
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
    grid: :class:`~.ding0.core.network.GridDing0`
    line_data: :pandas:`pandas.DataFrame<dataframe>`
        DataFrame containing active/reactive at nodes obtained from PF analysis
    """

    package_path = ding0.__path__[0]

    edges = [edge for edge in grid.graph_edges()
             if (edge['adj_nodes'][0] in grid.graph.nodes() and
                 not isinstance(edge['adj_nodes'][0], LVLoadAreaCentreDing0))
             and (edge['adj_nodes'][1] in grid.graph.nodes() and
                  not isinstance(edge['adj_nodes'][1], LVLoadAreaCentreDing0))]

    decimal_places = 6
    for edge in edges:
        s_res = [
            round(sqrt(
                max(abs(line_data.loc[repr(edge['branch']), 'p0'][0]),
                    abs(line_data.loc[repr(edge['branch']), 'p1'][0])) ** 2 +
                max(abs(line_data.loc[repr(edge['branch']), 'q0'][0]),
                    abs(line_data.loc[repr(edge['branch']), 'q1'][0])) ** 2),
                decimal_places),
            round(sqrt(
                max(abs(line_data.loc[repr(edge['branch']), 'p0'][1]),
                    abs(line_data.loc[repr(edge['branch']), 'p1'][1])) ** 2 +
                max(abs(line_data.loc[repr(edge['branch']), 'q0'][1]),
                    abs(line_data.loc[repr(edge['branch']), 'q1'][1])) ** 2),
                decimal_places)]

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
        if component in ['Switch', 'HVMV_Transformer']:
            continue
        network.import_components_from_dataframe(components[component],
                                                 component)

    return network, snapshots
