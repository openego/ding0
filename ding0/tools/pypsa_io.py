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
from ding0.core.network.stations import LVStationDing0, MVStationDing0
from ding0.core.network import BranchDing0, CircuitBreakerDing0, GeneratorDing0
from ding0.core import MVCableDistributorDing0
from ding0.core.structure.regions import LVLoadAreaCentreDing0

from geoalchemy2.shape import from_shape
from math import tan, acos, pi, sqrt
from shapely.geometry import LineString
from pandas import Series, DataFrame, DatetimeIndex
from pypsa.io import import_series_from_dataframe
from pypsa import Network

from datetime import datetime
import sys
import os
import re
import logging


logger = logging.getLogger('ding0')


def export_to_dir(network, export_dir):
    """
    Exports PyPSA network as CSV files to directory

    Args:
        network: pypsa.Network
        export_dir: str
            Sub-directory in output/debug/grid/ where csv Files of PyPSA network are exported to.
    """

    package_path = ding0.__path__[0]

    network.export_to_csv_folder(os.path.join(package_path,
                                              'output',
                                              'debug',
                                              'grid',
                                              export_dir))


def q_sign(power_factor_mode_string, sign_convention):
    """
    Gets the correct sign for Q time series given 'inductive' and 'capacitive' and the 'generator'
    or 'load' convention.

    Parameters
    ----------
    power_factor_mode_string: :obj:`str`
        Either 'inductive' or 'capacitive'
    sign_convention: :obj:`str`
        Either 'load' or 'generator'
    Return
    ------
    :obj: `int` : +1 or -1
        A sign to mulitply to Q time sereis
    """

    comparestr = power_factor_mode_string.lower()

    if re.fullmatch('inductive', comparestr):
        if re.fullmatch('generator', sign_convention):
            return -1
        elif re.fullmatch('load', sign_convention):
            return 1
        else:
            raise ValueError("Unknown sign conention {}".format(sign_convention))
    elif re.fullmatch('capacitive', comparestr):
        if re.fullmatch('generator', sign_convention):
            return 1
        elif re.fullmatch('load', sign_convention):
            return -1
        else:
            raise ValueError("Unknown sign conention {}".format(sign_convention))
    else:
        raise ValueError("Unknown value {} in power_factor_mode".format(power_factor_mode_string))


def nodes_to_dict_of_dataframes(grid, nodes, lv_transformer=True):
    """
    Creates dictionary of dataframes containing grid

    Parameters
    ----------
    grid: ding0.Network
    nodes: list of ding0 grid components objects
        Nodes of the grid graph
    lv_transformer: bool, True
        Toggle transformer representation in power flow analysis

    Returns:
    components: dict of pandas.DataFrame
        DataFrames contain components attributes. Dict is keyed by components
        type
    components_data: dict of pandas.DataFrame
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

    Q_factor_load = q_sign(cos_phi_load_mode[1:-1], 'load') * tan(acos(cos_phi_load))
    Q_factor_generation = q_sign(cos_phi_feedin_mode[1:-1], 'generator') * tan(acos(cos_phi_feedin))

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

    # # TODO: only for debugging, remove afterwards
    # import csv
    # nodeslist = sorted([node.__repr__() for node in nodes
    #                     if node not in grid.graph_isolated_nodes()])
    # with open('/home/guido/ding0_debug/nodes_via_dataframe.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter='\n')
    #     writer.writerow(nodeslist)

    for node in nodes:
        if node not in grid.graph_isolated_nodes():
            # buses only
            if isinstance(node, MVCableDistributorDing0):
                buses['bus_id'].append(node.pypsa_id)
                buses['v_nom'].append(grid.v_level)
                buses['geom'].append(from_shape(node.geo_data, srid=srid))
                buses['grid_id'].append(grid.id_db)

                bus_v_mag_set['bus_id'].append(node.pypsa_id)
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

                buses['bus_id'].append(node.pypsa_id)
                buses['v_nom'].append(grid.v_level)
                buses['geom'].append(from_shape(node.geo_data, srid=srid))
                buses['grid_id'].append(grid.id_db)

                bus_v_mag_set['bus_id'].append(node.pypsa_id)
                bus_v_mag_set['temp_id'].append(1)
                bus_v_mag_set['grid_id'].append(grid.id_db)

                generator['grid_id'].append(grid.id_db)
                generator['bus'].append(node.pypsa_id)


            # aggregated load at hv/mv substation
            elif isinstance(node, LVLoadAreaCentreDing0):
                load['load_id'].append(node.pypsa_id)
                load['bus'].append('_'.join(['HV', str(grid.id_db), 'trd']))
                load['grid_id'].append(grid.id_db)

                load_pq_set['load_id'].append(node.pypsa_id)
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
                load['bus'].append(node.pypsa_id)
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
                buses['bus_id'].append(node.pypsa_id)
                buses['v_nom'].append(grid.v_level)
                buses['geom'].append(from_shape(node.geo_data, srid=srid))
                buses['grid_id'].append(grid.id_db)

                bus_v_mag_set['bus_id'].append(node.pypsa_id)
                bus_v_mag_set['temp_id'].append(1)
                bus_v_mag_set['v_mag_pu_set'].append([1, 1])
                bus_v_mag_set['grid_id'].append(grid.id_db)

                # generator representing generation capacity in LV grid
                generator['generator_id'].append('_'.join(
                    ['MV', str(grid.id_db), 'gen', str(node.id_db)]))
                generator['control'].append('PQ')
                generator['p_nom'].append(node.peak_generation)
                generator['grid_id'].append(grid.id_db)
                generator['bus'].append(node.pypsa_id)

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


def edges_to_dict_of_dataframes(grid, edges):
    """
    Export edges to DataFrame

    Parameters
    ----------
    grid: ding0.Network
    edges: list
        Edges of Ding0.Network graph

    Returns
    -------
    edges_dict: dict
    """
    omega = 2 * pi * 50
    srid = int(cfg_ding0.get('geo', 'srid'))

    lines = {'line_id': [], 'bus0': [], 'bus1': [], 'x': [], 'r': [],
             's_nom': [], 'length': [], 'cables': [], 'geom': [],
             'grid_id': []}

    # iterate over edges and add them one by one
    for edge in edges:

        line_name = '_'.join(['MV',
                              str(grid.id_db),
                              'lin',
                              str(edge['branch'].id_db)])

        # TODO: find the real cause for being L, C, I_th_max type of Series
        if (isinstance(edge['branch'].type['L'], Series) or
                isinstance(edge['branch'].type['C'], Series)):
            x = omega * edge['branch'].type['L'].values[0] * 1e-3
        else:

            x = omega * edge['branch'].type['L'] * 1e-3

        if isinstance(edge['branch'].type['R'], Series):
            r = edge['branch'].type['R'].values[0]
        else:
            r = edge['branch'].type['R']

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
        lines['bus0'].append(edge['adj_nodes'][0].pypsa_id)
        lines['bus1'].append(edge['adj_nodes'][1].pypsa_id)
        lines['x'].append(x * l)
        lines['r'].append(r * l)
        lines['s_nom'].append(s_nom)
        lines['length'].append(l)
        lines['cables'].append(3)
        lines['geom'].append(from_shape(
            LineString([edge['adj_nodes'][0].geo_data,
                        edge['adj_nodes'][1].geo_data]),
            srid=srid))
        lines['grid_id'].append(grid.id_db)

    return {'Line': DataFrame(lines).set_index('line_id')}


def run_powerflow_onthefly(components, components_data, grid, export_pypsa_dir=None, debug=False):
    """
    Run powerflow to test grid stability

    Two cases are defined to be tested here:
     i) load case
     ii) feed-in case

    Parameters
    ----------
    components: dict of pandas.DataFrame
    components_data: dict of pandas.DataFrame
    export_pypsa_dir: str
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
    bus_data: pandas.DataFrame
        Voltage level results at buses
    line_data: pandas.DataFrame
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
    bus_data: pandas.DataFrame
        DataFrame containing voltage levels obtained from PF analysis
    """

    # iterate of nodes and assign voltage obtained from power flow analysis
    for node in grid._graph.nodes():
        # check if node is connected to graph
        if (node not in grid.graph_isolated_nodes()
            and not isinstance(node,
                               LVLoadAreaCentreDing0)):
            if isinstance(node, LVStationDing0):
                node.voltage_res = bus_data.loc[node.pypsa_id, 'v_mag_pu']
            elif isinstance(node, (LVStationDing0, LVLoadAreaCentreDing0)):
                if node.lv_load_area.is_aggregated:
                    node.voltage_res = bus_data.loc[node.pypsa_id, 'v_mag_pu']
            elif not isinstance(node, CircuitBreakerDing0):
                node.voltage_res = bus_data.loc[node.pypsa_id, 'v_mag_pu']
            else:
                logger.warning("Object {} has been skipped while importing "
                               "results!")


def assign_line_results(grid, line_data):
    """
    Write results obtained from PF to graph

    Parameters
    -----------
    grid: ding0.network
    line_data: pandas.DataFrame
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
                max(abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                    'branch'].id_db), 'p0'][0]),
                    abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                        'branch'].id_db), 'p1'][0])) ** 2 +
                max(abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                    'branch'].id_db), 'q0'][0]),
                    abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                        'branch'].id_db), 'q1'][0])) ** 2),decimal_places),
            round(sqrt(
                max(abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                    'branch'].id_db), 'p0'][1]),
                    abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                        'branch'].id_db), 'p1'][1])) ** 2 +
                max(abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                    'branch'].id_db), 'q0'][1]),
                    abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                        'branch'].id_db), 'q1'][1])) ** 2),decimal_places)]

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