from egoio.db_tables import calc_ego_mv_powerflow as orm_pypsa
from egoio.db_tables.calc_ego_mv_powerflow import ResBus, ResLine, \
    ResTransformer, Bus, Line, Transformer

from egopowerflow.tools.io import get_timerange, import_components, \
    import_pq_sets, create_powerflow_problem, results_to_oedb, \
    transform_timeseries4pypsa
from egopowerflow.tools.plot import add_coordinates, plot_line_loading

from egoio.db_tables.calc_ego_mv_powerflow import Bus, Line, Generator, Load, \
    Transformer, TempResolution, BusVMagSet, GeneratorPqSet, LoadPqSet
from egoio.db_tables.calc_ego_mv_powerflow import TempResolution

from dingo.tools import config as cfg_dingo
from dingo.core.network.stations import LVStationDingo, MVStationDingo
from dingo.core.network import BranchDingo, CircuitBreakerDingo, GeneratorDingo
from dingo.core import MVCableDistributorDingo
from dingo.core.structure.regions import LVLoadAreaCentreDingo

from geoalchemy2.shape import from_shape
from math import tan, acos, pi, sqrt
from shapely.geometry import LineString
from pandas import Series, read_sql_query, DataFrame, DatetimeIndex
from pypsa.io import import_series_from_dataframe

from datetime import datetime


def delete_powerflow_tables(session):
    """Empties all powerflow tables to start export from scratch

    Parameters
    ----------
    session: SQLAlchemy session object
    """
    tables = [orm_pypsa.Bus, orm_pypsa.BusVMagSet, orm_pypsa.Load,
              orm_pypsa.LoadPqSet, orm_pypsa.Generator,
              orm_pypsa.GeneratorPqSet, orm_pypsa.Line, orm_pypsa.Transformer,
              orm_pypsa.TempResolution]

    for table in tables:
        session.query(table).delete()
    session.commit()


def export_nodes(grid, session, nodes, temp_id, lv_transformer=True):
    """Export nodes of grid graph representation to pypsa input tables

    Parameters
    ----------
    grid: MVGridDingo
        Instance of MVGridDingo class
    session: SQLAlchemy session object
    temp_id: int
        ID of `temp_resolution` table
    lv_transformer: boolean, default True
        Toggles between representation of LV transformer

    Notes
    -----
    Reactive power of DEA is modeled with cos phi = 1 according to "DENA
    Verteilnetzstudie"


    """

    mv_routing_loads_cos_phi = float(
        cfg_dingo.get('mv_routing_tech_constraints',
                      'mv_routing_loads_cos_phi'))
    srid = int(cfg_dingo.get('geo', 'srid'))

    load_in_generation_case = cfg_dingo.get('assumptions',
                                            'load_in_generation_case')

    Q_factor_load = tan(acos(mv_routing_loads_cos_phi))

    voltage_set_slack = cfg_dingo.get("mv_routing_tech_constraints",
                                      "mv_station_v_level_operation")

    kw2mw = 1e-3

    # Create all busses
    for node in nodes:
        if node not in grid.graph_isolated_nodes():
            if isinstance(node, LVStationDingo):
                # MV side bus
                bus_mv = orm_pypsa.Bus(
                    bus_id=node.pypsa_id,
                    v_nom=grid.v_level,
                    geom=from_shape(node.geo_data, srid=srid),
                    grid_id=grid.id_db)
                bus_pq_set_mv = orm_pypsa.BusVMagSet(
                    bus_id=node.pypsa_id,
                    temp_id=temp_id,
                    v_mag_pu_set=[1, 1],
                    grid_id=grid.id_db)
                session.add(bus_mv)
                session.add(bus_pq_set_mv)
                if lv_transformer is True:
                    # Add transformer to bus
                    print("Regarding x, r and s_nom LV station {} only " \
                          "first transformer in considered in PF " \
                          "analysis".format(node))
                    # TODO: consider multiple transformers and remove above warning
                    transformer = orm_pypsa.Transformer(
                        trafo_id='_'.join(
                            ['MV', str(grid.id_db), 'trf', str(node.id_db)]),
                        s_nom=node._transformers[0].s_max_a,
                        bus0=node.pypsa_id,
                        bus1='_'.join(
                            ['MV', str(grid.id_db), 'trd', str(node.id_db)]),
                        x=node._transformers[0].x,
                        r=node._transformers[0].r,
                        tap_ratio=1,
                        geom=from_shape(node.geo_data, srid=srid),
                        grid_id=grid.id_db)
                    session.add(transformer)
                    # Add bus on transformer's LV side
                    bus_lv = orm_pypsa.Bus(
                        bus_id='_'.join(
                            ['MV', str(grid.id_db), 'trd', str(node.id_db)]),
                        v_nom=node._transformers[0].v_level,
                        geom=from_shape(node.geo_data, srid=srid),
                        grid_id=grid.id_db)
                    bus_pq_set_lv = orm_pypsa.BusVMagSet(
                        bus_id='_'.join(
                            ['MV', str(grid.id_db), 'trd', str(node.id_db)]),
                        temp_id=temp_id,
                        v_mag_pu_set=[1, 1],
                        grid_id=grid.id_db)
                    session.add(bus_lv)
                    session.add(bus_pq_set_lv)
                    # Add aggregated LV load to LV bus
                    load = orm_pypsa.Load(
                        load_id='_'.join(
                            ['MV', str(grid.id_db), 'loa', str(node.id_db)]),
                        bus='_'.join(
                            ['MV', str(grid.id_db), 'trd', str(node.id_db)]),
                        grid_id=grid.id_db)
                else:
                    # Add aggregated LV load to MV bus
                    load = orm_pypsa.Load(
                        load_id='_'.join(
                            ['MV', str(grid.id_db), 'loa', str(node.id_db)]),
                        bus=node.pypsa_id,
                        grid_id=grid.id_db)
                load_pq_set = orm_pypsa.LoadPqSet(
                    load_id='_'.join(
                        ['MV', str(grid.id_db), 'loa', str(node.id_db)]),
                    temp_id=temp_id,
                    p_set=[node.peak_load * kw2mw,
                           load_in_generation_case * kw2mw],
                    q_set=[node.peak_load * Q_factor_load * kw2mw,
                           load_in_generation_case],
                    grid_id=grid.id_db)
                session.add(load)
                session.add(load_pq_set)
            elif isinstance(node, LVLoadAreaCentreDingo):
                load = orm_pypsa.Load(
                    load_id=node.pypsa_id,
                    bus='_'.join(['HV', str(grid.id_db), 'trd']),
                    grid_id=grid.id_db)
                load_pq_set = orm_pypsa.LoadPqSet(
                    load_id=node.pypsa_id,
                    temp_id=temp_id,
                    p_set=[node.lv_load_area.peak_load_sum * kw2mw,
                           load_in_generation_case * kw2mw],
                    q_set=[node.lv_load_area.peak_load_sum
                           * Q_factor_load * kw2mw,
                           load_in_generation_case * kw2mw],
                    grid_id=grid.id_db)
                session.add(load)
                session.add(load_pq_set)
            elif isinstance(node, MVCableDistributorDingo):
                bus = orm_pypsa.Bus(
                    bus_id=node.pypsa_id,
                    v_nom=grid.v_level,
                    geom=from_shape(node.geo_data, srid=srid),
                    grid_id=grid.id_db)
                bus_pq_set = orm_pypsa.BusVMagSet(
                    bus_id=node.pypsa_id,
                    temp_id=temp_id,
                    v_mag_pu_set=[1, 1],
                    grid_id=grid.id_db)
                session.add(bus)
                session.add(bus_pq_set)
            elif isinstance(node, MVStationDingo):
                print('Only MV side bus of MVStation will be added.')
                bus_mv_station = orm_pypsa.Bus(
                    bus_id=node.pypsa_id,
                    v_nom=grid.v_level,
                    geom=from_shape(node.geo_data, srid=srid),
                    grid_id=grid.id_db)
                bus_pq_set_mv_station = orm_pypsa.BusVMagSet(
                    bus_id=node.pypsa_id,
                    temp_id=temp_id,
                    v_mag_pu_set=[voltage_set_slack, voltage_set_slack],
                    grid_id=grid.id_db)
                slack_gen = orm_pypsa.Generator(
                    generator_id='_'.join(['MV', str(grid.id_db), 'slack']),
                    bus=node.pypsa_id,
                    control='Slack',
                    grid_id=grid.id_db)
                session.add(bus_mv_station)
                session.add(bus_pq_set_mv_station)
                session.add(slack_gen)
            elif isinstance(node, GeneratorDingo):
                bus_gen = orm_pypsa.Bus(
                    bus_id=node.pypsa_id,
                    v_nom=grid.v_level,
                    geom=from_shape(node.geo_data, srid=srid),
                    grid_id=grid.id_db)
                bus_pq_set_gen = orm_pypsa.BusVMagSet(
                    bus_id=node.pypsa_id,
                    temp_id=temp_id,
                    v_mag_pu_set=[1, 1],
                    grid_id=grid.id_db)
                generator = orm_pypsa.Generator(
                    generator_id='_'.join(
                        ['MV', str(grid.id_db), 'gen', str(node.id_db)]),
                    bus=node.pypsa_id,
                    control='PQ',
                    p_nom=node.capacity,
                    grid_id=grid.id_db)
                generator_pq_set = orm_pypsa.GeneratorPqSet(
                    generator_id='_'.join(
                        ['MV', str(grid.id_db), 'gen', str(node.id_db)]),
                    temp_id=temp_id,
                    p_set=[0 * kw2mw, node.capacity * kw2mw],
                    q_set=[0 * kw2mw, 0 * kw2mw],
                    grid_id=grid.id_db)
                session.add(bus_gen)
                session.add(bus_pq_set_gen)
                session.add(generator)
                session.add(generator_pq_set)
            elif isinstance(node, CircuitBreakerDingo):
                # TODO: remove this elif-case if CircuitBreaker are removed from graph
                continue
            else:
                raise TypeError("Node of type", node, "cannot be handled here")
        else:
            print("Node {} is not connected to the graph and will be omitted " \
                  "in power flow analysis".format(node))

    # write changes to database
    session.commit()


def export_edges(grid, session, edges):
    """Export nodes of grid graph representation to pypsa input tables

    Parameters
    ----------
    grid: MVGridDingo
        Instance of MVGridDingo class
    session: SQLAlchemy session object
    """

    omega = 2 * pi * 50
    srid = int(cfg_dingo.get('geo', 'srid'))

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

        line = orm_pypsa.Line(
            line_id=line_name,
            bus0=edge['adj_nodes'][0].pypsa_id,
            bus1=edge['adj_nodes'][1].pypsa_id,
            x=x * l,
            r=r * l,
            s_nom=s_nom,
            length=l,
            cables=3,
            geom=from_shape(LineString([edge['adj_nodes'][0].geo_data,
                                        edge['adj_nodes'][1].geo_data]),
                            srid=srid),
            grid_id=grid.id_db
        )
        session.add(line)
        session.commit()


def create_temp_resolution_table(session, timesteps, start_time, resolution='H',
                                 temp_id=1):
    """
    Write info about temporal coverage into table `temp_resolution`
    """

    temp_resolution = orm_pypsa.TempResolution(
        temp_id=temp_id,
        timesteps=timesteps,
        resolution=resolution,
        start_time=start_time
    )
    session.add(temp_resolution)
    session.commit()


def nodes_to_dict_of_dataframes(grid, nodes, lv_transformer=True):
    """
    Creates dictionary of dataframes containing grid

    Parameters
    ----------
    nodes: list of dingo grid components objects
        Nodes of the grid graph
    lv_transformer: bool, True
        Toggle transformer representation in power flow analysis

    Returns:
    components: dict of pandas.DataFrame
        DataFrames contain components attributes. Dict is keyed by components
        type
    """

    bus_instances = [LVStationDingo, GeneratorDingo, MVCableDistributorDingo,
                     MVStationDingo]

    generator_instances = [MVStationDingo, GeneratorDingo]
    # TODO: MVStationDingo has a slack generator

    mv_routing_loads_cos_phi = float(
        cfg_dingo.get('mv_routing_tech_constraints',
                      'mv_routing_loads_cos_phi'))
    srid = int(cfg_dingo.get('geo', 'srid'))

    load_in_generation_case = cfg_dingo.get('assumptions',
                                            'load_in_generation_case')

    Q_factor_load = tan(acos(mv_routing_loads_cos_phi))

    voltage_set_slack = cfg_dingo.get("mv_routing_tech_constraints",
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

    # TODO: consider other implications of `lv_transformer is True`
    if lv_transformer is True:
        bus_instances.append(Transformer)

    for node in nodes:
        if node not in grid.graph_isolated_nodes():
            # buses only
            if isinstance(node, MVCableDistributorDingo):
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
                if isinstance(node, MVStationDingo):
                    print('Only MV side bus of MVStation will be added.')
                    generator['generator_id'].append(
                        '_'.join(['MV', str(grid.id_db), 'slack']))
                    generator['control'].append('Slack')
                    generator['p_nom'].append(0)
                    bus_v_mag_set['v_mag_pu_set'].append(
                        [voltage_set_slack, voltage_set_slack])

                # other generators
                if isinstance(node, GeneratorDingo):
                    generator['generator_id'].append(
                        '_'.join(['MV', str(grid.id_db), 'gen']))
                    generator['control'].append('PQ')
                    generator['p_nom'].append(node.capacity)

                    generator_pq_set['generator_id'].append(
                        '_'.join(['MV', str(grid.id_db), 'gen']))
                    generator_pq_set['temp_id'].append(1)
                    generator_pq_set['p_set'].append(
                        [0 * kw2mw, node.capacity * kw2mw])
                    generator_pq_set['q_set'].append(
                        [0 * kw2mw, 0 * kw2mw])
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
            elif isinstance(node, LVLoadAreaCentreDingo):
                load['load_id'].append(node.pypsa_id)
                load['bus'].append('_'.join(['HV', str(grid.id_db), 'trd']))
                load['grid_id'].append(grid.id_db)

                load_pq_set['load_id'].append(node.pypsa_id)
                load_pq_set['temp_id'].append(1)
                load_pq_set['p_set'].append(
                    [node.lv_load_area.peak_load_sum * kw2mw,
                     load_in_generation_case * kw2mw])
                load_pq_set['q_set'].append(
                    [node.lv_load_area.peak_load_sum
                     * Q_factor_load * kw2mw,
                     load_in_generation_case * kw2mw])
                load_pq_set['grid_id'].append(grid.id_db)

            # bus + aggregate load of lv grids (at mv/ls substation)
            elif isinstance(node, LVStationDingo):
                load['load_id'].append(
                    '_'.join(['MV', str(grid.id_db), 'loa', str(node.id_db)]))
                load['bus'].append(node.pypsa_id)
                load['grid_id'].append(grid.id_db)

                load_pq_set['load_id'].append(node.pypsa_id)
                load_pq_set['temp_id'].append(1)
                load_pq_set['p_set'].append(
                    [node.peak_load * kw2mw,
                     load_in_generation_case * kw2mw])
                load_pq_set['q_set'].append(
                    [node.peak_load * Q_factor_load * kw2mw,
                     load_in_generation_case])
                load_pq_set['grid_id'].append(grid.id_db)

                buses['bus_id'].append(node.pypsa_id)
                buses['v_nom'].append(grid.v_level)
                buses['geom'].append(from_shape(node.geo_data, srid=srid))
                buses['grid_id'].append(grid.id_db)

                bus_v_mag_set['bus_id'].append(node.pypsa_id)
                bus_v_mag_set['temp_id'].append(1)
                bus_v_mag_set['v_mag_pu_set'].append([1, 1])
                bus_v_mag_set['grid_id'].append(grid.id_db)

            elif isinstance(node, CircuitBreakerDingo):
                # TODO: remove this elif-case if CircuitBreaker are removed from graph
                continue
            else:
                raise TypeError("Node of type", node, "cannot be handled here")
        else:
            print("Node {} is not connected to the graph and will be omitted " \
                  "in power flow analysis".format(node))

    components = {'Bus': DataFrame(buses).set_index('bus_id'),
                  'Generator': DataFrame(generator).set_index('generator_id'),
                  'Load': DataFrame(load).set_index('load_id')}

    components_data = {'Bus': DataFrame(bus_v_mag_set).set_index('bus_id'),
                       'Generator': DataFrame(generator_pq_set).set_index(
                           'generator_id'),
                       'Load': DataFrame(load_pq_set).set_index('load_id')}

    return components, components_data


def edges_to_dict_of_dataframes(grid, edges):
    """
    Export edges to DataFrame

    :param grid:
    :param edges:
    Returns
    -------
    edges_dict: dict
    """
    omega = 2 * pi * 50
    srid = int(cfg_dingo.get('geo', 'srid'))

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

    return {'Line': DataFrame(lines)}



def run_powerflow(conn):
    """
    Run powerflow to test grid stability

    Two cases are defined to be tested here:
     i) load case
     ii) feed-in case

    Parameters
    ----------
    conn: SQLAlchemy session object

    """

    scenario = cfg_dingo.get("powerflow", "test_grid_stability_scenario")
    start_hour = cfg_dingo.get("powerflow", "start_hour")
    end_hour = cfg_dingo.get("powerflow", "end_hour")

    # choose temp_id
    temp_id_set = 1

    # define investigated time range
    timerange = get_timerange(conn, temp_id_set, TempResolution)

    # define relevant tables
    tables = [Bus, Line, Generator, Load, Transformer]

    # get components from database tables
    components = import_components(tables, conn, scenario)

    # create PyPSA powerflow problem
    network, snapshots = create_powerflow_problem(timerange, components)

    # import pq-set tables to pypsa network (p_set for generators and loads)
    pq_object = [GeneratorPqSet, LoadPqSet]
    network = import_pq_sets(conn,
                             network,
                             pq_object,
                             timerange,
                             scenario,
                             columns=['p_set'],
                             start_h=start_hour,
                             end_h=end_hour)

    # import pq-set table to pypsa network (q_set for loads)
    network = import_pq_sets(conn,
                             network,
                             pq_object,
                             timerange,
                             scenario,
                             columns=['q_set'],
                             start_h=start_hour,
                             end_h=end_hour)

    # Import `v_mag_pu_set` for Bus
    network = import_pq_sets(conn,
                             network,
                             [BusVMagSet],
                             timerange,
                             scenario,
                             columns=['v_mag_pu_set'],
                             start_h=start_hour,
                             end_h=end_hour)

    # add coordinates to network nodes and make ready for map plotting
    network = add_coordinates(network)

    # start powerflow calculations
    network.pf(snapshots)

    # make a line loading plot
    # TODO: make this optional
    plot_line_loading(network, timestep=0,
                      filename='Line_loading_load_case.png')
    plot_line_loading(network, timestep=1,
                      filename='Line_loading_feed-in_case.png')

    results_to_oedb(conn, network)


def import_pfa_bus_results(session, grid):
    """
    Assign results from power flow analysis to grid network object

    As this function operates on nodes/buses of the graph, voltage levels from
    power flow analysis are assigned to attribute `voltage_res` of node objects
    of the mv grid graph.
    The attribute `voltage_res` is a list of two elements
    1. voltage in load case
    2. voltage in feed-in case

    Parameters
    ----------
    session: SQLAlchemy session object
    grid: networkX graph

    Returns
    -------
    None
    """

    # get bus data from database
    bus_query = session.query(ResBus.bus_id,
                              ResBus.v_mag_pu). \
        join(Bus, ResBus.bus_id == Bus.bus_id). \
        filter(Bus.grid_id == grid.id_db)

    bus_data = read_sql_query(bus_query.statement,
                              session.bind,
                              index_col='bus_id')

    # iterate of nodes and assign voltage obtained from power flow analysis
    for node in grid._graph.nodes():
        # check if node is connected to graph
        if node not in grid.graph_isolated_nodes():
            if isinstance(node, LVStationDingo):
                node.voltage_res = bus_data.loc[node.pypsa_id, 'v_mag_pu']
            elif isinstance(node, (LVStationDingo, LVLoadAreaCentreDingo)):
                if node.lv_load_area.is_aggregated:
                    node.voltage_res = bus_data.loc[node.pypsa_id, 'v_mag_pu']
            elif not isinstance(node, CircuitBreakerDingo):
                node.voltage_res = bus_data.loc[node.pypsa_id, 'v_mag_pu']
            else:
                print("Object {} has been skipped while importing results!")


def import_pfa_line_results(session, grid):
    """
    Assign results from power flow analysis to grid network object

    As this function operates on branches/lines of the graph, power flows from
    power flow analysis are assigned to attribute `s_res` of edge objects
    of the mv grid graph.
    The attribute `s_res` is computed according to

    .. math::
        s = \sqrt{{max(p0, p1)}^2 + {max(p0, p1)}^2}

    Parameters
    ----------
    session: SQLAlchemy session object
    grid: networkX graph

    Returns
    -------
    None
    """

    # get lines data from database
    lines_query = session.query(ResLine.line_id,
                                ResLine.p0,
                                ResLine.p1,
                                ResLine.q0,
                                ResLine.q1). \
        join(Line, ResLine.line_id == Line.line_id). \
        filter(Line.grid_id == grid.id_db)

    line_data = read_sql_query(lines_query.statement,
                               session.bind,
                               index_col='line_id')

    edges = [edge for edge in grid.graph_edges()
             if edge['adj_nodes'][0] in grid._graph.nodes()
             and edge['adj_nodes'][1] in grid._graph.nodes()]

    for edge in edges:
        s_res = [
            sqrt(
                max(abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                    'branch'].id_db), 'p0'][0]),
                    abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                        'branch'].id_db), 'p1'][0])) ** 2 +
                max(abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                    'branch'].id_db), 'q0'][0]),
                    abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                        'branch'].id_db), 'q1'][0])) ** 2),
            sqrt(
                max(abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                    'branch'].id_db), 'p0'][1]),
                    abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                        'branch'].id_db), 'p1'][1])) ** 2 +
                max(abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                    'branch'].id_db), 'q0'][1]),
                    abs(line_data.loc["MV_{0}_lin_{1}".format(grid.id_db, edge[
                        'branch'].id_db), 'q1'][1])) ** 2)]

        edge['branch'].s_res = s_res


def import_pfa_transformer_results():
    pass
