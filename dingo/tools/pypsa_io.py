from egoio.db_tables import calc_ego_mv_powerflow as orm_pypsa
from dingo.tools import config as cfg_dingo
from dingo.core.network.stations import LVStationDingo, MVStationDingo
from dingo.core.network import BranchDingo, CircuitBreakerDingo, GeneratorDingo
from dingo.core import MVCableDistributorDingo
from dingo.core.structure.regions import LVLoadAreaCentreDingo

from geoalchemy2.shape import from_shape
from math import tan, acos, pi
from shapely.geometry import LineString
from pandas import Series
import networkx as nx

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
    
    kw2mw = 1e-3

    # Create all busses
    # TODO: incorporates CableDists, LVStations
    # TODO: for all LVStations a representative load has to be added
    # TODO: use `for node in nd._mv_grid_districts[0].mv_grid._graph.node`
    for node in nodes:
        if isinstance(node, LVStationDingo):
            if node.lv_load_area.is_connected and grid._graph.adj[node]:
                # MV side bus
                bus_mv = orm_pypsa.Bus(
                    bus_id=node.pypsa_id,
                    v_nom=grid.v_level,
                    geom=from_shape(node.geo_data, srid=srid),
                    grid_id=grid.id_db)
                bus_pq_set_mv = orm_pypsa.BusVMagSet(
                    bus_id=node.pypsa_id,
                    temp_id = temp_id,
                    v_mag_pu_set=[1, 1],
                    grid_id=grid.id_db)
                session.add(bus_mv)
                session.add(bus_pq_set_mv)
                if lv_transformer is True:
                    # Add transformer to bus
                    transformer = orm_pypsa.Transformer(
                        trafo_id='_'.join(['MV', str(grid.id_db), 'trf', str(node.id_db)]),
                        s_nom=node._transformers[0].s_max_a,
                        bus0=node.pypsa_id,
                        bus1='_'.join(['MV', str(grid.id_db), 'trd', str(node.id_db)]),
                        x=node._transformers[0].x,
                        r=node._transformers[0].r,
                        tap_ratio=1,
                        geom=from_shape(node.geo_data, srid=srid),
                        grid_id=grid.id_db)
                    session.add(transformer)
                    # Add bus on transformer's LV side
                    bus_lv = orm_pypsa.Bus(
                        bus_id='_'.join(['MV', str(grid.id_db), 'trd', str(node.id_db)]),
                        v_nom=node._transformers[0].v_level,
                        geom=from_shape(node.geo_data, srid=srid),
                        grid_id=grid.id_db)
                    bus_pq_set_lv = orm_pypsa.BusVMagSet(
                        bus_id='_'.join(['MV', str(grid.id_db), 'trd', str(node.id_db)]),
                        temp_id=temp_id,
                        v_mag_pu_set=[1, 1],
                        grid_id=grid.id_db)
                    session.add(bus_lv)
                    session.add(bus_pq_set_lv)
                    # Add aggregated LV load to LV bus
                    load = orm_pypsa.Load(
                        load_id = '_'.join(['MV', str(grid.id_db), 'loa', str(node.id_db)]),
                        bus = '_'.join(['MV', str(grid.id_db), 'trd', str(node.id_db)]),
                        grid_id=grid.id_db)
                else:
                    # Add aggregated LV load to MV bus
                    load = orm_pypsa.Load(
                        load_id='_'.join(
                            ['MV', str(grid.id_db), 'loa', str(node.id_db)]),
                        bus=node.pypsa_id,
                        grid_id=grid.id_db)
                load_pq_set = orm_pypsa.LoadPqSet(
                    load_id = '_'.join(['MV', str(grid.id_db), 'loa', str(node.id_db)]),
                    temp_id = temp_id,
                    p_set = [node.peak_load * kw2mw,
                             load_in_generation_case * kw2mw],
                    q_set = [node.peak_load * Q_factor_load * kw2mw,
                             load_in_generation_case],
                    grid_id=grid.id_db)
                session.add(load)
                session.add(load_pq_set)
        elif isinstance(node, LVLoadAreaCentreDingo):
            if node.lv_load_area.is_connected:
                if node.lv_load_area.is_aggregated:
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
                v_mag_pu_set=[1, 1],
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
                generator_id='_'.join(['MV', str(grid.id_db), 'gen', str(node.id_db)]),
                bus=node.pypsa_id,
                control='PQ',
                p_nom=node.capacity,
                grid_id=grid.id_db)
            generator_pq_set = orm_pypsa.GeneratorPqSet(
                generator_id='_'.join(['MV', str(grid.id_db), 'gen', str(node.id_db)]),
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
            raise TypeError("Node of type", node, "cannot handled here")

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

    # for edge in grid.graph_edges():
    for edge in edges:
        # if isinstance(edge, BranchDingo):
        if not (isinstance(edge['adj_nodes'][0],
                           (LVLoadAreaCentreDingo,
                            # MVStationDingo,
                            CircuitBreakerDingo)) or
                    isinstance(edge['adj_nodes'][1],
                               (LVLoadAreaCentreDingo,
                                # MVStationDingo,
                                CircuitBreakerDingo))):
            # TODO: check s_nom calculation
            # TODO: 1. do we need to consider 3 cables
            # TODO: 2. do we need to respect to consider a load factor

            # TODO: find the real cause for being L, C, I_th_max type of Series
            if (isinstance(edge['branch'].type['L'], Series) or
                isinstance(edge['branch'].type['C'], Series)):
                # x = omega * edge['branch'].type['L'].values[0] * 1e-3 - 1 / (
                #     omega * edge['branch'].type['C'].values[0] * 1e-6)
                # TODO: not sure if capacity C can be omitted
                x = omega * edge['branch'].type['L'].values[0] * 1e-3
            else:
                # x = omega * edge['branch'].type['L'] * 1e-3 - 1 / (
                #     omega * edge['branch'].type['C'] * 1e-6)
                x = omega * edge['branch'].type['L'] * 1e-3

            if isinstance(edge['branch'].type['R'], Series) :
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

            line = orm_pypsa.Line(
                line_id='_'.join(['MV', str(grid.id_db), 'lin', str(edge['branch'].id_db)]),
                bus0=edge['adj_nodes'][0].pypsa_id,
                bus1=edge['adj_nodes'][1].pypsa_id,
                x=x * edge['branch'].length / 1e3,
                r=r * edge['branch'].length / 1e3,
                s_nom=s_nom,
                length=edge['branch'].length / 1e3,
                cables=3,
                geom=from_shape(LineString([edge['adj_nodes'][0].geo_data,
                                 edge['adj_nodes'][1].geo_data]), srid=srid),
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