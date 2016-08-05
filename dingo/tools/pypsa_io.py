from egoio.db_tables import calc_ego_mv_powerflow as orm_pypsa
from dingo.tools import config as cfg_dingo
from dingo.core.network.stations import LVStationDingo, MVStationDingo
from dingo.core.network import BranchDingo, CircuitBreakerDingo, GeneratorDingo
from dingo.core import MVCableDistributorDingo
from dingo.core.structure.regions import LVLoadAreaCentreDingo

from geoalchemy2.shape import from_shape
from math import tan, acos


def delete_powerflow_tables(session):
    """Empties all powerflow tables to start export from scratch

    Parameters
    ----------
    session: SQLAlchemy session object
    """
    tables = [orm_pypsa.Bus, orm_pypsa.BusVMagSet, orm_pypsa.Load,
              orm_pypsa.LoadPqSet]

    for table in tables:
        session.query(table).delete()
    session.commit()


def export_nodes(grid, session, temp_id):
    """Export nodes of grid graph representation to pypsa input tables

    Parameters
    ----------
    grid: MVGridDingo
        Instance of MVGridDingo class
    session: SQLAlchemy session object
    temp_id: int
        ID of `temp_resolution` table

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


    # Create all busses
    # TODO: incorporates CableDists, LVStations
    # TODO: for all LVStations a representative load has to be added
    # TODO: use `for node in nd._mv_grid_districts[0].mv_grid._graph.node`
    for node in grid._graph.nodes():
        if isinstance(node, LVStationDingo):
            if node.lv_load_area.is_connected:
                # MV side bus
                bus_mv = orm_pypsa.Bus(
                    bus_id = '_'.join(['MV', str(grid.id_db), 'tru', str(node.id_db)]),
                    v_nom = grid.v_level,
                    geom = from_shape(node.geo_data, srid=srid))
                bus_pq_set_mv = orm_pypsa.BusVMagSet(
                    bus_id='_'.join(['MV', str(grid.id_db), 'tru', str(node.id_db)]),
                    temp_id = temp_id,
                    v_mag_pu_set=[1, 1])
                # Add transformer to bus
                transformer = orm_pypsa.Transformer(
                    trafo_id='_'.join(['MV', str(grid.id_db), 'trf', str(node.id_db)]),
                    s_nom=node._transformers[0].s_max_a,
                    bus0='_'.join(['MV', str(grid.id_db), 'tru', str(node.id_db)]),
                    bus1='_'.join(['MV', str(grid.id_db), 'trd', str(node.id_db)]),
                    x=node._transformers[0].x,
                    r=node._transformers[0].r,
                    tap_ratio=1,
                    geom=from_shape(node.geo_data, srid=srid))
                # Add bus on transformer's LV side
                bus_lv = orm_pypsa.Bus(
                    bus_id='_'.join(['MV', str(grid.id_db), 'trd', str(node.id_db)]),
                    v_nom=node._transformers[0].v_level,
                    geom=from_shape(node.geo_data, srid=srid))
                bus_pq_set_lv = orm_pypsa.BusVMagSet(
                    bus_id='_'.join(['MV', str(grid.id_db), 'trd', str(node.id_db)]),
                    temp_id=temp_id,
                    v_mag_pu_set=[1, 1])
                # Add aggregated LV load to bus
                load = orm_pypsa.Load(
                    load_id = '_'.join(['MV', str(grid.id_db), 'loa', str(node.id_db)]),
                    bus = '_'.join(['MV', str(grid.id_db), 'trd', str(node.id_db)]))
                load_pq_set = orm_pypsa.LoadPqSet(
                    load_id = '_'.join(['MV', str(grid.id_db), 'loa', str(node.id_db)]),
                    temp_id = temp_id,
                    p_set = [node.peak_load, load_in_generation_case],
                    q_set = [node.peak_load *
                             Q_factor_load, load_in_generation_case])
            session.add(bus_mv)
            session.add(bus_pq_set_mv)
            session.add(transformer)
            session.add(bus_lv)
            session.add(bus_pq_set_lv)
            session.add(load)
            session.add(load_pq_set)
        elif isinstance(node, LVLoadAreaCentreDingo):
            if node.lv_load_area.is_connected:
                if node.lv_load_area.is_aggregated:
                    load = orm_pypsa.Load(
                        load_id='_'.join(['MV', str(grid.id_db), 'loa', str(node.id_db)]),
                        bus='_'.join(['HV', str(grid.id_db), 'trd', str(node.id_db)]))
                    load_pq_set = orm_pypsa.LoadPqSet(
                        load_id='_'.join(['MV', str(grid.id_db), 'loa', str(node.id_db)]),
                        temp_id=temp_id,
                        p_set=[node.lv_load_area.peak_load_sum,
                               load_in_generation_case],
                        q_set=[node.lv_load_area.peak_load_sum
                               * Q_factor_load, load_in_generation_case])
                    session.add(load)
                    session.add(load_pq_set)
        elif isinstance(node, MVCableDistributorDingo):
            bus = orm_pypsa.Bus(
                bus_id='_'.join(['MV', str(grid.id_db), 'cld', str(node.id_db)]),
                v_nom=grid.v_level,
                geom=from_shape(node.geo_data, srid=srid))
            bus_pq_set = orm_pypsa.BusVMagSet(
                bus_id='_'.join(['MV', str(grid.id_db), 'cld', str(node.id_db)]),
                temp_id=temp_id,
                v_mag_pu_set=[1, 1])
            session.add(bus)
            session.add(bus_pq_set)
        elif isinstance(node, MVStationDingo):
            print('Adding of MVStations is currently missing...')
        elif isinstance(node, GeneratorDingo):
            bus_gen = orm_pypsa.Bus(
                bus_id='_'.join(['MV', str(grid.id_db), 'gen', str(node.id_db)]),
                v_nom=grid.v_level,
                geom=from_shape(node.geo_data, srid=srid))
            bus_pq_set_gen = orm_pypsa.BusVMagSet(
                bus_id='_'.join(['MV', str(grid.id_db), 'gen', str(node.id_db)]),
                temp_id=temp_id,
                v_mag_pu_set=[1, 1])
            generator = orm_pypsa.Generator(
                generator_id='_'.join(['MV', str(grid.id_db), 'gen', str(node.id_db)]),
                bus='_'.join(['MV', str(grid.id_db), 'gen', str(node.id_db)]),
                control='PQ',
                p_nom=node.capacity)
            generator_pq_set = orm_pypsa.GeneratorPqSet(
                generator_id='_'.join(['MV', str(grid.id_db), 'gen', str(node.id_db)]),
                temp_id=temp_id,
                p_set=[0, node.capacity],
                q_set=[0, 0])
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