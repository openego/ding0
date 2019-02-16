"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__ = "Reiner Lemoine Institut gGmbH"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__ = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__ = "nesnoj, gplssm"

import numpy as np
import pandas as pd
from collections import namedtuple

import json

from ding0.core import NetworkDing0
from ding0.core import GeneratorDing0, GeneratorFluctuatingDing0
from ding0.core import LVCableDistributorDing0, MVCableDistributorDing0
from ding0.core import MVStationDing0, LVStationDing0
from ding0.core import CircuitBreakerDing0
from ding0.core.network.loads import LVLoadDing0, MVLoadDing0
from ding0.core import LVLoadAreaCentreDing0

from shapely.geometry import Point, MultiPoint, MultiLineString, LineString, MultiPolygon, shape, mapping

Network = namedtuple(
    'Network',
    [
        'run_id', 'metadata_json', 'lv_grid', 'lv_gen', 'lv_cd', 'lv_stations', 'mvlv_trafos', 'lv_loads',
        'mv_grid', 'mv_gen', 'mv_cb', 'mv_cd', 'mv_stations', 'hvmv_trafos', 'mv_loads', 'lines', 'mvlv_mapping'
    ]
)


def export_network(nw, mode='', run_id=None):
    """
    Export all nodes and lines of the network nw as DataFrames

    Parameters
    ----------
    nw: :any:`list` of NetworkDing0
        The MV grid(s) to be studied
    mode: str
        If 'MV' export only medium voltage nodes and lines
        If 'LV' export only low voltage nodes and lines
        else, exports MV and LV nodes and lines

    Returns
    -------
    pandas.DataFrame
        nodes_df : Dataframe containing nodes and its attributes
    pandas.DataFrame
        lines_df : Dataframe containing lines and its attributes
    """

    # close circuit breakers
    nw.control_circuit_breakers(mode='close')
    # srid
    srid = str(int(nw.config['geo']['srid']))
    ##############################
    # check what to do
    lv_info = True
    mv_info = True
    if mode == 'LV':
        mv_info = False
    if mode == 'MV':
        lv_info = False
    ##############################
    # from datetime import datetime
    print("1 " + str(run_id))
    if not run_id:
        run_id = nw.metadata['run_id']  # datetime.now().strftime("%Y%m%d%H%M%S")
        metadata_json = json.dumps(nw.metadata)
    else:
        print("test")
        # nw.metadata['run_id'] = run_id

        print("2 " + str(run_id))
        metadata_json = json.dumps(nw.metadata(run_id))

    metadata_json1 = json.loads(metadata_json)
    print("3" + str(metadata_json1['run_id']))
    ##############################
    #############################
    # go through the grid collecting info
    lvgrid_idx = 0
    lv_grid_dict = {}
    lvloads_idx = 0
    lv_loads_dict = {}
    mvgrid_idx = 0
    mv_grid_dict = {}
    mvloads_idx = 0
    mv_loads_dict = {}
    mvgen_idx = 0
    mv_gen_dict = {}
    mvcb_idx = 0
    mvcb_dict = {}
    mvcd_idx = 0
    mv_cd_dict = {}
    # mvstations_idx = 0
    mv_stations_dict = {}
    mvtrafos_idx = 0
    hvmv_trafos_dict = {}
    lvgen_idx = 0
    lv_gen_dict = {}
    lvcd_idx = 0
    lv_cd_dict = {}
    lvstations_idx = 0
    lv_stations_dict = {}
    lvtrafos_idx = 0
    mvlv_trafos_dict = {}
    areacenter_idx = 0
    areacenter_dict = {}
    lines_idx = 0
    lines_dict = {}
    LVMVmapping_idx = 0
    mvlv_mapping_dict = {}

    def aggregate_loads(la_center, aggr):
        """Aggregate consumption in load area per sector
        Parameters
        ----------
        la_center: LVLoadAreaCentreDing0
            Load area center object from Ding0
        Returns
        -------
        """
        for s in ['retail', 'industrial', 'agricultural', 'residential']:
            if s not in aggr['load']:
                aggr['load'][s] = {}

            for t in ['nominal', 'peak']:
                if t not in aggr['load'][s]:
                    aggr['load'][s][t] = 0

        aggr['load']['retail']['nominal'] += sum(
            [_.sector_consumption_retail
             for _ in la_center.lv_load_area._lv_grid_districts])
        aggr['load']['industrial']['nominal'] += sum(
            [_.sector_consumption_industrial
             for _ in la_center.lv_load_area._lv_grid_districts])
        aggr['load']['agricultural']['nominal'] += sum(
            [_.sector_consumption_agricultural
             for _ in la_center.lv_load_area._lv_grid_districts])
        aggr['load']['residential']['nominal'] += sum(
            [_.sector_consumption_residential
             for _ in la_center.lv_load_area._lv_grid_districts])

        aggr['load']['retail']['peak'] += sum(
            [_.peak_load_retail
             for _ in la_center.lv_load_area._lv_grid_districts])
        aggr['load']['industrial']['peak'] += sum(
            [_.peak_load_industrial
             for _ in la_center.lv_load_area._lv_grid_districts])
        aggr['load']['agricultural']['peak'] += sum(
            [_.peak_load_agricultural
             for _ in la_center.lv_load_area._lv_grid_districts])
        aggr['load']['residential']['peak'] += sum(
            [_.peak_load_residential
             for _ in la_center.lv_load_area._lv_grid_districts])

        return aggr

    for mv_district in nw.mv_grid_districts():
        from shapely.wkt import dumps as wkt_dumps
        mv_grid_id = mv_district.mv_grid.id_db
        mv_grid_name = '_'.join(
            [str(mv_district.mv_grid.__class__.__name__), 'MV', str(mv_grid_id),
             str(mv_district.mv_grid.id_db)])

        if mv_info:
            lv_grid_id = 0

            # MV-grid
            # ToDo: geom <- Polygon
            mvgrid_idx += 1
            mv_grid_dict[mvgrid_idx] = {
                'id': mv_grid_id,
                'name': mv_grid_name,
                'geom': wkt_dumps(mv_district.geo_data),
                'population':
                    sum([_.zensus_sum
                         for _ in
                         mv_district._lv_load_areas
                         if not np.isnan(_.zensus_sum)]),
                'voltage_nom': mv_district.mv_grid.v_level,  # in kV
                'run_id': run_id
            }

            # MV station
            mv_station = mv_district.mv_grid._station
            mv_station_name = '_'.join(
                ['MVStationDing0', 'MV', str(mv_station.id_db),
                 str(mv_station.id_db)])
            mv_stations_dict[0] = {
                'id': mv_district.mv_grid.id_db,
                'name': mv_station_name,
                'geom': wkt_dumps(mv_station.geo_data),
                'run_id': run_id}

            # Trafos MV
            for t in mv_station.transformers():
                mvtrafos_idx += 1
                hvmv_trafos_dict[mvtrafos_idx] = {
                    'id': mv_station.id_db,
                    'geom': wkt_dumps(mv_station.geo_data),
                    'name': '_'.join(
                        ['MVTransformerDing0', 'MV', str(mv_station.id_db),
                         str(mv_station.id_db)]),
                    'voltage_op': t.v_level,
                    'S_nom': t.s_max_a,
                    'X': t.x,
                    'R': t.r,
                    'run_id': run_id,
                }

            # MV grid components
            for node in mv_district.mv_grid.graph_nodes_sorted():
                geom = wkt_dumps(node.geo_data)

                # LVStation
                if isinstance(node, LVStationDing0):
                    if not node.lv_load_area.is_aggregated:
                        lvstations_idx += 1
                        lv_grid_name = '_'.join(
                            ['LVGridDing0', 'LV', str(node.id_db),
                             str(node.id_db)])
                        lv_stations_dict[lvstations_idx] = {
                            'id': node.id_db,
                            'name': '_'.join([node.__class__.__name__,
                                              'MV', str(mv_grid_id),
                                              str(node.id_db)]),
                            'geom': geom,
                            'run_id': run_id,
                        }

                        # LV-MV mapping
                        LVMVmapping_idx += 1
                        mvlv_mapping_dict[LVMVmapping_idx] = {
                            'mv_grid_id': mv_grid_id,
                            'mv_grid_name': mv_grid_name,
                            'lv_grid_id': node.id_db,
                            'lv_grid_name': lv_grid_name,
                            'run_id': run_id,
                        }

                        # Trafos LV
                        for t in node.transformers():
                            lvtrafos_idx += 1
                            mvlv_trafos_dict[lvtrafos_idx] = {
                                'id': node.id_db,
                                'geom': geom,
                                'name': '_'.join(['LVTransformerDing0', 'LV',
                                                  str(node.id_db),
                                                  str(node.id_db)]),
                                'voltage_op': t.v_level,
                                'S_nom': t.s_max_a,
                                'X': t.x,
                                'R': t.r,
                                'run_id': run_id,
                            }

                # MVGenerator
                elif isinstance(node, (GeneratorDing0, GeneratorFluctuatingDing0)):
                    if node.subtype == None:
                        subtype = 'other'
                    else:
                        subtype = node.subtype
                    if isinstance(node, GeneratorFluctuatingDing0):
                        type = node.type
                        mvgen_idx += 1
                        mv_gen_dict[mvgen_idx] = {
                            'id': node.id_db,
                            'name': '_'.join(['GeneratorFluctuatingDing0', 'MV',
                                              str(mv_grid_id),
                                              str(node.id_db)]),
                            'geom': geom,
                            'type': type,
                            'subtype': subtype,
                            'v_level': node.v_level,
                            'nominal_capacity': node.capacity,
                            'run_id': run_id,
                            'is_aggregated': False,
                            'weather_cell_id': node.weather_cell_id
                        }
                    else:
                        type = node.type
                        mvgen_idx += 1
                        mv_gen_dict[mvgen_idx] = {
                            'id': node.id_db,
                            'name': '_'.join(
                                ['GeneratorDing0', 'MV', str(mv_grid_id),
                                 str(node.id_db)]),
                            'geom': geom,
                            'type': type,
                            'subtype': subtype,
                            'v_level': node.v_level,
                            'nominal_capacity': node.capacity,
                            'run_id': run_id,
                            'is_aggregated': False,
                            'weather_cell_id': np.nan
                        }

                # MVBranchTees
                elif isinstance(node, MVCableDistributorDing0):
                    mvcd_idx += 1
                    mv_cd_dict[mvcd_idx] = {
                        'id': node.id_db,
                        'name': '_'.join(
                            [str(node.__class__.__name__), 'MV',
                             str(mv_grid_id), str(node.id_db)]),
                        'geom': geom,
                        'run_id': run_id,
                    }

                # LoadAreaCentre
                elif isinstance(node, LVLoadAreaCentreDing0):

                    # type = 'Load area center of aggregated load area'

                    areacenter_idx += 1
                    aggr_lines = 0

                    aggr = {'generation': {}, 'load': {}, 'aggregates': []}

                    # Determine aggregated load in MV grid
                    # -> Implement once loads in Ding0 MV grids exist

                    # Determine aggregated load in LV grid
                    aggr = aggregate_loads(node, aggr)

                    # Collect metadata of aggregated load areas
                    aggr['aggregates'] = {
                        'population': node.lv_load_area.zensus_sum,
                        'geom': wkt_dumps(node.lv_load_area.geo_area)}
                    aggr_line_type = nw._static_data['MV_cables'].iloc[
                        nw._static_data['MV_cables']['I_max_th'].idxmax()]
                    geom = wkt_dumps(node.geo_data)

                    for aggr_node in aggr:
                        if aggr_node == 'generation':
                            pass

                        elif aggr_node == 'load':
                            for type in aggr['load']:
                                mvloads_idx += 1
                                aggr_line_id = 100 * node.lv_load_area.id_db + mvloads_idx + 1
                                mv_aggr_load_name = '_'.join(
                                    ['Load_aggregated', str(type),
                                     repr(mv_district.mv_grid),
                                     # str(node.lv_load_area.id_db)])
                                     str(aggr_line_id)])
                                mv_loads_dict[mvloads_idx] = {
                                    # Exception: aggregated loads get a string as id
                                    'id':  aggr_line_id, #node.lv_load_area.id_db, #mv_aggr_load_name,
                                    'name': mv_aggr_load_name,
                                    'geom': geom,
                                    'consumption': json.dumps(
                                        {type: aggr['load'][type]['nominal']}),
                                    'is_aggregated': True,
                                    'run_id': run_id,
                                }

                                lines_idx += 1
                                aggr_lines += 1
                                edge_name = '_'.join(
                                    ['line_aggr_load_la',
                                     str(node.lv_load_area.id_db), str(type),
                                     # str(node.lv_load_area.id_db)])
                                     str(aggr_line_id)])
                                lines_dict[lines_idx] = {
                                    'id': aggr_line_id, #node.lv_load_area.id_db,
                                    'edge_name': edge_name,
                                    'grid_name': mv_grid_name,
                                    'type_name': aggr_line_type.name,
                                    'type_kind': 'cable',
                                    'length': 1e-3,  # in km
                                    'U_n': aggr_line_type.U_n,
                                    'I_max_th': aggr_line_type.I_max_th,
                                    'R': aggr_line_type.R,
                                    'L': aggr_line_type.L,
                                    'C': aggr_line_type.C,
                                    'node1': mv_aggr_load_name,
                                    'node2': mv_station_name,
                                    'run_id': run_id,
                                    'geom': LineString([mv_station.geo_data, mv_station.geo_data])
                                }

                # TODO: eventually remove export of DisconnectingPoints from export
                # DisconnectingPoints
                elif isinstance(node, CircuitBreakerDing0):
                    mvcb_idx += 1
                    mvcb_dict[mvcb_idx] = {
                        'id': node.id_db,
                        'name': '_'.join([str(node.__class__.__name__), 'MV',
                                           str(mv_grid_id), str(node.id_db)]),
                        'geom': geom,
                        'status': node.status,
                        'run_id': run_id,
                    }
                else:
                    type = 'Unknown'

            # MVedges
            for branch in mv_district.mv_grid.graph_edges():
                # geom_string = from_shape(LineString([branch['adj_nodes'][0].geo_data,
                #                                     branch['adj_nodes'][1].geo_data]),
                #                                     srid=srid)
                # geom = wkt_dumps(geom_string)

                if not any([isinstance(branch['adj_nodes'][0],
                                       LVLoadAreaCentreDing0),
                            isinstance(branch['adj_nodes'][1],
                                       LVLoadAreaCentreDing0)]):
                    lines_idx += 1
                    lines_dict[lines_idx] = {
                        'id': branch['branch'].id_db,
                        'edge_name': '_'.join(
                            [branch['branch'].__class__.__name__,
                             str(branch['branch'].id_db)]),
                        'grid_name': mv_grid_name,
                        'type_name': branch['branch'].type['name'],
                        'type_kind': branch['branch'].kind,
                        'length': branch['branch'].length / 1e3,
                        'U_n': branch['branch'].type['U_n'],
                        'I_max_th': branch['branch'].type['I_max_th'],
                        'R': branch['branch'].type['R'],
                        'L': branch['branch'].type['L'],
                        'C': branch['branch'].type['C'],
                        'node1': '_'.join(
                            [str(branch['adj_nodes'][0].__class__.__name__),
                             'MV', str(mv_grid_id),
                             str(branch['adj_nodes'][0].id_db)]),
                        'node2': '_'.join(
                            [str(branch['adj_nodes'][1].__class__.__name__),
                             'MV', str(mv_grid_id),
                             str(branch['adj_nodes'][1].id_db)]),
                        'run_id': run_id,
                        'geom': LineString([branch['adj_nodes'][0].geo_data, branch['adj_nodes'][1].geo_data])
                    }

        if lv_info:
            for LA in mv_district.lv_load_areas():
                for lv_district in LA.lv_grid_districts():

                    if not lv_district.lv_grid.grid_district.lv_load_area.is_aggregated:
                        lv_grid_id = lv_district.lv_grid.id_db
                        lv_grid_name = '_'.join(
                            [str(lv_district.lv_grid.__class__.__name__), 'LV',
                             str(lv_district.lv_grid.id_db),
                             str(lv_district.lv_grid.id_db)])

                        lvgrid_idx += 1
                        lv_grid_dict[lvgrid_idx] = {
                            'id': lv_district.lv_grid.id_db,
                            'name': lv_grid_name,
                            'geom': wkt_dumps(lv_district.geo_data),
                            'population': lv_district.population,
                            'voltage_nom': lv_district.lv_grid.v_level / 1e3,
                            'run_id': run_id
                        }

                    # geom = from_shape(Point(lv_district.lv_grid.station().geo_data), srid=srid)
                    # geom = wkt_dumps(lv_district.geo_data)# lv_grid.station() #ding0_lv_grid.grid_district.geo_data
                    for node in lv_district.lv_grid.graph_nodes_sorted():
                        # geom = wkt_dumps(node.geo_data)

                        # LVGenerator
                        if isinstance(node, (GeneratorDing0, GeneratorFluctuatingDing0)):
                            if node.subtype == None:
                                subtype = 'other'
                            else:
                                subtype = node.subtype
                            if isinstance(node, GeneratorFluctuatingDing0):
                                type = node.type
                                lvgen_idx += 1
                                lv_gen_dict[lvgen_idx] = {
                                    'id': node.id_db,
                                    'la_id': LA.id_db,
                                    'name': '_'.join(
                                        ['GeneratorFluctuatingDing0', 'LV',
                                         str(lv_grid_id),
                                         str(node.id_db)]),
                                    'lv_grid_id': lv_grid_id,
                                    'geom': wkt_dumps(node.geo_data),
                                    'type': type,
                                    'subtype': subtype,
                                    'v_level': node.v_level,
                                    'nominal_capacity': node.capacity,
                                    'run_id': run_id,
                                    'is_aggregated': node.lv_load_area.is_aggregated,
                                    'weather_cell_id': node.weather_cell_id,
                                }
                            else:
                                type = node.type
                                lvgen_idx += 1
                                lv_gen_dict[lvgen_idx] = {
                                    'id': node.id_db,
                                    'name': '_'.join(
                                        ['GeneratorDing0', 'LV',
                                         str(lv_grid_id),
                                         str(node.id_db)]),
                                    'la_id': LA.id_db,
                                    'lv_grid_id': lv_grid_id,
                                    'geom': wkt_dumps(node.geo_data),
                                    'type': type,
                                    'subtype': subtype,
                                    'v_level': node.v_level,
                                    'nominal_capacity': node.capacity,
                                    'run_id': run_id,
                                    'is_aggregated': node.lv_load_area.is_aggregated,
                                    'weather_cell_id': np.nan
                                }

                        # LVcd
                        elif isinstance(node, LVCableDistributorDing0):
                            if not node.grid.grid_district.lv_load_area.is_aggregated:
                                lvcd_idx += 1
                                lv_cd_dict[lvcd_idx] = {
                                    'name': '_'.join(
                                        [str(node.__class__.__name__), 'LV',
                                         str(lv_grid_id), str(node.id_db)]),
                                    'id': node.id_db,
                                    'lv_grid_id': lv_grid_id,
                                    'geom': None,
                                    # wkt_dumps(lv_district.geo_data),#wkt_dumps(node.geo_data), Todo: why no geo_data?
                                    'run_id': run_id,
                                }

                        # LVload
                        elif isinstance(node, LVLoadDing0):
                            if not node.grid.grid_district.lv_load_area.is_aggregated:
                                lvloads_idx += 1
                                lv_loads_dict[lvloads_idx] = {
                                    'id': node.id_db,
                                    'name': '_'.join(
                                        [str(node.__class__.__name__), 'LV',
                                         str(lv_grid_id), str(node.id_db)]),
                                    'lv_grid_id': lv_grid_id,
                                    'geom': None,
                                    # wkt_dumps(lv_district.geo_data),#wkt_dumps(node.geo_data), Todo: why no geo_data?
                                    'consumption': json.dumps(node.consumption),
                                    'run_id': run_id,
                                }

                    # LVedges
                    for branch in lv_district.lv_grid.graph_edges():
                        if not branch['branch'].connects_aggregated:
                            if not any([isinstance(branch['adj_nodes'][0],
                                                   LVLoadAreaCentreDing0),
                                        isinstance(branch['adj_nodes'][1],
                                                   LVLoadAreaCentreDing0)]):
                                lines_idx += 1
                                lines_dict[lines_idx] = {
                                    'id': branch['branch'].id_db,
                                    'edge_name': '_'.join(
                                        [branch.__class__.__name__,
                                         str(branch['branch'].id_db)]),
                                    'grid_name': lv_grid_name,
                                    'type_name': branch[
                                        'branch'].type.to_frame().columns[0],
                                    'type_kind': branch['branch'].kind,
                                    'length': branch['branch'].length / 1e3,
                                    # length in km
                                    'U_n': branch['branch'].type['U_n'] / 1e3,
                                    # U_n in kV
                                    'I_max_th': branch['branch'].type[
                                        'I_max_th'],
                                    'R': branch['branch'].type['R'],
                                    'L': branch['branch'].type['L'],
                                    'C': branch['branch'].type['C'],
                                    'node1': '_'.join(
                                        [str(branch['adj_nodes'][
                                                 0].__class__.__name__), 'LV',
                                         str(lv_grid_id),
                                         str(branch['adj_nodes'][0].id_db)])
                                    if not isinstance(branch['adj_nodes'][0],
                                                      LVStationDing0) else '_'.join(
                                        [str(branch['adj_nodes'][
                                                 0].__class__.__name__), 'MV',
                                         str(mv_grid_id),
                                         str(branch['adj_nodes'][0].id_db)]),
                                    'node2': '_'.join(
                                        [str(branch['adj_nodes'][
                                                 1].__class__.__name__), 'LV',
                                         str(lv_grid_id),
                                         str(branch['adj_nodes'][1].id_db)])
                                    if not isinstance(branch['adj_nodes'][1],
                                                      LVStationDing0) else '_'.join(
                                        [str(branch['adj_nodes'][
                                                 1].__class__.__name__), 'MV',
                                         str(mv_grid_id),
                                         str(branch['adj_nodes'][1].id_db)]),
                                    'run_id': run_id,
                                    'geom': None
                                }

    lv_grid = pd.DataFrame.from_dict(lv_grid_dict, orient='index')
    lv_gen = pd.DataFrame.from_dict(lv_gen_dict, orient='index')
    lv_cd = pd.DataFrame.from_dict(lv_cd_dict, orient='index')
    lv_stations = pd.DataFrame.from_dict(lv_stations_dict, orient='index')
    mvlv_trafos = pd.DataFrame.from_dict(mvlv_trafos_dict, orient='index')
    lv_loads = pd.DataFrame.from_dict(lv_loads_dict, orient='index')
    mv_grid = pd.DataFrame.from_dict(mv_grid_dict, orient='index')
    mv_gen = pd.DataFrame.from_dict(mv_gen_dict, orient='index')
    mv_cb = pd.DataFrame.from_dict(mvcb_dict, orient='index')
    mv_cd = pd.DataFrame.from_dict(mv_cd_dict, orient='index')
    mv_stations = pd.DataFrame.from_dict(mv_stations_dict, orient='index')
    hvmv_trafos = pd.DataFrame.from_dict(hvmv_trafos_dict, orient='index')
    mv_loads = pd.DataFrame.from_dict(mv_loads_dict, orient='index')
    lines = pd.DataFrame.from_dict(lines_dict, orient='index')
    mvlv_mapping = pd.DataFrame.from_dict(mvlv_mapping_dict, orient='index')

    lines = lines[sorted(lines.columns.tolist())]

    return Network(
        run_id, metadata_json, lv_grid, lv_gen, lv_cd, lv_stations, mvlv_trafos, lv_loads, mv_grid, mv_gen, mv_cb,
        mv_cd, mv_stations, hvmv_trafos, mv_loads, lines, mvlv_mapping
    )
