"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__ = "Reiner Lemoine Institut gGmbH"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__ = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__ = "nesnoj, gplssm"

import networkx as nx
import nxmetis
# G. Karypis, V. Kumar: A fast and high quality multilevel scheme for partitioning irregular graphs
# https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf

import osmnx as ox
import numpy as np
from shapely.geometry import LineString, Point

from ding0.core.network import BranchDing0
from ding0.core.network.cable_distributors import LVCableDistributorDing0
from ding0.core.network.loads import LVLoadDing0
from ding0.tools import config as cfg_ding0
from ding0.grid.lv_grid.routing import identify_street_loads
from ding0.grid.mv_grid.tools import get_shortest_path_shp_single_target, get_shortest_path_shp_multi_target

from ding0.config.config_lv_grids_osm import get_config_osm, get_load_profile_categories
from ding0.grid.lv_grid.parameterization import get_peak_load_for_residential
from ding0.grid.lv_grid.graph_processing import simplify_graph_adv

import logging

logger = logging.getLogger(__name__)


def get_routed_graph(
        graph, station_node, building_nodes, generator_nodes=None):
    '''
    routing via shortest path from station to loads and gens
    input graph representing all streets in lvgd and
        node of station in graph, all nodes of buildings.
        generator_nodes is none by default due to they are connected
        in a ding0 default way at a later point corresponding to load
    remove edges which are not mandatory and return routed graph
    '''
    routed_graph = graph.copy()

    edges_to_keep = []
    for n in building_nodes:
        route = nx.shortest_path(graph, n, station_node, weight='length')
        for i, r in enumerate(route[:-1]):
            edges_to_keep.append((r, route[i + 1]))
    edges_to_keep = list(set(edges_to_keep))
    edges_to_del = []
    for edge in list(graph.edges()):
        if edge not in edges_to_keep:
            if station_node not in edge:
                edges_to_del.append(edge)
    routed_graph.remove_edges_from(edges_to_del)
    logging.info('Routing shortest path for each node to station. '
                 f'{len(edges_to_del)} edges removed in graph to obtain '
                 'shortest path tree.')
    # due to edges are removed, nodes which are not connected anymore needs to be removed also
    # e.g. nodes which representing intersections on ways but not mandatory for shortest tree
    # graph to route loads to station need to be removed to dont add as ding0CableDist
    for loose_components in list(nx.weakly_connected_components(routed_graph)):
        if len(loose_components) == 1:  # there is only one not conencted node
            for node in loose_components:
                routed_graph.remove_node(node)

    return routed_graph


##############

def relocate_buildings_with_station_as_nn(full_graph, station_id, lv_loads_grid):
    station_nbs = [n for n in full_graph.neighbors(station_id)]

    if station_nbs:

        G = full_graph.copy()
        G = G.subgraph(station_nbs)

        station_as_nn = lv_loads_grid.loc[lv_loads_grid['nn'] == station_id]

        # if all or one building has station as nearest nodes, pass over relocation
        if len(station_as_nn) != len(lv_loads_grid) or len(station_as_nn) > 1:

            for building_node, row in station_as_nn.iterrows():
                node_id, nn_dist = ox.nearest_nodes(G, row["geometry"].x, row["geometry"].y, return_dist=True)
                lv_loads_grid.at[building_node, 'nn'] = node_id
                lv_loads_grid.at[building_node, 'nn_coords'] = Point(G.nodes[node_id]['x'], G.nodes[node_id]['y'])
                lv_loads_grid.at[building_node, 'nn_dist'] = nn_dist

    return lv_loads_grid


from ding0.grid.mv_grid.tools import get_edge_tuples_from_path


def get_shortest_path_tree(graph, station_node, building_nodes, generator_nodes=None):
    '''
    routing via the shortest path from station to loads and gens
    input graph representing all streets in lvgd and
        node of station in graph, all nodes of buildings.
        generator_nodes is none by default due to they are connected
        in a ding0 default way at a later point corresponding to load
    remove edges which are not mandatory and return shortest path tree
    '''

    G = graph.copy()

    if (len(building_nodes) == 1 and building_nodes[0] == station_node) or (len(building_nodes) == 0):

        sp_tree = G.subgraph([station_node])

    else:

        if len(G) > 1:

            edges = []

            for n in building_nodes:
                route = nx.shortest_path(G, n, station_node, weight='length')

                edge_path = get_edge_tuples_from_path(G, route)

                edges.extend(edge_path)

            edges_to_keep = set(edges)

            sp_tree = G.edge_subgraph(edges_to_keep)

        else:

            sp_tree = G

    return sp_tree


###############

def allocate_street_load_nodes(lv_loads_grid, shortest_tree_district_graph, station_id):
    # in order to do a size constrainted tree partioning
    # allocate capacity to street nodes (as integer)
    # allocate number_households to household nodes
    street_loads, household_loads = identify_street_loads(lv_loads_grid, shortest_tree_district_graph)
    street_loads_map = street_loads.to_dict()['capacity']
    household_loads_map = household_loads.to_dict()['number_households']

    for node in shortest_tree_district_graph.nodes:

        if node in street_loads_map.keys():
            shortest_tree_district_graph.nodes[node]['load'] = int(
                np.ceil(street_loads_map[node] * 1e3))  # as int * 1000
        else:
            shortest_tree_district_graph.nodes[node]['load'] = 0
        if node in household_loads_map:
            shortest_tree_district_graph.nodes[node]['n_households'] = household_loads_map[node]
        else:
            shortest_tree_district_graph.nodes[node]['n_households'] = 0

    # make sure substation's capacity is always zero
    shortest_tree_district_graph.nodes[station_id]['load'] = 0

    return shortest_tree_district_graph, street_loads, household_loads


def get_cable_type_by_load(lvgd, capacity, cable_lf, cos_phi_load, v_nom):
    """ get cable type for given capacity as param """
    I_max_load = capacity / (3 ** 0.5 * v_nom) / cos_phi_load

    # determine suitable cable for this current
    suitable_cables_stub = lvgd.lv_grid.network.static_data['LV_cables'][
        (lvgd.lv_grid.network.static_data['LV_cables'][
             'I_max_th'] * cable_lf) > I_max_load]
    if len(suitable_cables_stub):
        cable_type_stub = suitable_cables_stub.loc[suitable_cables_stub['I_max_th'].idxmin(), :]
    else:  # TODO: what to do if no cable is suitable for I_max_load, e.g. many loads connected to feeder?
        cable_type_stub = lvgd.lv_grid.network.static_data['LV_cables'].iloc[0]  # take strongest cable if no one suits

    return cable_type_stub


def get_n_feeder_mandatory(capacity, v_nom, cos_phi_load):
    I_max_allowed = 275  ##TODO: get value via config from standard cable type # refers to 150 mm2 standard type ref: dena Verteilnetzstudie
    I_max_load_at_feeder = capacity / (3 ** 0.5 * v_nom) / cos_phi_load
    return np.ceil(I_max_load_at_feeder / I_max_allowed)


def build_branches_on_osm_ways(lvgd):
    """
    Based on osm ways, the according grid topology for
    residential sector is determined and attached to the grid graph
    Parameters
    ----------
    lvgd : LVGridDistrictDing0
        Low-voltage grid district object
    """
    cable_lf = cfg_ding0.get('assumptions',
                             'load_factor_lv_cable_lc_normal')
    cos_phi_load = cfg_ding0.get('assumptions',
                                 'cos_phi_load')
    v_nom = cfg_ding0.get('assumptions', 'lv_nominal_voltage') / 1e3  # v_nom in kV

    # obtain shortest_tree_graph_district from graph_district
    # due to graph_district contains all osm ways in district
    # e.g. circles and it is not shortest paths
    # 1.) start with filtering for loads < 100kW.
    #     they will be connected in grid
    # 2.) get loads w. capacity >= 100kW and capacity < 200 kW
    #     and route em via shortest path to station. connect em
    #     with em own feeder/ cabel
    # 3.) each load with capacity >= 200kW and < 5.5 MW have its
    #     own station to which it is connected

    # get overall count of branches to set unique branch_no
    # mandatory to add loads between 100 - 200 kW
    # branch_count_sum = len(list(
    #    lvgd.lv_grid.graph.neighbors(lvgd.lv_grid.station())))

    # get diversity factor for non residential
    diversity_factor = get_config_osm('diversity_factor_not_residential')
    load_profile_categories = get_load_profile_categories()

    # station_id is node in graph which is root node
    station_id = lvgd.lv_grid._station.osm_id_node
    logger.info(f"LVGD: {lvgd}")
    # separate loads w. capacity: loads < 100 kW connected to grid
    lv_loads_grid = lvgd.buildings.loc[
        lvgd.buildings.capacity < get_config_osm('lv_threshold_capacity')]

    # full graph for routing 100 - 200 kW to station
    full_graph = lvgd.graph_district.copy()  # .to_undirected()
    full_graph_simple = nx.Graph(full_graph)

    # if possible change allocation of buildings to station and find another street node nearby
    lv_loads_grid = relocate_buildings_with_station_as_nn(full_graph, station_id, lv_loads_grid)

    # get graph for loads < 100 kW
    shortest_tree_district_graph = get_shortest_path_tree(
        full_graph,  # graph in district
        station_id,  # osmid of node representating station
        lv_loads_grid.nn.tolist())  # list of nodes connected to station


    # pre-process graph
    # allocate loads to street nodes
    shortest_tree_district_graph, street_loads, household_loads = allocate_street_load_nodes(lv_loads_grid,
                                                                                             shortest_tree_district_graph,
                                                                                             station_id)
    # simplify graph (get rid of cable dists for loads > 100 kW)
    shortest_tree_district_graph = simplify_graph_adv(shortest_tree_district_graph,
                                                      lv_loads_grid.nn.tolist() + [station_id])
    # split the graph into subtrees based on station's incident edges
    # if number of edges larger 1
    # prepare the shortest path tree graph
    g = shortest_tree_district_graph.copy()
    station_attr = g.nodes[station_id]

    # find subtrees in shortest path tree graph using substation's incident edges
    # TODO: Check if len(G) is bigger 1 / G.has_edges: lvgd graph with one node -> remove station leads to empty graph
    if len(g.nodes) > 1:  # only if there are at least 2 nodes, else keep working on full graph
        g.remove_node(station_id)
        graph_has_edges = True
    else:
        graph_has_edges = False

    nodelists = list(nx.weakly_connected_components(g))

    feederID = 0  # starting with feederId=0 for station. All leaving feeders will get an incremented feederId
    new_lvgd_peak_load_considering_simultaneity = 0
    feeder_graph_list = []  # append each feeder here

    # create subtrees from tree graph based on number of with station incident edges
    for nodelist in nodelists:

        subtree_graph = shortest_tree_district_graph.subgraph(list(nodelist) + [station_id]).to_undirected()

        cum_subtree_load = sum([subtree_graph.nodes[node]['load'] for node in subtree_graph.nodes]) / 1e3

        n_feeder = get_n_feeder_mandatory(cum_subtree_load, v_nom, cos_phi_load)

        # print(f"load {cum_subtree_load} needs n_feeder {n_feeder}")

        if n_feeder > 1:

            # Partition graph without station node
            G = nx.Graph(subtree_graph.subgraph(list(nodelist)).copy())

            # If the Graph has to be portioned to more than the number of nodes without the station node,
            # make every Node to a feeder, because METIS sometimes portion not contiguous.
            if len(G) - 1 > n_feeder:
                # transform edge length to int
                for edge in G.edges:
                    G.edges[edge]['length'] = int(np.ceil(G.edges[edge]['length']))

                (_, parts) = nxmetis.partition(G, int(n_feeder),
                                                 node_weight='load', edge_weight='length',
                                                 options=nxmetis.MetisOptions(contig=True))

            else:
                parts = [list(nodelist)]
                msg = f"In the LVGrid {lvgd} every grid_district.graph node becomes a feeder."
                logger.warning(msg)
                lvgd.network.message.append(msg)

            checked_contiguos_parts = []
            for cluster in parts:
                if len(cluster):
                    for nodes in nx.connected_components(subtree_graph.subgraph(cluster)):
                        checked_contiguos_parts.append(list(nodes))

            for cluster in checked_contiguos_parts:

                feederID += 1

                # get feeder graphs separately
                feeder_graph = subtree_graph.subgraph(cluster).copy()

                cum_feeder_graph_load = sum([feeder_graph.nodes[node]['load'] for node in feeder_graph.nodes]) / 1e3

                cable_type_stub = get_cable_type_by_load(lvgd, cum_feeder_graph_load, cable_lf, cos_phi_load, v_nom)
                for node in cluster:
                    feeder_graph.nodes[node]['feederID'] = feederID

                for edge in feeder_graph.edges:
                    feeder_graph.edges[edge]['feederID'] = feederID
                    feeder_graph.edges[edge]['cable_type_stub'] = cable_type_stub

                if not station_id in feeder_graph.nodes:
                    line_shp, line_length, line_path = get_shortest_path_shp_multi_target(subtree_graph, station_id,
                                                                                          cluster)
                    feeder_graph.add_edge(line_path[0], station_id, 0, geometry=line_shp, length=line_length,
                                          feederID=feederID, cable_type_stub=cable_type_stub)
                    feeder_graph.add_node(station_id, **station_attr)

                feeder_graph_list.append(feeder_graph)

        else:

            feederID += 1

            feeder_graph = subtree_graph.copy()

            cable_type_stub = get_cable_type_by_load(lvgd, cum_subtree_load, cable_lf, cos_phi_load, v_nom)

            for node in feeder_graph.nodes:
                feeder_graph.nodes[node]['feederID'] = feederID

            for edge in feeder_graph.edges:
                feeder_graph.edges[edge]['feederID'] = feederID
                feeder_graph.edges[edge]['cable_type_stub'] = cable_type_stub

            feeder_graph_list.append(feeder_graph)

    # Check if a graph of the feeder_graph_list has more than one connected components
    if max([len(list(nx.connected_components(_))) for _ in feeder_graph_list]) > 1:
        raise ValueError("Probably partitioned not contiguously.")
    G = nx.compose_all(feeder_graph_list)
    G.nodes[station_id]['feederID'] = 0

    station_node = G.nodes[station_id]
    # reset loads to zero due to adding buildings to graph with em real load
    # to avoid adding loads twice
    for node in G.nodes:
        G.nodes[node]['load'] = 0
        G.nodes[node]['ding0_node_type'] = "cable_distributor"

    G.nodes[station_id]['ding0_node_type'] = "station"

    # add loads < 100 kW to Graph G
    for building_node, row in lv_loads_grid.iterrows():
        nn_attr = G.nodes[row.nn]
        attr = {'x': row.geometry.x,
                'y': row.geometry.y,
                'node_type': 'non_synthetic',
                'ding0_node_type': "load",
                'cluster': nn_attr['cluster'],
                'load': row.capacity,
                'capacity': row.capacity,
                'residential_capacity': row.residential_capacity,
                'number_households': row.number_households,
                'cts_capacity': row.cts_capacity,
                'industrial_capacity': row.industrial_capacity,
                'feederID': nn_attr['feederID'],
                }

        cable_type_stub = get_cable_type_by_load(lvgd, row.capacity, cable_lf, cos_phi_load, v_nom)
        G.add_node(building_node, **attr)
        G.add_edge(building_node, row.nn, 0, geometry=LineString([row.geometry, row.nn_coords]),
                   length=row.nn_dist, feederID=nn_attr['feederID'], cable_type_stub=cable_type_stub)

    # loads 100 - 200 kW connected to lv station directly
    lv_loads_to_station = lvgd.buildings.loc[
        (get_config_osm('lv_threshold_capacity') <= lvgd.buildings.capacity) &
        (lvgd.buildings.capacity < get_config_osm('mv_lv_threshold_capacity'))]

    for building_node, row in lv_loads_to_station.iterrows():
        feederID += 1
        # todo: update capacity with load for residentials
        attr = {'x': row.geometry.x,
                'y': row.geometry.y,
                'node_type': 'non_synthetic',
                'ding0_node_type': "load",
                'capacity': row.capacity,
                'residential_capacity': row.residential_capacity,
                'number_households': row.number_households,
                'cts_capacity': row.cts_capacity,
                'industrial_capacity': row.industrial_capacity,
                'feederID': feederID,
                }
        G.add_node(building_node, **attr)
        # add new edge to street graph
        stub_line_shp = LineString([row.geometry, row.nn_coords])
        # add edge to full graph for shortest path calculation
        full_graph.add_edge(building_node, row.nn, geometry=stub_line_shp, length=stub_line_shp.length)
        # route singular cable
        line_shp, line_length, line_path = get_shortest_path_shp_single_target(full_graph, building_node, station_id,
                                                                               return_path=True, nodes_as_str=False)
        cable_type_stub = get_cable_type_by_load(lvgd, row.capacity, cable_lf, cos_phi_load, v_nom)
        G.add_edge(building_node, station_id, geometry=line_shp,
                   length=line_length, feederID=feederID, cable_type_stub=cable_type_stub)

    lvgd.graph_district = G  # update graph for district.

    def transform_graph_to_ding0_graph(G, grid):
        # New approach of transforming graph to ding0 objects graph by mltja
        # Add nodes
        cable_distributor_no = 0
        helper_no = 0
        load_no = 0
        relabel_nodes_dict = {}

        for node in G.nodes(data=True):
            node_name = node[0]
            node_data = node[1]
            node_coordinates = Point(node_data['x'], node_data['y'])

            if node_data["ding0_node_type"] == "cable_distributor":
                cable_distributor_no += 1
                cable_distributor = LVCableDistributorDing0(
                    grid=grid,
                    id_db=cable_distributor_no,
                    geo_data=node_coordinates,
                    in_building=False,
                    osm_id=node_name
                )
                grid.add_cable_dist(cable_distributor)
                relabel_nodes_dict[node_name] = cable_distributor
            elif node_data["ding0_node_type"] == "load":
                load_no += 1  # load_number in every component
                helper_no += 1
                # Helper components are used for better pypsa export
                # Add helper cable distributor
                cable_distributor = LVCableDistributorDing0(
                        grid=grid,
                        id_db=f"helper_{helper_no}",
                        helper_component=True,
                    )
                # Add helper cable
                branch = BranchDing0(
                    grid=grid,
                    id_db=f"helper_{helper_no}",
                    helper_component=True,
                )
                load = LVLoadDing0(
                    grid=grid,
                    id_db=load_no,
                    load_no=load_no,
                    peak_load=node_data["capacity"],
                    peak_load_residential=node_data["residential_capacity"],
                    number_households=node_data["number_households"],
                    peak_load_cts=node_data["cts_capacity"],
                    peak_load_industrial=node_data["industrial_capacity"],
                    geo_data=node_coordinates,
                    building_id=node_name,
                    type="conventional_load"
                )
                grid.add_cable_dist(cable_distributor)
                grid.graph.add_edge(
                    cable_distributor,
                    load,
                    branch=branch
                )
                grid.add_load(load)
                relabel_nodes_dict[node_name] = cable_distributor

            elif node_data["ding0_node_type"] == "station":
                # Station node is already in the graph
                relabel_nodes_dict[node_name] = grid._station
            else:
                raise ValueError(f"No ding0_node_type or the false: {node_data['ding0_node_type']}")

        G = nx.relabel_nodes(G, relabel_nodes_dict)

        # Add edges
        branch_no = 0
        for edge in G.edges(data=True):
            branch_no += 1

            edge_data = edge[2]

            branch = BranchDing0(
                grid=grid,
                id_db=branch_no,
                length=edge_data["length"],
                kind='cable',
                type=edge_data["cable_type_stub"],
                geometry=edge_data["geometry"],
                feeder=edge_data['feederID'],
            )
            grid.graph.add_edge(edge[0], edge[1], branch=branch)

    transform_graph_to_ding0_graph(G, lvgd.lv_grid)

            # direct load - station connection is deprecated
            #if u == station_id:
            #    print(v, load_v_capacity)
            #    load_v = lvgd.lv_grid.station()
            #else:

            # connect load
            load_v = lv_cable_dist_v
            # set in_building parameter
            load_v.in_building = True

            lvgd.lv_grid.graph.add_edge(
                load_v,
                lv_load,
                branch=BranchDing0(
                    length=1,
                    geometry=geometry,
                    kind='cable',
                    grid=lvgd.lv_grid,
                    type=cable_type,
                    feeder=node_u['feederID'],
                    id_db='branch_{branch}_{load}'.format(
                        branch=branch_no,
                        load=load_no)))

    # beacause of simultaneity calculation for residentials, capacity changed.
    # simultaneity calculation can happen only after ids of feeder are identified
    # thus, we can group loads by feederId, to which loads are connected and do calculation
    # set new peak_load for lvgd BUT CONSIDERING ONLY simultaneity loads per feeder
    lvgd.peak_load = new_lvgd_peak_load_considering_simultaneity

    #### in_building bugfix
    #for la in nd._mv_grid_districts[0]._lv_load_areas:
    #    for lvgd in la._lv_grid_districts:
    #        for node in lvgd.lv_grid._loads:
    #            branches = node.grid.graph_branches_from_node(node)
    #            if len(branches) == 1 and branches[0][1]['branch'].length == 1:
    #                cable_dist = branches[0][0]
    #                cable_dist.in_building = True
