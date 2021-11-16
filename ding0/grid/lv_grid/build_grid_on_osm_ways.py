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


import networkx as nx
import osmnx as ox

from ding0.core.network.cable_distributors import LVCableDistributorDing0
from ding0.core.network.loads import LVLoadDing0

from config.config_lv_grids_osm import get_config_osm


import logging
logger = logging.getLogger('ding0')


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
            edges_to_keep.append((r, route[i+1]))
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


def get_edges_leaving_feeder(graph, starting_node, feeder, seen, seen_w_feeder, lv_station_nn):
    """
    # get {edge: feederId} leaving the lv station.
    # depending on its number at later its possible to
    # add load residentials per feeder by formula of stetz:
    # vid parameterization.get_peak_load_for_residential()
    """
    # check if we already checked this node
    if starting_node not in seen:

        # if not checked yet, check it. get its edges and check its target nodes also
        # get edges from starting node. edges are touple (starting_node, target_node)
        new_target_nodes = graph.edges(starting_node)

        # mark starting node as visited to dont check it again.
        # save the feeder also that to count the number for each feeder 
        # respectively group_by(feeder) at a later point
        seen.add(starting_node)
        seen_w_feeder.update({starting_node: feeder})

        # for each node we received of starting node
        for nodes in new_target_nodes:

            # increment feeder as long as edge is leaving the lv station
            if starting_node == lv_station_nn:
                feeder += 1

            # get node connected via edge to starting node to follow 
            # itd edges until end to give these edges all same feeder id
            if nodes[1] == starting_node:
                node = nodes[0]
            else:
                node = nodes[1]

            # recusrive call to check its target nodes also 
            seen_w_feeder = get_edges_leaving_feeder(graph, node, feeder, seen, seen_w_feeder, lv_station_nn)

    return seen_w_feeder


def add_feeder_to_buildings(lvgd, graph):
    """
    for each feeder/ Strang leaving ons
    add id for feeder and the number of residentials connected to feeder
    """

    # key is node/ osmid, value is feeder
    feeder = 0  # set feeder = 0 due to its lv station what is root node

    # in set and dictioniary we gonna save seen nodes and its feeder.
    seen_nodes = set()
    nodes_w_feeder = {}

    # get nodes with the feeder they connected with. starting point is lv station.
    nodes_w_feeder = get_edges_leaving_feeder(graph, lvgd.lv_grid._station.osm_id_node,
                                              feeder, seen_nodes, nodes_w_feeder,
                                              lvgd.lv_grid._station.osm_id_node)

    nodes = lvgd.buildings
    nodes['feeder'] = 0
    # init n_residential_at_this_feeder with number_households
    # afterwards loads < 100 kW will be grouped per feeder and
    # n_residential_at_this_feeder will be updated but in this
    # way, loads >= 100 kW have value for n_residential_at_this_feeder
    nodes['n_residential_at_this_feeder'] = nodes['number_households']

    # due to buildings are not in graph, update em nearest nodes: loc[buildings.nn, 'feeder']
    for building_nearest_node, feeder_id in nodes_w_feeder.items():
        nodes.loc[nodes.nn == building_nearest_node, 'feeder'] = feeder_id

    # get number residentials per feeder
    nodes = nodes[nodes.capacity < get_config_osm('lv_threshold_capacity')]
    nodes.loc[nodes['category'] == 'residential', 'n_residential_at_this_feeder'] = nodes[nodes['category'] == 'residential'].groupby('feeder')['number_households'].transform('sum')


def add_lines(grid, graph, lv_vnom, cable_type='NAYY 4x120 SE'):
    """
    Parameters
    ----------
    grid : pypsa.Network
    graph : NetworkX.MultiDiGraph
        edges from graph to connect grid.
    Returns
    -------
    None.
    """
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    uvk_list = edges.index.tolist()
    edges[['bus0', 'bus1', 'k']] = uvk_list
    edges = edges[edges.k == 0]  # drop MultiDi-edges with same bus0, bus1
    edges['bus0'] = ['osm_id_' + str(e) for e in edges['bus0'].to_list()]
    edges['bus1'] = ['osm_id_' + str(e) for e in edges['bus1'].to_list()]
    edges['length'] /= 1000  # m to km

    index_line_name = ['street_cable_from_to_' + str(osmid)
                       for osmid in edges.index.tolist()]
    edges.index = index_line_name

    for i, line in edges.iterrows():
        coords = list(line.geometry.coords)
        coords = [list(coord) for coord in coords]

        if line.bus0 not in grid.buses.index:
            grid.add("Bus", line.bus0, v_nom=lv_vnom,
                     x=coords[0][0], y=coords[0][1])
            grid.buses.loc[line.bus0, 'coordinates'] = str([coords[0]])

        if line.bus1 not in grid.buses.index:
            grid.add("Bus", line.bus1, v_nom=lv_vnom,
                     x=coords[-1][0], y=coords[-1][1])
            grid.buses.loc[line.bus1, 'coordinates'] = str([coords[-1]])

        grid.add("Line", i, type=cable_type,
                 bus0=line.bus0,
                 bus1=line.bus1,
                 length=line.length)

        grid.lines.loc[i, 'coordinates'] = str(coords)
        grid.lines.loc[i, 'street_type'] = line.highway


def build_branches_on_osm_ways(lvgd):
    """Based on osm ways, the according grid topology for 
    residential sector is determined and attached to the grid graph

    Parameters
    ----------
    lvgd : LVGridDistrictDing0
        Low-voltage grid district object
    """
    lv_vnom = lvgd.lv_grid.v_level * 1e-3
    
    # TODO: get loads w 100 < capacity <= 200
    # filter shortest path tree for loads capacity <= 100

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

    
    # check if there is a graph. a graph only exists for lv grids 
    # where each load alone has a capacity < 200 kW
    if lvgd.graph_district:
        
        # separate loads w. capacity: loads < 100 kW connected to grid
        lv_loads_grid = lvgd.buildings.loc[
            lvgd.buildings.capacity < get_config_osm('lv_threshold_capacity')]

        # get graph for loads < 100 kW
        shortest_tree_district_graph = get_routed_graph(
            lvgd.graph_district,  # graph in district
            lvgd.lv_grid._station.osm_id_node,  # osmid of node representating station
            lv_loads_grid.nn.tolist())             # list of nodes connected to staion
        
        # make shortest_tree_district_graph to undirected due to 
        # dont need duplicated edegs u to v and v to u
        # furthermore mandatory for add_feeder_to_buildings
        # due to graph.edges(node) only wokring for undirected graph
        shortest_tree_district_graph = shortest_tree_district_graph.to_undirected()
        
        # add feeder id to each feeder leaving ons and
        # count residentials connected to each feeder
        # but only for lv_loads_grid due to higher loads 
        # are considered on them own
        add_feeder_to_buildings(lvgd, shortest_tree_district_graph)
        
        # separate loads w. capacity: loads < 100 kW connected to grid
        # update with number residentials per feeder and feeder id
        lv_loads_grid = lvgd.buildings.loc[
            lvgd.buildings.capacity < get_config_osm('lv_threshold_capacity')]


        # loads 100 - 200 kW connected to lv station diretly
        lv_loads_to_station = lvgd.buildings.loc[
            (get_config_osm('lv_threshold_capacity') <= lvgd.buildings.capacity) &
            (lvgd.buildings.capacity < get_config_osm('mv_lv_threshold_capacity'))]
        
        return lv_loads_grid, lv_loads_to_station, shortest_tree_district_graph

        # TODO: ADD LVCableDistributorDing0 FOR NODES IN GRAPH: shortest_tree_district_graph
        # ADD BRANCHES TO CONNECT LVCableDistributorDing0

        # PROCESSING LV LOADS W. CAPACITY < 100 KW
        # add LVCableDistributorDing0 for lv_loads_grid buildings and
        # em nearest nodes. add load to building.
        # build branch between each building and nearest node in ways
        for i, row in lv_loads_grid.iterrows():

            # cable distributor to divert from main branch
            lv_cable_dist = LVCableDistributorDing0(
                grid=lvgd.lv_grid,
                branch_no=i,
                load_no=i,
                geo_data=row.nn_coords)
            # add lv_cable_dist to graph
            lvgd.lv_grid.add_cable_dist(lv_cable_dist)

            # cable distributor within building (to connect load+geno)
            lv_cable_dist_building = LVCableDistributorDing0(
                grid=lvgd.lv_grid,
                branch_no=i,
                load_no=i,
                in_building=True,
                geo_data=row.raccordement_building)
            # add lv_cable_dist_building to graph
            lvgd.lv_grid.add_cable_dist(lv_cable_dist_building)

            # create an instance of Ding0 LV load
            lv_load = LVLoadDing0(grid=lvgd.lv_grid,
                                  branch_no=i,
                                  load_no=i,
                                  peak_load=row.capacity,  # in kW
                                  consumption=row.capacity * 1000)  # TODO: what here? isnt known yet.

            # add lv_load to graph
            lvgd.lv_grid.add_load(lv_load)

            # connect load


            # TODO:
            # route each load in lv_loads_to_station to get edges to add as cables
            # for i, row in lv_loads_to_station.iterrows():
            # get shortest path from row.nn to statio_osm_id
            # get its coords and total length
            # add 
            # add it as ding0Branch 

    # there is no graph, thus, each load has a capacity >= 200 kW
    # there exist the node for the load and the next nearest osm node
    # the next osm node is located on a osm way to be able to route
    # the station along streets
    else:
        pass

        # get cables to add
        
        #if .15 <= capacity <= .2:
        #    cable_type = 'NAYY 4x120 SE'
        #    cables = [cable_type, cable_type]
        #elif .1 <= capacity < .15:
        #    cable_type = 'NAYY 4x150 SE'
        #    cables = [cable_type]
        #elif .5 <= capacity < .1:
        #    cable_type = 'NAYY 4x120 SE'
        #    cables = [cable_type]
        #elif capacity < .5:
        #    cable_type = 'NAYY 4x50 SE'
        #    cables = [cable_type]
        #else:
        #    logger.warning(f'capacity {building.capacity} isnt covered!')

        # add mandatory cable
        #for i_cable, cable in enumerate(cables):
        #    grid.add("Line", f"{capacity_lvl}_building_connecting_cable_" +
        #             str(i) + "_" + str(i_cable), type=cable,
        #             bus0="osm_id_" + str(i),
        #             bus1="osm_id_" + str(building.nn),
        #             length=building.nn_dist * 1e-3)
        #    grid.lines.loc[f"{capacity_lvl}_building_connecting_cable_" +
        #                   str(i) + "_" + str(i_cable), 'coordinates'] = \
        #        str([[x_b, y_b], [x_nn, y_nn]])
        #    grid.lines.loc[f"{capacity_lvl}_building_connecting_cable_" +
        #                   str(i) + "_" + str(i_cable),
        #                   'street_type'] = 'building_to_street'
                
    # build branches for each edge in shortest_tree_district_graph
    # add_way_lines(grid, routed_graph, lv_vnom)
    
    
    # TODO: what to do with average_load, average_consumption?
    # especially average_consumption not known yet.
    # average_load = lvgd.peak_load_residential / houses_connected
    # average_consumption = lvgd.sector_consumption_residential / houses_connected
    # e.g.  'peak_load_residential': 1.39742806193774, 'sector_consumption_residential': 6538.59284080918;