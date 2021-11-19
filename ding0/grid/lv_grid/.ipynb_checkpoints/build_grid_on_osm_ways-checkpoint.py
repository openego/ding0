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
from shapely.geometry import LineString

from ding0.core.network import BranchDing0
from ding0.core.network.cable_distributors import LVCableDistributorDing0
from ding0.core.network.loads import LVLoadDing0
from ding0.tools import config as cfg_ding0

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

            # recursive call to check its target nodes also 
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

    return nodes_w_feeder  # need to return to get feeder id for edges


def get_feeder_cable_type_dict(lvgd, cable_lf, cos_phi_load, v_nom):
    """
    Get cable type for each feeder based on sum(peak_load) connected to each feeder.
    If cable_type doesn't cover load, set cable_type='NAYY 4x1x150' with intention
    to ensure grid integrity at a later point.
    return feeder_cable_type_dict: dict key: feederId, value: cable_type
    e.g. feeder_cable_type_dict = {1: 'NAYY 4x1x150', 2: 'NAYY 4x1x150', 3: 'NAYY 4x1x150'}
    """

    feeder_ids = list(set(lvgd.buildings.feeder.tolist()))

    feeder_cable_type_dict = {}
    for feeder_id in feeder_ids:
        I_max_load = lvgd.buildings.loc[lvgd.buildings.feeder == feeder_id].capacity.sum() / (3 ** 0.5 * v_nom) / cos_phi_load

        # determine suitable cable for this current
        suitable_cables_stub = lvgd.lv_grid.network.static_data['LV_cables'][
            (lvgd.lv_grid.network.static_data['LV_cables'][
                'I_max_th'] * cable_lf) > I_max_load]
        suitable_cables_stub = lvgd.lv_grid.network.static_data['LV_cables'][
            (lvgd.lv_grid.network.static_data['LV_cables'][
                'I_max_th'] * cable_lf) > I_max_load]
        if len(suitable_cables_stub):
            cable_type_stub = suitable_cables_stub.loc[suitable_cables_stub['I_max_th'].idxmin(), :]
        else:
            cable_type_stub = 'NAYY 4x1x150'   #this will probably not satisfy

        feeder_cable_type_dict[feeder_id] = cable_type_stub

    return feeder_cable_type_dict


def get_cable_type_by_load(lvgd, capacity, cable_lf, cos_phi_load, v_nom):
    """ get cable type for given capacity as param """
    I_max_load = capacity / (3 ** 0.5 * v_nom) / cos_phi_load

    # determine suitable cable for this current
    suitable_cables_stub = lvgd.lv_grid.network.static_data['LV_cables'][
        (lvgd.lv_grid.network.static_data['LV_cables'][
            'I_max_th'] * cable_lf) > I_max_load]
    suitable_cables_stub = lvgd.lv_grid.network.static_data['LV_cables'][
        (lvgd.lv_grid.network.static_data['LV_cables'][
            'I_max_th'] * cable_lf) > I_max_load]
    if len(suitable_cables_stub):
        cable_type_stub = suitable_cables_stub.loc[suitable_cables_stub['I_max_th'].idxmin(), :]
    else:
        cable_type_stub = 'NAYY 4x1x150'   #this will probably not satisfy.

    return cable_type_stub


def build_branches_on_osm_ways(lvgd):
    """Based on osm ways, the according grid topology for 
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
    node_feederId_dict = add_feeder_to_buildings(lvgd, shortest_tree_district_graph)

    # separate loads w. capacity: loads < 100 kW connected to grid
    # update with number residentials per feeder and feeder id
    lv_loads_grid = lvgd.buildings.loc[
        lvgd.buildings.capacity < get_config_osm('lv_threshold_capacity')]


    # loads 100 - 200 kW connected to lv station diretly
    lv_loads_to_station = lvgd.buildings.loc[
        (get_config_osm('lv_threshold_capacity') <= lvgd.buildings.capacity) &
        (lvgd.buildings.capacity < get_config_osm('mv_lv_threshold_capacity'))]

    # Add LVCableDistributorDing0 for nodes and branch for each edge to connect 
    # LVCableDistributorDing0 based on graph: shortest_tree_district_graph
    nodes, edges = ox.graph_to_gdfs(shortest_tree_district_graph, nodes=True, edges=True)

    # Get cable_type_dict based on sum(capacity) per feeder
    # e.g. cable_type_dict{0: NAYY 150, 1: NAYY 120, 2: NAYY 150, ...}
    feeder_cable_type_dict = get_feeder_cable_type_dict(lvgd, cable_lf, cos_phi_load, v_nom)

    # add lv_cable_dist for each node in graph. add branch for each edge in graph.
    # this is representing the street network in lvgd. there are no laods yet.
    # Loads will be connected afterwards.
    # Increment load_no for each edge in graph max(load_no) == len(edges)
    # edges are representing osm ways in street network, there are no loads
    # but each node u or v of edge represents a nearest node for osm buildings
    # mostly, to u or v buildings (with loads) are connected.
    # not every node has a load, e.g. nearest node in graph to osm buildings.
    load_no = 1  # init load_no=1
    added_cable_dist_list = []  # store id_db here
    for i, edge in edges.iterrows():  # starting index = 1
        u = i[0]  # bus0
        v = i[1]  # bus1
        u_is_station = False
        v_is_station = False
        # edge.geometry may be not in right direction, e.g.
        # u=Point(0,0) and v=Point(1,1) edge.geometry may
        # be LineString(Point(0,0), Point(1,1) BUT it may
        # be LineString(Point(1,1), Point(0,0) also
        # Thus, u_point, v_point = edge.geometry.boundary NOT working
        u_point = nodes.loc[u].geometry
        v_point = nodes.loc[v].geometry

        # get feederId as branch Id. ensuring it is not 0
        # due to stations as root node has feederId = 0
        if u == lvgd.lv_grid._station.osm_id_node:
            branch_no = node_feederId_dict.get(v)
            u_is_station = True
        if v == lvgd.lv_grid._station.osm_id_node:
            branch_no = node_feederId_dict.get(u)
            v_is_station = True
        else:  # just take u or v, both have same feeder/ branch Id 
            branch_no = node_feederId_dict.get(u)

        # get cable_type for branch
        cable_type = feeder_cable_type_dict.get(branch_no)

        # u and v are mandatory nodes in graph. they exist as
        # nearest nodes for buildings. buildings will be connected
        # in next step.
        # cable distributor to divert from main branch
        if u not in added_cable_dist_list:  # create new LVCableDistributorDing0
            lv_cable_dist_u = LVCableDistributorDing0(
                id_db=load_no,  # need to set an unqie identifier due to u may apper multiple times in edges
                grid=lvgd.lv_grid,
                branch_no=branch_no,
                load_no=load_no,  # has to be a unique id to avoid duplicates
                geo_data=u_point)
            # add lv_cable_dist to graph
            lvgd.lv_grid.add_cable_dist(lv_cable_dist_u)
            added_cable_dist_list.append(u)
            load_no += 1
        else:  # load from lvgd.lv_grid._cable_distributors
            lv_cable_dist_u = lvgd.lv_grid.get_cable_dist(u)

        # cable distributor to divert from main branch
        if v not in added_cable_dist_list:  # create new LVCableDistributorDing0
            lv_cable_dist_v = LVCableDistributorDing0(
                id_db=load_no,  # need to set an unqie identifier due to v may apper multiple times in edges
                grid=lvgd.lv_grid,
                branch_no=branch_no,
                load_no=load_no,  # has to be a unique id to avoid duplicates
                geo_data=v_point)
            # add lv_cable_dist_building to graph
            lvgd.lv_grid.add_cable_dist(lv_cable_dist_v)
            added_cable_dist_list.append(v)
            load_no += 1
        else:  # load from lvgd.lv_grid._cable_distributors
            lv_cable_dist_v = lvgd.lv_grid.get_cable_dist(v)

        # station is a own node. in case u or v of graph have the osm node id
        # of the station, connect node u or v to station to connect elements
        # (nodes and edges of graph) routed_graph to station
        if u_is_station:
            # case cable dist u <-> station
            lvgd.lv_grid.graph.add_edge(
                lvgd.lv_grid.station(),
                lv_cable_dist_u,
                branch=BranchDing0(
                    length=1,  # set 1 meter from node in graph to station
                    kind='cable',
                    grid=lvgd.lv_grid,
                    type=cable_type,
                    id_db='branch_{branch}_{load}'.format(
                        branch=branch_no,
                        load=load_no)))  # load has to be a unique id to avoid duplicates
                        # ,sector=sector_short)))
        elif v_is_station:
            # case cable dist v <-> station
            lvgd.lv_grid.graph.add_edge(
                lvgd.lv_grid.station(),
                lv_cable_dist_v,
                branch=BranchDing0(
                    length=1,  # set 1 meter from node in graph to station
                    kind='cable',
                    grid=lvgd.lv_grid,
                    type=cable_type,
                    id_db='branch_{branch}_{load}'.format(
                        branch=branch_no,
                        load=load_no)))  # load has to be a unique id to avoid duplicates
                        # , sector=sector_short)))

        # connect u and v no matter what to keep grid connected.
        lvgd.lv_grid.graph.add_edge(
            lv_cable_dist_u,  # connect u
            lv_cable_dist_v,  # with v
            branch=BranchDing0(
                length=edge.length,
                kind='cable',
                grid=lvgd.lv_grid,
                type=cable_type,
                geometry=edge.geometry,
                id_db='branch_{branch}_{load}'.format(
                    branch=branch_no,
                    load=load_no)))  # load has to be a unique id to avoid duplicates
                    # load is incremented for each edge in graph
                    # , sector=sector_short)))

    # PROCESSING LV LOADS W. CAPACITY < 100 KW
    # add LVCableDistributorDing0 for lv_loads_grid buildings and
    # em nearest nodes. add load to building.
    # build branch between each building and nearest node in ways
    # TODO: checkconnect load
    for u, row in lv_loads_grid.iterrows():

        # u is osm id building
        v = row.nn  # v is nearest osm id of nearest node in graph

        # get branch_no
        branch_no = node_feederId_dict.get(v)
        # get cable_type
        cable_type = get_cable_type_by_load(lvgd, row.capacity, cable_lf, cos_phi_load, v_nom)

        # cable distributor within building (to connect load+geno)
        if u not in added_cable_dist_list:  # create new LVCableDistributorDing0 for building
            lv_cable_dist_u = LVCableDistributorDing0(
                id_db=load_no,  # need to set an unqie identifier due to u may apper multiple times in edges
                grid=lvgd.lv_grid,
                branch_no=branch_no,
                load_no=load_no,  # has to be a unique id to avoid duplicates
                in_building=True,
                geo_data=row.raccordement_building)
            # add lv_cable_dist to graph
            lvgd.lv_grid.add_cable_dist(lv_cable_dist_u)
            added_cable_dist_list.append(u)
            load_no += 1
        else:  # load from lvgd.lv_grid._cable_distributors
            lv_cable_dist_u = lvgd.lv_grid.get_cable_dist(u)


        # cable distributor to divert from main branch
        if v not in added_cable_dist_list:  # create new LVCableDistributorDing0
            lv_cable_dist_v = LVCableDistributorDing0(
                id_db=load_no,  # need to set an unqie identifier due to u may apper multiple times in edges
                grid=lvgd.lv_grid,
                branch_no=branch_no,
                load_no=load_no,  # has to be a unique id to avoid duplicates
                geo_data=row.nn_coords)
            # add lv_cable_dist to graph
            lvgd.lv_grid.add_cable_dist(lv_cable_dist_v)
            added_cable_dist_list.append(v)
            load_no += 1
        else:  # load from lvgd.lv_grid._cable_distributors
            lv_cable_dist_v = lvgd.lv_grid.get_cable_dist(v)

        # create an instance of Ding0 LV load
        lv_load = LVLoadDing0(grid=lvgd.lv_grid,
                              branch_no=branch_no,
                              load_no=load_no,
                              peak_load=row.capacity  # in kW
                             )  # TODO: what here? isnt known yet.

        # add lv_load to graph
        lvgd.lv_grid.add_load(lv_load)

        # connect load
        lvgd.lv_grid.graph.add_edge(
            lv_cable_dist_u,  # is building
            lv_load,
            branch=BranchDing0(
                length=1,
                kind='cable',
                grid=lvgd.lv_grid,
                type=cable_type,
                id_db='branch_{branch}_{load}'.format(
                    branch=branch_no,
                    load=load_no
                    # , sector='HH'
                )))

        # connect u v
        lvgd.lv_grid.graph.add_edge(
            lv_cable_dist_u,  # is building
            lv_cable_dist_v,  # nearest node to building
            branch=BranchDing0(
                length=row.nn_dist,
                kind='cable',
                grid=lvgd.lv_grid,
                type=cable_type,
                geometry=LineString([row.raccordement_building, row.nn_coords]),
                id_db='branch_{branch}_{load}'.format(
                    branch=branch_no,
                    load=load_no
                    # , sector='HH'
                )))

    # adding cable_dist and loads and edges/Brahcnes for lv_loads_to_station
    for u, row in lv_loads_to_station.iterrows():

                    # u is osm id building
        v = row.nn  # v is nearest osm id of nearest node in graph

        # get branch_no
        branch_no = node_feederId_dict.get(v)
        # get cable_type
        cable_type = get_cable_type_by_load(lvgd, row.capacity, cable_lf, cos_phi_load, v_nom)

        route = nx.shortest_path(shortest_tree_district_graph, source=v, target=lvgd.lv_grid._station.osm_id_node, weight='length')

        point_list = []  # for LineString
        point_list.append(row.raccordement_building) # geometry of load
        for r in route:
            point_list.append(nodes.loc[r].geometry)
        ls = LineString(point_list)
        length = ls.length # dist from nearest node to station + building to nearest node

        # cable distributor within building (to connect load+geno)
        if u not in added_cable_dist_list:  # create new LVCableDistributorDing0 for building
            lv_cable_dist_u = LVCableDistributorDing0(
                id_db=load_no,  # need to set an unique identifier due to u may apper multiple times in edges
                grid=lvgd.lv_grid,
                branch_no=branch_no,
                load_no=load_no,  # has to be a unique id to avoid duplicates
                in_building=True,
                geo_data=row.raccordement_building)
            # add lv_cable_dist to graph
            lvgd.lv_grid.add_cable_dist(lv_cable_dist_u)
            added_cable_dist_list.append(u)
            load_no += 1
        else:  # load from lvgd.lv_grid._cable_distributors
            lv_cable_dist_u = lvgd.lv_grid.get_cable_dist(u)

        # cable distributor to divert from main branch
        if v not in added_cable_dist_list:  # create new LVCableDistributorDing0
            lv_cable_dist_v = LVCableDistributorDing0(
                id_db=load_no,  # need to set an unqie identifier due to u may apper multiple times in edges
                grid=lvgd.lv_grid,
                branch_no=branch_no,
                load_no=load_no,  # has to be a unique id to avoid duplicates
                geo_data=row.nn_coords)
            # add lv_cable_dist to graph
            lvgd.lv_grid.add_cable_dist(lv_cable_dist_v)
            added_cable_dist_list.append(v)
            load_no += 1
        else:  # load from lvgd.lv_grid._cable_distributors
            lv_cable_dist_v = lvgd.lv_grid.get_cable_dist(v)

        # create an instance of Ding0 LV load
        lv_load = LVLoadDing0(grid=lvgd.lv_grid,
                              branch_no=branch_no,
                              load_no=load_no,
                              peak_load=row.capacity  # in kW
                              )  # TODO: what here? isnt known yet.

        # add lv_load to graph
        lvgd.lv_grid.add_load(lv_load)

        # add cable load to bus
        lvgd.lv_grid.graph.add_edge(
            lv_cable_dist_u,  # is building
            lv_load,
            branch=BranchDing0(
                length=1,
                kind='cable',
                grid=lvgd.lv_grid,
                type=cable_type,
                id_db='branch_{branch}_{load}'.format(
                    branch=branch_no,
                    load=load_no
                    # , sector='HH'
                )))

        # connect building to station
        lvgd.lv_grid.graph.add_edge(
            lv_cable_dist_u,  # is building
            lvgd.lv_grid.station(),  # nearest node to building
            branch=BranchDing0(
                length=length,
                kind='cable',
                grid=lvgd.lv_grid,
                type=cable_type,
                geometry=ls,
                id_db='branch_{branch}_{load}'.format(
                    branch=branch_no,
                    load=load_no
                    # , sector='HH'
                )))

    return shortest_tree_district_graph