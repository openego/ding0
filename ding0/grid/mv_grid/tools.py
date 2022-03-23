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


from ding0.core.network.stations import LVStationDing0
from ding0.core.network.loads import MVLoadDing0
from ding0.core import MVCableDistributorDing0
from ding0.core.network import CableDistributorDing0, GeneratorDing0, LoadDing0
from ding0.tools.geo import calc_geo_centre_point
from ding0.tools import config as cfg_ding0
import logging


logger = logging.getLogger('ding0')


def set_circuit_breakers(mv_grid, mode='load', debug=False):
    """ Calculates the optimal position of a circuit breaker at lv stations (if existing)
    on all routes of mv_grid, adds and connects them to graph.
    
    Args
    ----
    mv_grid: MVGridDing0
       MV grid instance
    debug: bool, defaults to False
       If True, information is printed during process
    

    Note
    -----
    According to planning principles of MV grids, a MV ring is run as two strings (half-rings) separated by a
    circuit breaker which is open at normal operation [#]_, [#]_.
    Assuming a ring (route which is connected to the root node at either sides), the optimal position of a circuit
    breaker is defined as the position (virtual cable) between two nodes where the conveyed current is minimal on
    the route. Instead of the peak current, the peak load is used here (assuming a constant voltage).
    
    The circuit breaker will be installed to a LV station, unless none
    exists in a ring. In this case, a node of arbitrary type is chosen for the
    location of the switch disconnecter.
    
    If a ring is dominated by loads (peak load > peak capacity of generators), only loads are used for determining
    the location of circuit breaker. If generators are prevailing (peak load < peak capacity of generators),
    only generator capacities are considered for relocation.

    The core of this function (calculation of the optimal circuit breaker position) is the same as in
    ding0.grid.mv_grid.models.Route.calc_circuit_breaker_position but here it is
    1. applied to a different data type (NetworkX Graph) and it
    2. adds circuit breakers to all rings.

    The re-location of circuit breakers is necessary because the original position (calculated during routing with
    method mentioned above) shifts during the connection of satellites and therefore it is no longer valid.

    References
    ----------
    .. [#] X. Tao, "Automatisierte Grundsatzplanung von Mittelspannungsnetzen", Dissertation, 2006
    .. [#] FGH e.V.: "Technischer Bericht 302: Ein Werkzeug zur Optimierung der Störungsbeseitigung
        für Planung und Betrieb von Mittelspannungsnetzen", Tech. rep., 2008

    """

    def relocate_circuit_breaker():
        """
        Moves circuit breaker to different position in ring.

            
        Note
        -----
        Branch of circuit breaker should be set to None in advance. 
        So far only useful to relocate all circuit breakers in a grid as the 
        position of the inserted circuit breaker is not checked beforehand. If
        used for single circuit breakers make sure to insert matching ring and
        circuit breaker.
        """
        node_cb = ring[position]
        # check if node is last node of ring
        if position < len(ring):
            # check which branch to disconnect by determining load difference
            # of neighboring nodes
            diff2 = abs(sum(node_peak_data[0:position+1]) -
                        sum(node_peak_data[position+1:len(node_peak_data)]))
            if diff2 < diff_min:
                node2 = ring[position+1]
            else:
                node2 = ring[position-1]
        else:
            node2 = ring[position-1]

        circ_breaker.branch = mv_grid.graph.adj[node_cb][node2]['branch']
        circ_breaker.branch_nodes = (node_cb, node2)
        circ_breaker.switch_node = node_cb
        circ_breaker.branch.circuit_breaker = circ_breaker
        line_shp = circ_breaker.branch.geometry
        circ_breaker.geo_data = cut_line_by_distance(line_shp, 0.5, normalized=True)[0] # PAUL new


    # get power factor for loads and generators
    cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
    cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
    
    # "disconnect" circuit breakers from branches
    for cb in mv_grid.circuit_breakers():
        cb.branch.circuit_breaker = None

    # iterate over all rings and circuit breakers
    for ring, circ_breaker in zip(mv_grid.rings_nodes(include_root_node=False), 
                                  mv_grid.circuit_breakers()):

        nodes_peak_load = []
        nodes_peak_generation = []

        # iterate over all nodes of ring
        for node in ring:

            # node is LV station -> get peak load and peak generation
            if isinstance(node, LVStationDing0):
                nodes_peak_load.append(node.peak_load / cos_phi_load)
                nodes_peak_generation.append(
                    node.peak_generation / cos_phi_feedin)
                
            # node is LV station -> get peak load and peak generation
            elif isinstance(node, MVLoadDing0):
                nodes_peak_load.append(node.peak_load / cos_phi_load)
                #nodes_peak_generation.append(
                #    node.peak_generation / cos_phi_feedin) # TODO include feedin for mv loads

            # node is cable distributor -> get all connected nodes of subtree using graph_nodes_from_subtree()
            elif isinstance(node, CableDistributorDing0):
                nodes_subtree = mv_grid.graph_nodes_from_subtree(node)
                nodes_subtree_peak_load = 0
                nodes_subtree_peak_generation = 0

                for node_subtree in nodes_subtree:

                    # node is LV station -> get peak load and peak generation
                    if isinstance(node_subtree, LVStationDing0):
                        nodes_subtree_peak_load += node_subtree.peak_load / \
                                                   cos_phi_load
                        nodes_subtree_peak_generation += node_subtree.peak_generation / \
                                                         cos_phi_feedin

                    # node is LV station -> get peak load and peak generation
                    if isinstance(node_subtree, GeneratorDing0):
                        nodes_subtree_peak_generation += node_subtree.capacity / \
                                                         cos_phi_feedin

                nodes_peak_load.append(nodes_subtree_peak_load)
                nodes_peak_generation.append(nodes_subtree_peak_generation)

            else:
                raise ValueError('Ring node has got invalid type.')

        if mode == 'load':
            node_peak_data = nodes_peak_load
        elif mode == 'generation':
            node_peak_data = nodes_peak_generation
        elif mode == 'loadgen':
            # is ring dominated by load or generation?
            # (check if there's more load than generation in ring or vice versa)
            if sum(nodes_peak_load) > sum(nodes_peak_generation):
                node_peak_data = nodes_peak_load
            else:
                node_peak_data = nodes_peak_generation
        else:
            raise ValueError('parameter \'mode\' is invalid!')

        # calc optimal circuit breaker position
        # Set start value for difference in ring halfs
        diff_min = 10e9

        # if none of the nodes is of the type LVStation, a switch
        # disconnecter will be installed anyways.
        if any([isinstance(n, LVStationDing0) for n in ring]):
            has_lv_station = True
        else:
            has_lv_station = False
            logging.debug("Ring {} does not have a LV station. "
                          "Switch disconnecter is installed at arbitrary "
                          "node.".format(ring))

        # check where difference of demand/generation in two half-rings is minimal
        for ctr in range(len(node_peak_data)):
            # check if node that owns the switch disconnector is of type
            # LVStation
            if isinstance(ring[ctr], LVStationDing0) or not has_lv_station:
                # split route and calc demand difference
                route_data_part1 = sum(node_peak_data[0:ctr])
                route_data_part2 = sum(node_peak_data[ctr:len(node_peak_data)])
                diff = abs(route_data_part1 - route_data_part2)
    
                # equality has to be respected, otherwise comparison stops when demand/generation=0
                if diff <= diff_min:
                    diff_min = diff
                    position = ctr
                else:
                    break

        # relocate circuit breaker
        relocate_circuit_breaker()

        if debug:
            logger.debug('Ring: {}'.format(ring))
            logger.debug('Circuit breaker {0} was relocated to edge {1} '
                  '(position on route={2})'.format(
                    circ_breaker, repr(circ_breaker.branch), position)
                )
            logger.debug('Peak load sum: {}'.format(sum(nodes_peak_load)))
            logger.debug('Peak loads: {}'.format(nodes_peak_load))

from shapely.ops import linemerge
from shapely.geometry import Point, LineString
from ding0.grid.lv_grid.graph_processing import simplify_graph_adv, remove_unloaded_deadends, remove_parallels_and_loops
import networkx as nx
import pandas as pd


def get_edge_tuples_from_path(G, path_list):
    
    if G is None:
    
        edge_path = [(path_list[i], path_list[i+1]) for i in range(len(path_list)-1)]
        
    else:

        edge_path = [(path_list[i], path_list[i+1], next(iter(G.get_edge_data(path_list[i],path_list[i+1]).keys()))) for i in range(len(path_list)-1)]
    
    return edge_path


def get_line_shp_from_shortest_path(osm_graph, node1, node2, return_path=False):
    
    sp = nx.shortest_path(osm_graph, str(node1), str(node2), weight='length')
    edge_path = get_edge_tuples_from_path(osm_graph, sp)
    line_shp = linemerge([osm_graph.edges[edge]['geometry'] for edge in edge_path])
    
    if line_shp.is_empty: # in case of route_length == 1
        line_shp = osm_graph.edges[str(node1), str(node2), 0]['geometry']
    
    line_length = line_shp.length
    
    if line_length == 0:
        line_length = 1
        logger.warning('Geo distance is zero, check objects\' positions. '
                       'Distance is set to 1m')

    if return_path:
        return line_shp, line_length, sp
    else:
        return line_shp, line_length


def cut_line_by_distance(line, distance, normalized=True):
    # inspried from shapely

    if normalized:
        distance = distance*line.length
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [Point(line.coords[0]), line]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            cp = line.interpolate(distance)
            return [Point(cp),
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [Point(cp),
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]
        

def reduce_graph_for_dist_matrix_calc(graph, nodes_to_keep):
    
    '''
    simplifies graph, until number of graph nodes between two iterations
    is almost the same (ratio=0.95), keeps nodes_of_interest in graph
    during simplification
    returns graph

    '''
    
    post_graph_no, pre_graph_no = 0, len(graph)
    
    while post_graph_no/pre_graph_no <= 0.95: 
        pre_graph_no = len(graph)
        graph = simplify_graph_adv(graph, nodes_to_keep)
        graph = remove_parallels_and_loops(graph)
        graph = remove_unloaded_deadends(graph, nodes_to_keep)
        post_graph_no = len(graph)
    
    return graph


def calc_street_dist_matrix(G, matrix_node_list):

    node_list = matrix_node_list.copy()
    node_list.extend(list(set(G.nodes()) - set(matrix_node_list)))

    dm_floyd = nx.floyd_warshall_numpy(G, nodelist=node_list, weight='length')
    df_floyd = pd.DataFrame(dm_floyd)
    #reduce matrix
    df_floyd = df_floyd.iloc[:len(matrix_node_list),:len(matrix_node_list)]
    df_floyd = df_floyd.div(1000) # unit is km
    df_floyd = df_floyd.round(3) # accuracy in m
    # pass osmid nodes as indices
    df_floyd = df_floyd.set_index(pd.Series(matrix_node_list))
    df_floyd.columns = matrix_node_list
    # transform matrix to dict
    matrix_dict = df_floyd.to_dict('dict')
    
    return matrix_dict


def conn_ding0_obj_to_osm_graph(osm_graph, ding0_obj, search_shp=None):

    '''
    connects ding0 object with shortest euclidic distance to existing osm graph node
    if search_shp is not None but shapely Point, those coordinates are used for
    finding nearest graph node
    create new node and edges
    osm_graph is of type MultiDiGraph
    '''

    # search_shp is None, take ding0 objects' shp
    if not search_shp:
        x,y = ding0_obj.geo_data.x, ding0_obj.geo_data.y
    # search_shp is passed
    else:
        x,y = search_shp.x, search_shp.y

    nn = ox.distance.nearest_nodes(osm_graph, x, y, return_dist=False)
    line_shp = LineString([(osm_graph.nodes[nn]['x'], osm_graph.nodes[nn]['y']),
                           (ding0_obj.geo_data.x, ding0_obj.geo_data.y)])
    osm_graph.add_node(str(ding0_obj), x=ding0_obj.geo_data.x, y=ding0_obj.geo_data.y, node_type='synthetic')
    osm_graph.add_edge(nn, str(ding0_obj), geometry=line_shp, length=line_shp.length) #missing: highway, osmid
    osm_graph.add_edge(str(ding0_obj), nn, geometry=line_shp, length=line_shp.length) #missing: highway, osmid

    return osm_graph



# graph processing

def get_core_graph(G):
    C = ox.utils_graph.get_digraph(G, weight='length')
    C.remove_edges_from(nx.selfloop_edges(C))
    C = nx.k_core(C, k=3, core_number=None)
    C = nx.MultiDiGraph(G.subgraph(C.nodes))

    return C


def get_stub_graph(osm_graph_red, core_graph):
    stub_edges = [n for n in osm_graph_red.edges if n in core_graph.edges]
    R = osm_graph_red.copy()
    R.remove_edges_from(stub_edges)
    R.remove_nodes_from(list(nx.isolates(R)))

    return R

def split_graph_by_core(street_graph, depot):

    G = ox.utils_graph.get_digraph(street_graph, weight='length')
    G.remove_edges_from(nx.selfloop_edges(G))

    core_graph = nx.k_core(G, k=3, core_number=None)
    core_graph = nx.MultiDiGraph(G.subgraph(core_graph.nodes))

    core_edges = [n for n in street_graph.edges if n in core_graph.edges]
    stub_graph = street_graph.copy()
    stub_graph.remove_edges_from(core_edges)
    stub_graph.remove_nodes_from(list(nx.isolates(stub_graph)))

    # make sure mv_station node is connected to core graph
    core_graph = conn_ding0_obj_to_osm_graph(core_graph, depot)
    if stub_graph.has_node(str(depot)):
        stub_graph.remove_node(str(depot))

    return core_graph, stub_graph


def update_graphs(core_graph, stub_graph, nodes_to_switch):

    switch_graph = stub_graph.subgraph(nodes_to_switch)
    core_graph = nx.compose(core_graph, switch_graph)
    edges_to_swith = list(switch_graph.edges)

    stub_graph.remove_edges_from(edges_to_swith)
    stub_graph.remove_nodes_from(list(nx.isolates(stub_graph)))
    stub_graph = nx.MultiDiGraph(stub_graph) # vielleicht nicht notwegnig, auch eher DiGraph
    
    return core_graph, stub_graph


# stub processing


# stores node data of stubs
# comp being all nodes part of stub
# root...root_node, load...nodes with demand, dist...nodes have function as cable_dist

def create_stub_dict(stub_graph, root_nodes, node_list):
    
    node_set = set(node_list.keys())
    
    stub_dict = {}

    if nx.number_weakly_connected_components(stub_graph):

        for i, comp in enumerate(nx.weakly_connected_components(stub_graph)):

            root_node_set = comp & root_nodes
            load_node_set = comp & node_set - root_node_set
            cable_dist_set = comp - root_node_set - load_node_set

            stub_dict[i] = {}
            stub_dict[i]['comp'] = comp
            stub_dict[i]['root'] = list(root_node_set)[0] #next(iter())
            stub_dict[i]['load'] = {node: int(node_list[node].peak_load / cfg_ding0.get('assumptions', 'cos_phi_load'))
                                    for node in load_node_set} #{node: demand}
            stub_dict[i]['dist'] = cable_dist_set
            
    return stub_dict


def check_stub_criterion(stub_dict, stub_graph):

    stubs_to_del_dict = {}
    mod_stubs_list = []

    for key, stub_data in stub_dict.items():

        if not stub_data['load']: # delete stubs without load

            stubs_to_del_dict[key] = stub_data['comp']

        cum_stub_load = sum(stub_data['load'].values())

        if cum_stub_load > cfg_ding0.get('mv_connect', 'load_area_sat_string_load_threshold'):

            if len(stub_data['load']) == 1:

                # save key and nodes in dict
                stubs_to_del_dict[key] = stub_data['comp']

            else: # stub is tree or stub with more than one edge

                #build directed tree subgraph from stub_graph for analysis
                root = stub_data['root']
                tree = stub_graph.subgraph(stub_data['comp'])
                tree = nx.dfs_tree(tree, root) #directed from root
                deg_root = tree.degree[root]
                deg_tree_max = max(dict(tree.degree).values())

                if deg_tree_max <= 2 and deg_root == 1: #tree is stub with more than one edge

                    load_nodes = list(tree)[1:]

                    for n in load_nodes:
                        cum_load = sum([stub_data['load'][n] for n in load_nodes])
                        if cum_load <= cfg_ding0.get('mv_connect', 'load_area_sat_string_load_threshold'):
                            comp = [root] + load_nodes
                            mod_stubs_list.append(comp)
                            break
                        else:
                            root = load_nodes[0]
                            load_nodes = load_nodes[1:]

                else: # tree is tree, get stubs from tree

                    for node in list(tree.nodes)[1:]:

                        subtree = nx.dfs_tree(tree, node)
                        deg_subtree_max = max(dict(subtree.degree(list(subtree))).values())

                        root = list(tree.predecessors(node))[0]
                        deg_root = len(list((tree.successors(root)))) #tree.degree[root]

                        #identify stubs by degree
                        if deg_root >= 2 and deg_subtree_max <= 1:

                            load_nodes = list(subtree)

                            if not all(n in stub_data['load'] for n in load_nodes):
                                root = load_nodes[0]
                                load_nodes = [load_nodes[1:] for n in load_nodes if n not in stub_data['load']][0]

                            for n in load_nodes:

                                cum_load = sum([stub_data['load'][n] for n in load_nodes])
                                if cum_load <= cfg_ding0.get('mv_connect', 'load_area_sat_string_load_threshold'):
                                    comp = [root] + load_nodes
                                    mod_stubs_list.append(comp)
                                    break
                                else:
                                    root = load_nodes[0]
                                    load_nodes = load_nodes[1:]

                #remove old comp not satisfying stub criterion
                stubs_to_del_dict[key] = stub_data['comp']

    for key in stubs_to_del_dict.keys(): del stub_dict[key]

    old_stub_nodes = {x for v in stubs_to_del_dict.values() for x in v}
    mod_stub_roots = {x[0] for x in mod_stubs_list}
    mod_stub_loads = {x for v in mod_stubs_list for x in v} - mod_stub_roots

    nodes_to_switch = old_stub_nodes - mod_stub_loads


    return mod_stubs_list, nodes_to_switch


def update_stub_dict(stub_dict, mod_stubs_list, node_list):

    for comp in mod_stubs_list:

        root = comp[0]
        load_node_set = comp[1:]

        i = max(stub_dict.keys()) + 1 

        stub_dict[i] = {}
        stub_dict[i]['load'] = {node: int(node_list[node].peak_load / cfg_ding0.get('assumptions', 'cos_phi_load'))
                                for node in load_node_set} #{node: demand}
        stub_dict[i]['root'] = root
        stub_dict[i]['comp'] = set(comp)
        stub_dict[i]['dist'] = set()

        i+=1 
    
    return stub_dict


import geopandas as gpd
from requests import get
from shapely import wkb, wkt
import osmnx as ox

def get_mvgd_ids_for_city(osm_city_name):
    
    version='v0.4.5'

    mvgds = get('https://openenergy-platform.org/api/v0/schema/grid/tables/ego_dp_mv_griddistrict/rows/?where=version='+version,)#+'&where=subst_id='+subst_id,)
    gdf_mvgds = gpd.GeoDataFrame(mvgds.json())
    gdf_mvgds['geom'] = gdf_mvgds['geom'].apply(lambda x: wkb.loads(x, hex=True))
    gdf_mvgds = gdf_mvgds.set_geometry('geom')
    gdf_mvgds.crs = 'epsg:3035'

    city = ox.geocode_to_gdf(osm_city_name)
    gdf_city = ox.project_gdf(city, to_crs='epsg:3035')

    gdf_mvgds['poly_inside']  = gdf_mvgds.geom.apply(lambda x: gdf_city.geometry[0].contains(x.centroid))
    gdf_mvgds_in_city = gdf_mvgds.loc[gdf_mvgds['poly_inside'] == True]

    return gdf_mvgds_in_city['subst_id'].to_list()

### functions for cable routing in settlements

def relocate_cable_dists_settle(load_area, branches):

    # returns list of relocated cable dists in load area

    cable_dist_settle = set()

    # find cable distributors inside load area
    for branch in branches:
        cd_in_load_area = [node for node in branch['adj_nodes'] if isinstance(node, MVCableDistributorDing0) \
                           and node.geo_data.intersects(load_area.geo_area)]

        cable_dist_settle.update(cd_in_load_area)

    # if found, update cable distributors' coordinates by finding
    # nearest node's position in street graph, update branch geoms as well
    if cable_dist_settle:

        for cd in cable_dist_settle:

            G = load_area.load_area_graph.copy()
            cd_shp = cd.geo_data

            # osm_ids allocated to ding0 supply nodes are unique, can't be used for cable dists
            locked_osm_ids = set([mv_load.osmid_building for mv_load in load_area._mv_loads] +
                                 [lvgd.lv_grid._station.osm_id_node for lvgd in load_area._lv_grid_districts])
            G.remove_nodes_from(locked_osm_ids)

            osm_id = ox.nearest_nodes(G, cd_shp.x, cd_shp.y, return_dist=False)
            osm_node_shp = Point([(G.nodes[osm_id]['x'], G.nodes[osm_id]['y'])])

            # update cable distributors position
            cd.geo_data = osm_node_shp
            cd.osm_id_node = osm_id

            # update adjacent branch geoms
            branches_upd = load_area.mv_grid_district.mv_grid.graph_branches_from_node(cd)

            for node, branch in branches_upd:

                line_shp_upd = LineString([node.geo_data, cd.geo_data])

                branch['branch'].geometry = line_shp_upd

                circ_breaker = branch['branch'].circuit_breaker
                if circ_breaker is not None:
                    circ_breaker.geo_data = cut_line_by_distance(line_shp_upd, 0.5, normalized=True)[0]

    return list(cable_dist_settle)

import networkx as nx

def relabel_graph_nodes(load_area, cable_dists):
    # get ding0 nodes
    lv_stations = [lvgd.lv_grid._station for lvgd in load_area._lv_grid_districts]
    mv_loads = [mv_load for mv_load in load_area._mv_loads]

    # create mapping dicts: get nodes as mapping dicts {osmid: str(ding0_name)}
    lv_stations_map = {lv_station.osm_id_node: str(lv_station) for lv_station in
                       lv_stations}  # TODO: make osmid access consistent
    mv_loads_map = {mv_load.osmid_building: str(mv_load) for mv_load in mv_loads}

    if cable_dists:
        cable_dists_map = {cd.osm_id_node: str(cd) for cd in cable_dists}
        ding0_nodes_map = {**lv_stations_map, **mv_loads_map, **cable_dists_map}
    else:
        ding0_nodes_map = {**lv_stations_map, **mv_loads_map}

    # update
    street_graph = load_area.load_area_graph
    osmids_to_str_map = {osmid: str(osmid) for osmid in street_graph.nodes}
    street_graph = nx.relabel_nodes(street_graph, {**osmids_to_str_map, **ding0_nodes_map})

    return street_graph, ding0_nodes_map

from shapely.geometry import MultiPoint, LinearRing
from shapely.ops import nearest_points
import numpy as np
from ding0.core.network.stations import MVStationDing0

def update_branch_shps_settle(load_area, branches, street_graph):
    # update branches routinh in settlement areas
    # if both endpoints in la update whole shape
    # if one endpoint in la update part of shape
    # if no endpoints in la, do nothing (maybe check if part of linestring crosses same/other la

    path_passed_osmids = {}

    for branch in branches:

        mv_station = [node for node in branch['adj_nodes'] if isinstance(node, MVStationDing0)]
        G = street_graph

        # add node to street graph, if not element of it
        if mv_station:
            if not G.has_node(str(mv_station[0])):
                G = conn_ding0_obj_to_osm_graph(G, mv_station[0])

        endpoints = [load_area.geo_area.intersects(node.geo_data) for node in branch['adj_nodes']]

        # both branch endpoints are within load area - update whole branch geometry
        if all(endpoints):

            node1, node2 = branch['adj_nodes']
            line_shp, line_length, path = get_line_shp_from_shortest_path(G, node1, node2, return_path=True)

            branch['branch'].geometry = line_shp
            branch['branch'].length = line_length

            circ_breaker = branch['branch'].circuit_breaker
            if circ_breaker is not None:
                circ_breaker.geo_data = cut_line_by_distance(line_shp, 0.5, normalized=True)[0]

            # fill branch path dict
            path_passed_osmids.update(dict.fromkeys([(branch['adj_nodes'][0], branch['adj_nodes'][1]),
                                                     (branch['adj_nodes'][1], branch['adj_nodes'][0])], path))

        # one branch endpoint is within load area - update inner part of branch geometry
        elif endpoints.count(True) == 1:

            node_in = list(np.array(branch['adj_nodes'])[endpoints])[0]
            node_out = list(set(branch['adj_nodes']) - {node_in})[0]
            line_shp = branch['branch'].geometry
            branch_coords = line_shp.coords[:]

            # make sure branch shp is directed with origin inside of load area
            if branch_coords[0] != node_in.geo_data.coords[0]:
                branch_coords = list(reversed(branch_coords))
                line_shp = LineString(branch_coords)

            # retrieve intersection point(s) with load area polygon
            ring_shp = LinearRing(load_area.geo_area.exterior.coords)
            intersect_shp = line_shp.intersection(ring_shp)

            # in case of more than one intersection, find first intersection point
            if not isinstance(intersect_shp, Point):
                mp = MultiPoint(intersect_shp)
                intersect_shp = nearest_points(mp, node_out.geo_data)[0]

            # find nearest street graph node from intersection point
            G = conn_ding0_obj_to_osm_graph(G, node_out, search_shp=intersect_shp)

            # retrieve new geoemtry for branch
            line_shp, line_length, path = get_line_shp_from_shortest_path(G, node_in, node_out, return_path=True)

            branch['branch'].geometry = line_shp
            branch['branch'].length = line_length

            circ_breaker = branch['branch'].circuit_breaker
            if circ_breaker is not None:
                circ_breaker.geo_data = cut_line_by_distance(line_shp, 0.5, normalized=True)[0]

            # fill branch path dict
            path_passed_osmids.update(dict.fromkeys([(branch['adj_nodes'][0], branch['adj_nodes'][1]),
                                                     (branch['adj_nodes'][1], branch['adj_nodes'][0])], path))

    return path_passed_osmids, G

