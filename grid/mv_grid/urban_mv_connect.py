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


import time

from ding0.tools import config as cfg_ding0
from ding0.core.network.stations import *
from ding0.core.structure.regions import LVLoadAreaCentreDing0
from ding0.core.network.cable_distributors import MVCableDistributorDing0
from ding0.core.network import RingDing0, BranchDing0, CircuitBreakerDing0
import logging

#PAUl new
from ding0.grid.mv_grid.tools import get_edge_tuples_from_path, cut_line_by_distance
from shapely.ops import linemerge, split
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point


logger = logging.getLogger('ding0')

def mv_urban_connect(mv_grid, osm_graph_red, core_graph, stub_graph, stub_dict, osmid_branch_dict):

    node_list = {str(n):n for n in mv_grid.graph.nodes()}
    routed_graph_node_set = set(osmid_branch_dict.keys())
    forbidden_object_set = {n for n in routed_graph_node_set if n in node_list \
                            and isinstance(node_list[n], (MVStationDing0, MVCableDistributorDing0))}
    routed_graph_node_set = routed_graph_node_set - forbidden_object_set
    root_nodes_to_remove = []

    for key, stub_data in stub_dict.items():

        # get root_node, loads (mv_station or mv_load) and cable distributor nodes in osm stub graph
        load_node_set = set(stub_data['load'].keys())
        cabledist_node_set = stub_data['dist']
        root_node = stub_data['root']
        comp = load_node_set.union({root_node}, cabledist_node_set)

        # stub root node does not intersect with routed branches
        if not root_node in osmid_branch_dict:

            root_deg = stub_graph.out_degree(root_node)

            # root node has branching, old root node will be cable dist, new root node intersect with routed branches
            if root_deg > 1 and len(comp) > 2:
                # in case root node has deg > 1, build new edge between old and new root
                node = root_node 
                if not node in node_list.keys():
                    cabledist_node_set.add(node)

                _, path = nx.multi_source_dijkstra(osm_graph_red, routed_graph_node_set, node, weight='length')
                edge_path = get_edge_tuples_from_path(osm_graph_red, path)
                line_shp = linemerge([osm_graph_red.edges[edge]['geometry'] for edge in edge_path])

                root_node = path[0]
                stub_graph.add_node(root_node, x=osm_graph_red.nodes[root_node]['x'], y=osm_graph_red.nodes[root_node]['y'])
                stub_graph.add_edge(root_node, node, geometry=line_shp)
                comp.add(root_node)

            else:
                # in case deg = 1, delete old root and compute new edge from root neignbor to graph
                # compute new edge geometry to ding0 graph 
                root_nb = list({n for n in stub_graph.neighbors(root_node)} & set(comp))[0]
                _, path = nx.multi_source_dijkstra(osm_graph_red, routed_graph_node_set, root_nb, weight='length')
                edge_path = get_edge_tuples_from_path(osm_graph_red, path)
                line_shp = linemerge([osm_graph_red.edges[edge]['geometry'] for edge in edge_path])
                #remove old root node from graph and component
                stub_graph.remove_edge(root_node, root_nb)
                comp.discard(root_node)
                # introduce new root node and update edge
                root_node = path[0]
                stub_graph.add_node(root_node, x=osm_graph_red.nodes[root_node]['x'], y=osm_graph_red.nodes[root_node]['y'])
                stub_graph.add_edge(root_node, root_nb, geometry=line_shp)
                #print(root_node, root_nb)
                comp.add(root_node)

    #### SPLIT MAIN ROUTE EDGE INTO 2 SEGMENTS

        # root node is not station or mv load
        if not root_node in node_list.keys():

            # get branch to split, due to possible overlapping branch geometries of different rings
            # choose branch with lowest ring demand
            # branch_to_split = min([(branch, branch.ring._demand) for branch in osmid_branch_dict[root_node]], key=lambda x: x[1])[0]
            # print(branch_to_split.length, branch_to_split.ring)
            if root_node in osmid_branch_dict: # in case of the same root node for different stubs just do the splittage one time

                #add cable_dist  
                root_shp = Point(core_graph.nodes[root_node]['x'], core_graph.nodes[root_node]['y']) #core_graph
                cable_dist = MVCableDistributorDing0(geo_data=root_shp, grid=mv_grid)
                mv_grid.add_cable_distributor(cable_dist)

                # find branch to split in branch dict, do a shapely line.contains(point) check up 
                # in order to ensure successful splittage #check up has to be done because the 2 
                # new segements are added to the branch_dict for every iteration
                root_branches = sorted([(branch, branch.ring._demand) for branch in osmid_branch_dict[root_node]], key=lambda x: x[1]) #TODO sort both values

                i=0
                while not root_branches[i][0].geometry.contains(root_shp):
                    i+=1
                branch_to_split = root_branches[i][0]

                #get adjacent nodes
                adj_node1, adj_node2 = mv_grid.graph_nodes_from_branch(branch_to_split)

                # split branch geometry
                shp_to_split = branch_to_split.geometry
                line_shps = split(shp_to_split, root_shp)

                if len(line_shps) == 2: # general case
                # find respective geometry for adjacent nodes
                    if adj_node1.geo_data == line_shps[0].boundary[0]:
                        line_shp_1 = line_shps[0]
                        line_shp_2 = line_shps[1]
                    elif adj_node1.geo_data == line_shps[1].boundary[1]:
                        line_shp_1 = line_shps[1]
                        line_shp_2 = line_shps[0]
                else: #workaround in case len(line_shps) == 1
                    line_shp_1 = LineString([adj_node1.geo_data, root_shp])
                    line_shp_2 = LineString([root_shp, adj_node2.geo_data])

                # check if there's a circuit breaker on current branch,
                # if yes set new position between first node (adj_node1) and newly created cable distributor
                circ_breaker = branch_to_split.circuit_breaker
                if circ_breaker is not None:
                    circ_breaker.geo_data = cut_line_by_distance(line_shp_1, 0.5, normalized=True)[0]

                branch_length = line_shp_1.length
                branch1 = BranchDing0(geometry=line_shp_1,
                                     length=branch_length,
                                     circuit_breaker=circ_breaker,
                                     kind=branch_to_split.kind,
                                     grid=mv_grid,
                                     type=branch_to_split.type,
                                     ring=branch_to_split.ring)

                if circ_breaker is not None:
                    circ_breaker.branch = branch1

                mv_grid.graph.add_edge(adj_node1, cable_dist, branch=branch1)

                branch_length = line_shp_2.length
                branch2=BranchDing0(geometry=line_shp_2,
                                   length=branch_length,
                                   kind=branch_to_split.kind,
                                   grid=mv_grid,
                                   type=branch_to_split.type,
                                   ring=branch_to_split.ring)

                mv_grid.graph.add_edge(adj_node2, cable_dist, branch=branch2)

                # update branch dict: once a branch has been splitted -> remove,
                # once a root_node has been used for splittage -> remove, add
                # new resulting branches to branch_dict
                for osmid, branch_set in osmid_branch_dict.items():
                    if branch_to_split in branch_set:
                        branch_set.discard(branch_to_split)
                        branch_set.add(branch1)
                        branch_set.add(branch2)

                osmid_branch_dict.pop(root_node, None)

                # split old ring main route into 2 segments (delete old branch and create 2 new ones
                # along node)
                mv_grid.graph.remove_edge(adj_node1, adj_node2)

                root_node_ding0 = cable_dist
                branch_ring = branch_to_split.ring

                node_list[str(root_node_ding0)] = root_node_ding0

        # root node is station or mvload
        else: 
            root_node_ding0 = node_list[root_node] #type, kind, ring, circuit_breaker from station (hopefully)
            branch_ring = mv_grid.get_ring_from_node(root_node_ding0) # get branch ring obj

        #### CREATE & ADD NEW STUB BRANCHES

        # get default branch kind and type from grid to use it for new branch
        # branch_ring parameter from above
        branch_kind = mv_grid.default_branch_kind
        branch_type = mv_grid.default_branch_type

         # for simple stubs, there is just one load (one edge)
        if len(comp) == 2: 

            load_node = list(load_node_set)[0] #next(iter(
            #edge_key = next(iter(stub_graph.get_edge_data(root_node, load_node).keys())) #TODO uncomment
            #print(root_node, load_node)
            branch_shp = stub_graph.edges[root_node, load_node, 0]['geometry']
            branch_length = branch_shp.length

            mv_grid.graph.add_edge(root_node_ding0, node_list[load_node], branch=BranchDing0(geometry=branch_shp,
                                                                       length=branch_length,
                                                                       kind=branch_kind,
                                                                       grid=mv_grid,
                                                                       type=branch_type,
                                                                       ring=branch_ring))

            # add demand to ring
            branch_ring._demand += int(node_list[load_node].peak_load / 0.97) # TODO config

        else: #for stubs having more than one edge

                dist_mapping = {} #for converting osm names to ding0 names in comp_graph
                dist_mapping[root_node] = str(root_node_ding0)

                # add cable distributor for root, if is not station or load
                # add cable distributor for branchings if necessary
                #if not root_node in node_list.keys():
                #    cabledist_node_set.add(root_node)

                if len(cabledist_node_set):

                    for node in cabledist_node_set:

                        cable_dist_shp = Point(stub_graph.nodes[node]['x'], stub_graph.nodes[node]['y'])
                        cable_dist = MVCableDistributorDing0(geo_data=cable_dist_shp, grid=mv_grid)
                        mv_grid.add_cable_distributor(cable_dist)
                        # relabel cable dist osmid in stub graph, node_list, component with ding0 name
                        #comp.discard(node)
                        #comp.add(str(cable_dist))
                        dist_mapping[node] = str(cable_dist)
                        #stub_graph = nx.relabel_nodes(stub_graph, {node:str(cable_dist)})
                        node_list[str(cable_dist)] = cable_dist

                # build subgraph from component in order to extract edges
                comp_graph = nx.Graph(stub_graph.subgraph(comp)) #not stub_graph
                comp_graph = nx.relabel_nodes(comp_graph, dist_mapping)

                for edge in comp_graph.edges:

                    #add branches of component
                    n1, n2 = edge[0], edge[1]
                    node1, node2 = node_list[n1], node_list[n2]
                    branch_shp = comp_graph.edges[n1, n2]['geometry']
                    branch_length = branch_shp.length

                    mv_grid.graph.add_edge(node1, node2, branch=BranchDing0(geometry=branch_shp,
                                                                           length=branch_length,
                                                                           kind=branch_kind,
                                                                           grid=mv_grid,
                                                                           type=branch_type,
                                                                           ring=branch_ring))

                #add demand to ring
                comp_demand = sum([node_list[n].peak_load for n in load_node_set]) / 0.97 #TODO config # in future from dict
                branch_ring._demand += int(comp_demand)
        
    return mv_grid
