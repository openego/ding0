#### RESULTS

import numpy as np
import networkx as nx
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

from ding0.core.structure.regions import LVLoadAreaCentreDing0, LVGridDistrictDing0
from ding0.core.network.stations import LVStationDing0
from ding0.core.network.loads import MVLoadDing0
from ding0.core import MVCableDistributorDing0


def evaluate_grid_length(nd):
    
    ring_length = [] # total length of grid's main rings
    stub_length = [] # total length of grid's stub lines

    for mvgd in nd.mv_grid_districts():

        g = mvgd.mv_grid.graph.copy()

        # k-core of graph describes main ring topology
        g_core = nx.k_core(g, k=2)
        rings_edges = [n for n in g.edges if n in g_core.edges]
        # stub graph described by residual components 
        g_stub = g.copy()
        g_stub.remove_edges_from(rings_edges)
        g_stub.remove_nodes_from(list(nx.isolates(g_stub)))

        # get edges separately per ring
        for c in nx.cycle_basis(g_core, mvgd.mv_grid._station):
            ring_edges = g_core.subgraph(c).edges
            ring_length.append(sum([mvgd.mv_grid._graph.edges[e]['branch'].length for e in ring_edges]))

        # get edges separately per stub component
        for c in nx.connected_components(g_stub):
            stub_edges = g_stub.subgraph(c).edges
            stub_length.append(sum([mvgd.mv_grid._graph.edges[e]['branch'].length for e in stub_edges]))

    # length data output row-wise per ring stub / component
    df = pd.DataFrame.from_dict({'ring_length': ring_length, 'stub_length': stub_length}, orient='index')
    df = df.transpose()
    
    return df


def get_load_specific_results(nd):
    
    #mvgd
    mvgd_no_tot = len(nd._mv_grid_districts)
    mvgd_area_tot = sum([mvgd.geo_data.area for mvgd in nd.mv_grid_districts()])

    #la
    las = []
    las_urban = []
    las_rural = []

    las_area = []
    las_area_urban = []
    las_area_rural = []

    las_urban_isolated = []
    las_rural_isolated = []

    for mvgd in nd.mv_grid_districts():
        for la in mvgd._lv_load_areas:
            las.append(la)
            las_area.append(la.geo_area.area)
            if la.is_aggregated:
                las_urban.append(la)
                las_area_urban.append(la.geo_area.area)
                if nx.is_empty(la.load_area_graph):
                    las_urban_isolated.append(la)
            else:
                las_rural.append(la)
                las_area_rural.append(la.geo_area.area)
                if nx.is_empty(la.load_area_graph):
                    las_rural_isolated.append(la)

    la_no_total = len(las)
    la_urban_no_total = len(las_urban)
    la_rural_no_total = len(las_rural)

    la_area_total = sum(las_area)
    la_urban_area_total = sum(las_area_urban)
    la_urban_area_mean = np.mean(las_area_urban)
    la_rural_area_total = sum(las_area_rural)
    la_rural_area_mean = np.mean(las_area_rural)
    
    if la_urban_no_total == 0:
        la_urban_isolated_tot_rate = 0
    else:
        la_urban_isolated_tot_rate = len(las_urban_isolated) / la_urban_no_total * 100
        
    if la_rural_no_total == 0:
        la_rural_isolated_tot_rate = 0
    else:
        la_rural_isolated_tot_rate = len(las_rural_isolated) / la_rural_no_total * 100

    # nodes
    mvgd_nodes_tot = []
    mvgd_nodes_tot_isl = []

    mvgd_urban_load_nodes = []
    mvgd_rural_load_nodes = []

    ring_stations_tot = []
    ring_loads_tot = []
    stub_stations_tot = []
    stub_loads_tot = []

    for mvgd in nd.mv_grid_districts():
        # total
        mvgd_g_nodes = [node for node in mvgd.mv_grid.graph.nodes() if isinstance(node, LVStationDing0) or isinstance(node, MVLoadDing0) or isinstance(node, MVCableDistributorDing0)] 
        mvgd_g_nodes_isl = [node for node in mvgd.mv_grid.graph_isolated_nodes() if isinstance(node, LVStationDing0) or isinstance(node, MVLoadDing0) or isinstance(node, MVCableDistributorDing0)]
        mvgd_nodes_tot.extend(mvgd_g_nodes)
        mvgd_nodes_tot_isl.extend(mvgd_g_nodes_isl)
        # urban
        for la in mvgd._lv_load_areas:
            if la.is_aggregated:
                mvgd_urban_load_nodes.extend(la._mv_loads)
                mvgd_urban_load_nodes.extend(la._lv_grid_districts)
            else:
                mvgd_rural_load_nodes.extend(la._mv_loads)
                mvgd_rural_load_nodes.extend(la._lv_grid_districts)

        g = mvgd.mv_grid.graph.copy() # remove isolated nodes
        g.remove_nodes_from(list(nx.isolates(g)))
        g_core = nx.k_core(g, k=2)
        ring_nodes = g_core.nodes
        stub_nodes = list(set(g.nodes) - set(ring_nodes))

        ring_stations = [node for node in ring_nodes if isinstance(node, LVStationDing0)]
        ring_stations_tot.extend(ring_stations)
        ring_loads = [node for node in ring_nodes if isinstance(node, MVLoadDing0)]
        ring_loads_tot.extend(ring_loads)
        stub_stations = [node for node in stub_nodes if isinstance(node, LVStationDing0)]
        stub_stations_tot.extend(stub_stations)
        stub_loads = [node for node in stub_nodes if isinstance(node, MVLoadDing0)]
        stub_loads_tot.extend(stub_loads)

    mvgd_nodes_tot_no = len(mvgd_nodes_tot)
    mvgd_nodes_dist_no = len([node for node in mvgd_nodes_tot if isinstance(node, MVCableDistributorDing0)])
    mvgd_nodes_subst_no = len([node for node in mvgd_nodes_tot if isinstance(node, LVStationDing0)])
    mvgd_nodes_load_no = len([node for node in mvgd_nodes_tot if isinstance(node, MVLoadDing0)])
    mvgd_nodes_tot_isl_no = len(mvgd_nodes_tot_isl)
    mvgd_urban_load_nodes_no = len(mvgd_urban_load_nodes) - len([node for node in mvgd_urban_load_nodes if isinstance(node, LVGridDistrictDing0) and int(node.peak_load) == 0])
    mvgd_rural_load_nodes_no = len(mvgd_rural_load_nodes) 
    mvgd_urban_isolated_nodes_no = len(set(mvgd_nodes_tot_isl) & set([node.lv_grid._station if isinstance(node, LVGridDistrictDing0) else node for node in mvgd_urban_load_nodes]))
    mvgd_rural_isolated_nodes_no = len(set(mvgd_nodes_tot_isl) & set([node.lv_grid._station if isinstance(node, LVGridDistrictDing0) else node for node in mvgd_rural_load_nodes]))

    ring_stations_tot_no = len(ring_stations_tot)
    ring_loads_tot_no = len(ring_loads_tot)
    stub_stations_tot_no = len(stub_stations_tot)
    stub_loads_tot_no = len(stub_loads_tot)

    res = {}
    
    res['mvgd_tot_no'] = int(mvgd_no_tot)
    res['mvgd_area_tot [km]'] = np.round_(mvgd_area_tot/1000000, 1)

    res['la_no_total [km]'] = la_no_total
    res['la_area_total [km]'] = np.round_(la_area_total/1000000, 1)
    res['la_urban_no_total [km]'] = la_urban_no_total
    res['la_urban_area_total [km]'] = np.round_(la_urban_area_total/1000000, 1)
    res['la_urban_area_mean [km]'] = np.round_(la_urban_area_mean/1000000, 1)
    res['la_urban_isolated_tot_rate [%]'] = np.round_(la_urban_isolated_tot_rate, 1)
    res['la_rural_no_total [km]'] = la_rural_no_total
    res['la_rural_area_total [km]'] = np.round_(la_rural_area_total/1000000, 1)
    res['la_rural_area_mean [km]'] = np.round_(la_rural_area_mean/1000000, 1)
    res['la_rural_isolated_tot_rate [%]'] = np.round_(la_rural_isolated_tot_rate, 1)

    res['mvgd_nodes_tot_no'] = mvgd_nodes_tot_no
    res['mvgd_nodes_dist_no'] = mvgd_nodes_dist_no
    res['mvgd_nodes_subst_no'] = mvgd_nodes_subst_no
    res['mvgd_nodes_load_no'] = mvgd_nodes_load_no
    res['mvgd_nodes_tot_isl_no'] = mvgd_nodes_tot_isl_no
    res['mvgd_urban_load_nodes_no'] = mvgd_urban_load_nodes_no
    res['mvgd_urban_isolated_nodes_no'] = mvgd_urban_isolated_nodes_no
    res['mvgd_rural_load_nodes_no'] = mvgd_rural_load_nodes_no
    res['mvgd_rural_isolated_nodes_no'] = mvgd_rural_isolated_nodes_no
    res['ring_stations_tot_no'] = ring_stations_tot_no
    res['stub_stations_tot_no'] = stub_stations_tot_no
    res['ring_loads_tot_no'] = ring_loads_tot_no
    res['stub_loads_tot_no'] = stub_loads_tot_no

    df = pd.DataFrame.from_dict(res, orient='index')
    
    return df


# suitable for comparison with simbench grid

def get_structural_data_per_mvgd(nd):
    
    # returns data per row-wise per mvgd

    data = {}

    for mvgd in nd.mv_grid_districts():

        data[mvgd] = {}

        #### MVGD
        # nominal voltage
        data[mvgd]['v_level [kV]'] = mvgd.mv_grid.v_level
        # hsms trafo sizes (list)
        hs_ms_trafo_sizes = [t.s_max_a for t in mvgd.mv_grid._station._transformers]
        data[mvgd]['hs_ms_trafo_sizes [MVA]'] = [int(t / 1000) for t in hs_ms_trafo_sizes]
        # geo area (not simbench)
        data[mvgd]['mvgd_geo_area [km2]'] = np.round_(mvgd.geo_data.area / 1000000, 1)

        #### LA / LVGD

        urban_la_area = sum([la.geo_area.area for la in mvgd._lv_load_areas if la.is_aggregated == True])
        tot_la_area = sum([la.geo_area.area for la in mvgd._lv_load_areas])
        # urbanisation ratio
        data[mvgd]['urban_area_ratio'] = int(urban_la_area / tot_la_area * 100)

        ms_ns_inst_power = 0 # per mvgd
        ms_ns_load_power_s = 0

        for la in mvgd._lv_load_areas:

            for lvgd in la._lv_grid_districts: # Buildings

                for trafo in lvgd.lv_grid._station._transformers:
                    ms_ns_inst_power += trafo.s_max_a

                ms_ns_load_power_s += lvgd.peak_load

        ms_load_power_s = sum([load.peak_load * 0.6 for load in mvgd.mv_grid._loads]) # no diversity

        # installed ms-ns-power
        data[mvgd]['ms_ns_inst_power [MVA]'] = np.round_(ms_ns_inst_power / 1000, 1)
        # ms-ns power / hs-ms power ratio
        data[mvgd]['ms_ns_hs_ms_power_ratio [%]'] = np.round_(ms_ns_inst_power / sum(hs_ms_trafo_sizes) * 100, 1)
        # ms-ns power + ms_load / hs-ms power ratio
        data[mvgd]['load_hs_ms_power_ratio [%]'] = np.round_((ms_ns_inst_power + ms_load_power_s) / sum(hs_ms_trafo_sizes) * 100, 1)
        # load_power_sum
        data[mvgd]['load_power_sum_p [MW]'] = np.round_((ms_load_power_s + ms_ns_load_power_s) * 0.97 / 1000, 1)

        #### BRANCHES

        branch_lengths = [edge['branch'].length for edge in list(mvgd.mv_grid.graph_edges())]
        cum_branch_length = int(sum(branch_lengths))
        cum_cable_length = int(sum([edge['branch'].length for edge in list(mvgd.mv_grid.graph_edges()) if edge['branch'].kind == 'cable']))
        mean_branch_length = np.mean(branch_lengths)
        no_branches = len(branch_lengths)
        data[mvgd]['cum_branch_length [km]'] = np.round_(cum_branch_length / 1000, 1)
        if cum_branch_length == 0:
            data[mvgd]['cable_ratio [%]'] = 0
        else:
            data[mvgd]['cable_ratio [%]'] = int(cum_cable_length / cum_branch_length * 100) # calculation not necessary, is set in validate mvgd
        
        data[mvgd]['branch_lengths [km]'] = list(np.round_(np.divide(branch_lengths, 1000), 3))
        data[mvgd]['mean_branch_length [km]'] = np.round_(mean_branch_length / 1000, 3)
        data[mvgd]['no_branches'] = no_branches

        ### GRAPH

        # open rings to extraxt feeders
        nd.control_circuit_breakers(mode='open')

        mv_station = mvgd.mv_grid._station

        G = mvgd.mv_grid._graph.copy()
        data[mvgd]['nodes_no'] = len([node for node in G.nodes])
        data[mvgd]['load_nodes_no'] = len([node for node in G.nodes if isinstance(node, LVStationDing0) or isinstance(node, MVLoadDing0)])
        G.remove_node(mv_station)
        G.remove_nodes_from(mvgd.mv_grid.graph_isolated_nodes())

        # feeder start nodes
        root_edges = [(mv_station,n) for n in mvgd.mv_grid._graph.neighbors(mv_station)] 
        # feeders (contain nodes)
        feeder_comp = list(nx.connected_components(G)) 
        # number of feeders
        data[mvgd]['no_feeders'] = len(feeder_comp) 

        feeder_branch_len_mvgd = []
        supply_nodes_per_feeder_mvgd = []

        for feeder in feeder_comp:

            station_nodes = [node for node in feeder if isinstance(node, LVStationDing0)]
            load_nodes = [node for node in feeder if isinstance(node, MVLoadDing0)]
            dist_nodes = [node for node in feeder if isinstance(node, MVCableDistributorDing0)]

            # nodes per feeder
            nodes_per_feeder = feeder
            # supply nodes per feeder
            supply_nodes_per_feeder = len([node for node in feeder if isinstance(node, LVStationDing0) or isinstance(node, MVLoadDing0)])
            supply_nodes_per_feeder_mvgd.append(supply_nodes_per_feeder)
            # feeder branches
            edges = list(G.edges(nodes_per_feeder))
            edges.append([e_tuple for e_tuple in root_edges if e_tuple[1] in nodes_per_feeder][0])
            # feeder len
            feeder_branch_len = sum([mvgd.mv_grid._graph.edges[e]['branch'].length for e in edges])
            if feeder_branch_len > 1:
                feeder_branch_len_mvgd.append(feeder_branch_len)
        
        data[mvgd]['feeder_lengths [km]'] = list(np.round_(np.divide(feeder_branch_len_mvgd, 1000), 1))
        data[mvgd]['feeder_len_mean [km]'] = np.round_(np.mean(feeder_branch_len_mvgd) / 1000, 1)
        data[mvgd]['feeder_len_min [km]'] = np.round_(min(feeder_branch_len_mvgd) / 1000, 1)
        data[mvgd]['feeder_len_max [km]'] = np.round_(max(feeder_branch_len_mvgd) / 1000, 1)
        
        data[mvgd]['supply_nodes_per_feeder'] = supply_nodes_per_feeder_mvgd
        data[mvgd]['supply_nodes_per_feeder_mean'] = np.round_(np.mean(supply_nodes_per_feeder_mvgd), 1)
        data[mvgd]['supply_nodes_per_feeder_min'] = min(supply_nodes_per_feeder_mvgd)
        data[mvgd]['supply_nodes_per_feeder_max'] = max(supply_nodes_per_feeder_mvgd)

        nd.control_circuit_breakers(mode='close')

    df = pd.DataFrame.from_dict(data, orient='index')
    
    return df


def evaluate_substation_geo_parameters(nd):

    ## BUILDINGS PER STATION

    buildings_per_station = []
    average_dist_stations = []
    max_supply_radius = []
    max_urban_radius = []

    for mvgd in nd.mv_grid_districts():
        for la in mvgd._lv_load_areas:
            if la.is_aggregated:

                # buildings_per_station
                buildings_per_station.extend([int(sum(lvgd.buildings.number_households)) for lvgd in la._lv_grid_districts])

                # average_dist_stations
                coords = [[lvgd.lv_grid._station.geo_data.x,lvgd.lv_grid._station.geo_data.y] for lvgd in la._lv_grid_districts]
                if len(coords) > 1:
                    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(coords)
                    distances, indices = nbrs.kneighbors(coords)
                    distances = [d[1] for d in distances]
                    average_dist_stations.extend(distances)

                # max_supply_radius_ms_ns
                for lvgd in la._lv_grid_districts:
                    station_coords = list(lvgd.lv_grid._station.geo_data.coords)
                    building_coords = [(p.x,p.y) for p in lvgd.buildings.raccordement_building.to_list()]
                    if len(building_coords) > 1:
                        radius_max = distance.cdist(station_coords, building_coords, 'euclidean').max()
                        max_supply_radius.append(radius_max)

                # max_supply_radius_hs_ms
                station_coords = list(la.mv_grid_district.mv_grid._station.geo_data.coords)
                poly_ext_coords = list(la.geo_area.exterior.coords)
                if len(poly_ext_coords) > 1:
                    radius_max = distance.cdist(station_coords, poly_ext_coords, 'euclidean').max()
                    max_urban_radius.append(radius_max)
                    
    lit = {}
    lit['buildings_per_station'] = str([int(b) for b in buildings_per_station])
    lit['average_dist_stations'] = str([int(b) for b in average_dist_stations])
    lit['max_supply_radius'] = str([int(b) for b in max_supply_radius])
    lit['max_urban_radius'] = str([int(b) for b in max_urban_radius])

    df = pd.DataFrame(lit, index=[0])

    return df


def evaluate_supply_point_parameters(nd):
    
    mv_peak_loads = [] # peak loads of mv loads
    station_load_factor = [] # ratio of lvgd peak load / installed trafo capacity of mv/lv substation 
    trafo_sizes = [] # distribution of  installed trafo capacity of mv/lv substation(s) 
    trafo_per_station = [] # nested list with installed trafos per mv_lv substation

    for mvgd in nd.mv_grid_districts():
        load_val = [load.peak_load for load in mvgd.mv_grid._loads]
        mv_peak_loads.extend(load_val)
        for la in mvgd._lv_load_areas:
            for lvgd in la._lv_grid_districts:
                station_pl = lvgd.peak_load
                trafo_power = [t.s_max_a for t in lvgd.lv_grid._station._transformers]
                trafo_sizes.extend(trafo_power)
                trafo_per_station.append(trafo_power)
                ratio = station_pl / sum(trafo_power)
                station_load_factor.append(ratio)
                
        
    d = {}
    d['mv_peak_loads'] = mv_peak_loads
    d['station_load_factor'] = station_load_factor
    d['trafo_sizes'] = trafo_sizes
    d['trafo_per_station'] = trafo_per_station
    
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()
    
    return df