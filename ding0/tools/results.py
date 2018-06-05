"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


import pickle
import numpy as np
import pandas as pd
import time
import os
import json
import re

from matplotlib import pyplot as plt
import seaborn as sns

from egoio.tools import db
from ding0.core import NetworkDing0
from ding0.core import GeneratorDing0
from ding0.core import LVCableDistributorDing0, MVCableDistributorDing0
from ding0.core import MVStationDing0, LVStationDing0
from ding0.core import CircuitBreakerDing0
from ding0.core.network.loads import LVLoadDing0, MVLoadDing0
from ding0.core import LVLoadAreaCentreDing0

from shapely.ops import transform
import pyproj
from functools import partial

from geoalchemy2.shape import from_shape
from sqlalchemy.orm import sessionmaker
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Point, MultiPoint, MultiLineString, LineString
from shapely.geometry import shape, mapping


import multiprocessing as mp

from math import floor, ceil, pi

from ding0.flexopt.check_tech_constraints import check_load, check_voltage, \
    get_critical_line_loading, get_critical_voltage_at_nodes
from ding0.tools import config as cfg_ding0

import networkx as nx

#############################################
plt.close('all')
cfg_ding0.load_config('config_db_tables.cfg')
cfg_ding0.load_config('config_calc.cfg')
cfg_ding0.load_config('config_files.cfg')
cfg_ding0.load_config('config_misc.cfg')

def lv_grid_generators_bus_bar(nd):
    """
    Calculate statistics about generators at bus bar in LV grids

    Parameters
    ----------
    nd : ding0.NetworkDing0
        Network container object

    Returns
    -------
    lv_stats : dict
        Dict with keys of LV grid repr() on first level. Each of the grids has
        a set of statistical information about its topology
    """

    lv_stats = {}

    for la in nd._mv_grid_districts[0].lv_load_areas():
        for lvgd in la.lv_grid_districts():

            station_neighbors = list(lvgd.lv_grid._graph[
                                         lvgd.lv_grid._station].keys())

            # check if nodes of a statio are members of list generators
            station_generators = [x for x in station_neighbors
                                  if x in lvgd.lv_grid.generators()]

            lv_stats[repr(lvgd.lv_grid._station)] = station_generators


    return lv_stats


def save_nd_to_pickle(nd, path='', filename=None):
    """Use pickle to save the whole nd-object to disc

    The network instance is entirely pickled to a file.

    Parameters
    ----------
    nd : NetworkDing0
        Ding0 grid container object
    path : str
        Absolute or relative path where pickle should be saved. Default is ''
        which means pickle is save to PWD

    """

    abs_path = os.path.abspath(path)

    if len(nd._mv_grid_districts) > 1:
        name_extension = '_{number}-{number2}'.format(
            number=nd._mv_grid_districts[0].id_db,
            number2=nd._mv_grid_districts[-1].id_db)
    else:
        name_extension = '_{number}'.format(number=nd._mv_grid_districts[0].id_db)

    if filename is None:
        filename = "ding0_grids_{ext}.pkl".format(
            ext=name_extension)

    # delete attributes of `nd` in order to make pickling work
    # del nd._config
    del nd._orm

    pickle.dump(nd, open(os.path.join(abs_path, filename),"wb"))


def load_nd_from_pickle(filename=None, path=''):
    """
    Use pickle to save the whole nd-object to disc

    Parameters
    ----------
    filename : str
        Filename of nd pickle
    path : str
        Absolute or relative path where pickle should be saved. Default is ''
        which means pickle is save to PWD

    Returns
    -------
    nd : NetworkDing0
        Ding0 grid container object
    """

    abs_path = os.path.abspath(path)

    if filename is None:
        raise NotImplementedError

    return pickle.load(open(os.path.join(abs_path, filename),"rb"))


def plot_cable_length(stats, plotpath):
    """
    Cable length per MV grid district
    """

    # cable and line kilometer distribution
    f, axarr = plt.subplots(2, sharex=True)
    stats.hist(column=['km_cable'], bins=5, alpha=0.5, ax=axarr[0])
    stats.hist(column=['km_line'], bins=5, alpha=0.5, ax=axarr[1])

    plt.savefig(os.path.join(plotpath,
                             'Histogram_cable_line_length.pdf'))

def plot_generation_over_load(stats, plotpath):
    """

    :param stats:
    :param plotpath:
    :return:
    """

    # Generation capacity vs. peak load
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("ticks")

    # reformat to MW
    stats[['generation_capacity', 'peak_load']] = stats[['generation_capacity',
                                                         'peak_load']] / 1e3

    sns.lmplot('generation_capacity', 'peak_load',
               data=stats,
               fit_reg=False,
               hue='v_nom',
               # hue='Voltage level',
               scatter_kws={"marker": "D",
                            "s": 100},
               aspect=2)
    plt.title('Peak load vs. generation capcity')
    plt.xlabel('Generation capacity in MW')
    plt.ylabel('Peak load in MW')

    plt.savefig(os.path.join(plotpath,
                             'Scatter_generation_load.pdf'))


def plot_km_cable_vs_line(stats, plotpath):
    """

    :param stats:
    :param plotpath:
    :return:
    """

    # Cable vs. line kilometer scatter
    sns.lmplot('km_cable', 'km_line',
               data=stats,
               fit_reg=False,
               hue='v_nom',
               # hue='Voltage level',
               scatter_kws={"marker": "D",
                            "s": 100},
               aspect=2)
    plt.title('Kilometer of cable/line')
    plt.xlabel('Km of cables')
    plt.ylabel('Km of overhead lines')

    plt.savefig(os.path.join(plotpath,
                             'Scatter_cables_lines.pdf'))


def concat_nd_pickles(self, mv_grid_districts):
    """
    Read multiple pickles, join nd objects and save to file

    Parameters
    ----------
    mv_grid_districts : list
        Ints describing MV grid districts
    """

    pickle_name = cfg_ding0.get('output', 'nd_pickle')
    # self.nd = self.read_pickles_from_files(pickle_name)


    # TODO: instead of passing a list of mvgd's, pass list of filenames plus optionally a basth_path
    for mvgd in mv_grid_districts[1:]:

        filename = os.path.join(
            self.base_path,
            'results', pickle_name.format(mvgd))
        if os.path.isfile(filename):
            mvgd_pickle = pickle.load(open(filename, 'rb'))
            if mvgd_pickle._mv_grid_districts:
                mvgd.add_mv_grid_district(mvgd_pickle._mv_grid_districts[0])

    # save to concatenated pickle
    pickle.dump(mvgd,
                open(os.path.join(
                    self.base_path,
                    'results',
                    "ding0_grids_{0}-{1}.pkl".format(
                        mv_grid_districts[0],
                        mv_grid_districts[-1])),
                    "wb"))

    # save stats (edges and nodes data) to csv
    nodes, edges = mvgd.to_dataframe()
    nodes.to_csv(os.path.join(
        self.base_path,
        'results', 'mvgd_nodes_stats_{0}-{1}.csv'.format(
            mv_grid_districts[0], mv_grid_districts[-1])),
        index=False)
    edges.to_csv(os.path.join(
        self.base_path,
        'results', 'mvgd_edges_stats_{0}-{1}.csv'.format(
            mv_grid_districts[0], mv_grid_districts[-1])),
        index=False)

####################################################
def calculate_lvgd_stats(nw):
    """
    LV Statistics for an arbitrary network

    Parameters
    ----------
    nw: :any:`list` of NetworkDing0
        The MV grid(s) to be studied

    Returns
    -------
    lvgd_stats : pandas.DataFrame
        Dataframe containing several statistical numbers about the LVGD
    """
    ##############################
    #  ETRS (equidistant) to WGS84 (conformal) projection
    proj = partial(
        pyproj.transform,
        # pyproj.Proj(init='epsg:3035'),  # source coordinate system
        #  pyproj.Proj(init='epsg:4326'))  # destination coordinate system
        pyproj.Proj(init='epsg:4326'), # source coordinate system
        pyproj.Proj(init='epsg:3035'))  # destination coordinate system
    ##############################
    #close circuit breakers
    nw.control_circuit_breakers(mode='close')
    ##############################
    lv_dist_idx = 0
    lv_dist_dict = {}
    lv_gen_idx = 0
    lv_gen_dict = {}
    lv_load_idx = 0
    lv_load_dict = {}
    branch_idx = 0
    branches_dict = {}
    trafos_idx = 0
    trafos_dict = {}
    for mv_district in nw.mv_grid_districts():
        for LA in mv_district.lv_load_areas():
            for lv_district in LA.lv_grid_districts():
                lv_dist_idx += 1
                branches_from_station = len(lv_district.lv_grid.graph_branches_from_node(lv_district.lv_grid.station()))
                lv_dist_dict[lv_dist_idx] = {
                    'MV_grid_id': mv_district.mv_grid.id_db,
                    'LV_grid_id': lv_district.lv_grid.id_db,
                    'Load Area ID': LA.id_db,
                    'Population': lv_district.population,
                    'Peak Load Residential':lv_district.peak_load_residential,
                    'Peak Load Retail':lv_district.peak_load_retail,
                    'Peak Load Industrial':lv_district.peak_load_industrial,
                    'Peak Load Agricultural':lv_district.peak_load_agricultural,
                    'N° of Sector Residential': lv_district.sector_count_residential,
                    'N° of Sector Retail': lv_district.sector_count_retail,
                    'N° of Sector Industrial': lv_district.sector_count_industrial,
                    'N° of Sector Agricultural': lv_district.sector_count_agricultural,
                    'Accum. Consumption Residential': lv_district.sector_consumption_residential,
                    'Accum. Consumption Retail': lv_district.sector_consumption_retail,
                    'Accum. Consumption Industrial': lv_district.sector_consumption_industrial,
                    'Accum. Consumption Agricultural': lv_district.sector_consumption_agricultural,
                    'N° of branches from LV Station': branches_from_station,
                    'Load Area is Aggregated': LA.is_aggregated,
                    'Load Area is Satellite': LA.is_satellite,
                }
                # generation capacity
                for g in lv_district.lv_grid.generators():
                    lv_gen_idx+=1
                    subtype = g.subtype
                    if subtype == None:
                        subtype = 'other'
                    type = g.type
                    if type == None:
                        type = 'other'
                    lv_gen_dict[lv_gen_idx] = {
                        'LV_grid_id': lv_district.lv_grid.id_db,
                        'v_level':g.v_level,
                        'subtype':type+'/'+subtype,
                        'GenCap':g.capacity,
                    }
                #nodes bzw. LV loads
                for node in lv_district.lv_grid.graph_nodes_sorted():
                    if isinstance(node,LVLoadDing0):
                        lv_load_idx +=1
                        if 'agricultural' in node.consumption:
                            tipo = 'agricultural'
                        elif 'industrial' in node.consumption:
                            tipo = 'ind_ret'
                        elif 'residential' in node.consumption:
                            tipo = 'residential'
                        else:
                            tipo = 'none'
                        lv_load_dict[lv_load_idx] = {
                            'LV_grid_id': lv_district.lv_grid.id_db,
                            'load_type': tipo,
                        }
                #branches
                for branch in lv_district.lv_grid.graph_edges():
                    branch_idx+=1
                    branches_dict[branch_idx] = {
                        'LV_grid_id':lv_district.lv_grid.id_db,
                        'length': branch['branch'].length / 1e3,
                        'type_name': branch['branch'].type.to_frame().columns[0],
                        'type_kind': branch['branch'].kind,
                    }
                #Trafos
                for trafo in lv_district.lv_grid.station().transformers():
                    trafos_idx+=1
                    trafos_dict[trafos_idx] = {
                        'LV_grid_id':lv_district.lv_grid.id_db,
                        's_max_a':trafo.s_max_a,
                    }

                # geographic
                district_geo = transform(proj, lv_district.geo_data)
                lv_dist_dict[lv_dist_idx].update({'Area':district_geo.area})

    lvgd_stats = pd.DataFrame.from_dict(lv_dist_dict, orient='index').set_index('LV_grid_id')
    #generate partial dataframes
    gen_df = pd.DataFrame.from_dict(lv_gen_dict, orient='index')
    load_df = pd.DataFrame.from_dict(lv_load_dict, orient='index')
    branch_df = pd.DataFrame.from_dict(branches_dict, orient='index')
    trafos_df = pd.DataFrame.from_dict(trafos_dict, orient='index')

    #resque desired data
    if not gen_df.empty:
        #generation by voltage level
        lv_generation = gen_df.groupby(['LV_grid_id', 'v_level'])['GenCap'].sum().to_frame().unstack(level=-1)
        lv_generation.columns = ['Gen. Cap. v_level ' + str(_[1]) if isinstance(_, tuple) else str(_) for _ in lv_generation.columns]
        lvgd_stats = pd.concat([lvgd_stats, lv_generation], axis=1)
        #generation by type/subtype
        lv_generation = gen_df.groupby(['LV_grid_id','subtype'])['GenCap'].sum().to_frame().unstack(level=-1)
        lv_generation.columns = ['Gen. Cap. type ' + str(_[1]) if isinstance(_, tuple) else str(_) for _ in lv_generation.columns]
        lvgd_stats = pd.concat([lvgd_stats, lv_generation], axis=1)
    if not load_df.empty:
        #number of residential loads
        lv_loads = load_df[load_df['load_type']=='residential'].groupby(['LV_grid_id'])['load_type'].count().to_frame()#.unstack(level=-1)
        lv_loads.columns = ['N° of loads residential']
        lvgd_stats = pd.concat([lvgd_stats, lv_loads], axis=1)
        #number of agricultural loads
        lv_loads = load_df[load_df['load_type']=='agricultural'].groupby(['LV_grid_id'])['load_type'].count().to_frame()#.unstack(level=-1)
        lv_loads.columns = ['N° of loads agricultural']
        lvgd_stats = pd.concat([lvgd_stats, lv_loads], axis=1)
        #number of mixed industrial / retail loads
        lv_loads = load_df[load_df['load_type']=='ind_ret'].groupby(['LV_grid_id'])['load_type'].count().to_frame()#.unstack(level=-1)
        lv_loads.columns = ['N° of loads mixed industrial/retail']
        lvgd_stats = pd.concat([lvgd_stats, lv_loads], axis=1)

    if not branch_df.empty:
        #branches by type name
        lv_branches = branch_df.groupby(['LV_grid_id','type_name'])['length'].sum().to_frame().unstack(level=-1)
        lv_branches.columns = ['Length Type ' + _[1] if isinstance(_, tuple) else _ for _ in lv_branches.columns]
        lvgd_stats = pd.concat([lvgd_stats, lv_branches], axis=1)
        #branches by kind
        lv_branches = branch_df[branch_df['type_kind']=='line'].groupby(['LV_grid_id'])['length'].sum().to_frame()
        lv_branches.columns = ['Length of overhead lines']
        lvgd_stats = pd.concat([lvgd_stats, lv_branches], axis=1)
        lv_branches = branch_df[branch_df['type_kind']=='cable'].groupby(['LV_grid_id'])['length'].sum().to_frame()
        lv_branches.columns = ['Length of underground cables']
        lvgd_stats = pd.concat([lvgd_stats, lv_branches], axis=1)
        #N°of branches
        lv_branches = branch_df.groupby(['LV_grid_id','type_name'])['length'].count().to_frame().unstack(level=-1)
        lv_branches.columns = ['N° of branches Type ' + _[1] if isinstance(_, tuple) else _ for _ in lv_branches.columns]
        lvgd_stats = pd.concat([lvgd_stats, lv_branches], axis=1)
        lv_branches = branch_df[branch_df['type_kind']=='line'].groupby(['LV_grid_id'])['length'].count().to_frame()
        lv_branches.columns = ['N° of branches overhead lines']
        lvgd_stats = pd.concat([lvgd_stats, lv_branches], axis=1)
        lv_branches = branch_df[branch_df['type_kind']=='cable'].groupby(['LV_grid_id'])['length'].count().to_frame()
        lv_branches.columns = ['N° of branches underground cables']
        lvgd_stats = pd.concat([lvgd_stats, lv_branches], axis=1)
    if not trafos_df.empty:
        #N of trafos
        lv_trafos = trafos_df.groupby(['LV_grid_id'])['s_max_a'].count().to_frame()
        lv_trafos.columns = ['N° of MV/LV Trafos']
        lvgd_stats = pd.concat([lvgd_stats, lv_trafos], axis=1)
        #Capacity of trafos
        lv_trafos = trafos_df.groupby(['LV_grid_id'])['s_max_a'].sum().to_frame()
        lv_trafos.columns = ['Accumulated s_max_a in MVLV trafos']
        lvgd_stats = pd.concat([lvgd_stats, lv_trafos], axis=1)

    lvgd_stats = lvgd_stats.fillna(0)
    lvgd_stats = lvgd_stats[sorted(lvgd_stats.columns.tolist())]
    return lvgd_stats

####################################################
def calculate_mvgd_stats(nw):
    """
    MV Statistics for an arbitrary network

    Parameters
    ----------
    nw: :any:`list` of NetworkDing0
        The MV grid(s) to be studied

    Returns
    -------
    mvgd_stats : pandas.DataFrame
        Dataframe containing several statistical numbers about the MVGD
    """
    ##############################

    omega = 2 * pi * 50

    #close circuit breakers
    nw.control_circuit_breakers(mode='close')
    ##############################
    # Collect info from nw into dataframes
    # define dictionaries for collection
    trafos_dict = {}
    generators_dict = {}
    branches_dict = {}
    ring_dict = {}
    LA_dict = {}
    other_nodes_dict = {}
    lv_branches_dict = {}
    # initiate indexes
    trafos_idx = 0
    gen_idx = 0
    branch_idx = 0
    ring_idx = 0
    LA_idx = 0
    lv_branches_idx = 0

    # loop over MV grid districts
    for district in nw.mv_grid_districts():

        # node of MV station
        root = district.mv_grid.station()

        ###################################
        # get impedance of path to each terminal node
        # and get thermal capacity of first segment of path to each terminal node

        # store properties of terminal nodes in dictionaries
        # properties are e.g. impedance of path, length of path, thermal limit of first segment of path
        mv_impedances  = {}
        mvlv_impedances = {}

        mv_path_lengths = {}
        mvlv_path_lengths = {}

        mv_thermal_limits = {} # I_max of first segment on MV for each MV path
        lv_thermal_limits = {} # I_max of first segment on LV for each LV path
        mvlv_thermal_limits = {} # I_max of first segment on MV for each MVLV path

        n_outgoing_LV = 0
        n_stations_LV = 0

        n_outgoing_MV = 0

        G = district.mv_grid._graph

        for node in G.nodes_iter():
            if isinstance(node, MVStationDing0):
                n_outgoing_MV += len(G.neighbors(node))
                continue
            mv_impedance = 0
            mv_path_length = 0
            if not isinstance(node, MVCableDistributorDing0) and not isinstance(node, CircuitBreakerDing0):
                if not nx.has_path(G, root, node):
                    continue
                    #print(node, node.lv_load_area.is_aggregated) # only debug
                else:
                    path = nx.shortest_path(G, root, node)
                    for i in range(len(path)-1):
                        mv_impedance += np.sqrt(
                            (G.edge[path[i]][path[i + 1]]['branch'].type[
                                 'L'] * 1e-3 * omega * \
                             G.edge[path[i]][path[i + 1]][
                                 'branch'].length) ** 2. + \
                            (G.edge[path[i]][path[i + 1]]['branch'].type[
                                 'R'] * \
                             G.edge[path[i]][path[i + 1]][
                                 'branch'].length) ** 2.)
                        mv_path_length += G.edge[path[i]][path[i + 1]][
                            'branch'].length

                    mv_impedances[node] = mv_impedance
                    mv_path_lengths[node] = mv_path_length
                    mv_thermal_limit = G.edge[path[0]][path[1]]['branch'].type['I_max_th']
                    mv_thermal_limits[node] = mv_thermal_limit

                    if isinstance(node, LVStationDing0):
                        # add impedance of transformers in LV station
                        lvstation_impedance = 0.
                        for trafo in node.transformers():
                            lvstation_impedance += 1. / np.sqrt(trafo.x**2. + trafo.r**2.) # transformers operating in parallel
                        if lvstation_impedance > 0.: # avoid dividing by zero
                            lvstation_impedance = 1. / lvstation_impedance 
                        else:
                            lvstation_impedance = 0.
                        # identify LV nodes belonging to LV station
                        for lv_LA in district.lv_load_areas():
                            for lv_dist in lv_LA.lv_grid_districts():
                                if lv_dist.lv_grid._station == node:
                                    G_lv = lv_dist.lv_grid._graph
                                    # loop over all LV terminal nodes belonging to LV station
                                    for lv_node in G_lv.nodes():
                                        if isinstance(lv_node, GeneratorDing0) or isinstance(lv_node, LVLoadDing0):
                                            path = nx.shortest_path(G_lv, node, lv_node)
                                            lv_impedance = lvstation_impedance
                                            lv_path_length = 0.
                                            for i in range(len(path)-1):
                                                lv_impedance += np.sqrt((G_lv.edge[path[i]][path[i+1]]['branch'].type['L'] * 1e-3 * omega * \
                                                                          G_lv.edge[path[i]][path[i+1]]['branch'].length)**2. + \
                                                                         (G_lv.edge[path[i]][path[i+1]]['branch'].type['R'] * \
                                                                          G_lv.edge[path[i]][path[i+1]]['branch'].length)**2.)
                                                lv_path_length += G_lv.edge[path[i]][path[i+1]]['branch'].length
                                            lv_thermal_limit = G_lv.edge[path[0]][path[1]]['branch'].type['I_max_th']

                                            mvlv_impedances[lv_node] = mv_impedance + lv_impedance
                                            mvlv_path_lengths[lv_node] = mv_path_length + lv_path_length
                                            lv_thermal_limits[lv_node] = lv_thermal_limit
                                            mvlv_thermal_limits[lv_node] = mv_thermal_limit

                                        elif isinstance(lv_node, LVStationDing0):
                                            n_outgoing_LV += len(G_lv.neighbors(lv_node))
                                            n_stations_LV += 1

        # compute mean values by looping over terminal nodes
        sum_impedances = 0.
        sum_thermal_limits = 0.
        sum_path_lengths = 0.
        n_terminal_nodes_MV = 0

        # terminal nodes on MV
        for terminal_node in mv_impedances.keys(): # neglect LVStations here because already part of MVLV paths below
            if not isinstance(terminal_node, LVStationDing0) and not isinstance(terminal_node, MVStationDing0):
                sum_impedances += mv_impedances[terminal_node]
                sum_thermal_limits += mv_thermal_limits[terminal_node]
                sum_path_lengths += mv_path_lengths[terminal_node]
                n_terminal_nodes_MV += 1

        sum_thermal_limits_LV = 0.
        n_terminal_nodes_LV = 0

        # terminal nodes on LV
        for terminal_node in mvlv_impedances.keys():
            sum_impedances += mvlv_impedances[terminal_node]
            sum_thermal_limits += mvlv_thermal_limits[terminal_node]
            sum_thermal_limits_LV += lv_thermal_limits[terminal_node]
            sum_path_lengths += mvlv_path_lengths[terminal_node]
            n_terminal_nodes_LV += 1

        n_terminal_nodes = n_terminal_nodes_MV + n_terminal_nodes_LV

        if n_terminal_nodes < 1:
            mean_impedance = np.nan
            mean_thermal_limit = np.nan
            mean_path_length = np.nan
        else:
            mean_impedance = sum_impedances / n_terminal_nodes
            mean_thermal_limit = sum_thermal_limits / n_terminal_nodes
            mean_path_length = sum_path_lengths / n_terminal_nodes
        if n_terminal_nodes_LV < 1:
            mean_thermal_limit_LV = np.nan
        else:
            mean_thermal_limit_LV = sum_thermal_limits_LV / n_terminal_nodes_LV
        number_outgoing_LV = n_outgoing_LV # / n_stations_LV
        number_outgoing_MV = n_outgoing_MV

        ###################################
        # compute path lengths (written by Miguel)
        max_mv_path = 0
        max_mvlv_path = 0

        # rings
        nodes_in_rings = []
        branches_in_rings = []
        for ring in district.mv_grid.rings_full_data():
            ring_idx+=1

            #generation cap
            ring_gen = 0
            for node in ring[2]:
                nodes_in_rings.append(node)
                if isinstance(node,GeneratorDing0):
                    ring_gen+=node.capacity
            #length
            ring_length = 0
            for branch in ring[1]:
                branches_in_rings.append(branch)
                ring_length+=branch.length/ 1e3

            ring_dict[ring_idx] = {
                'grid_id': district.mv_grid.id_db,
                'ring_length': ring_length,
                'ring_capacity': ring_gen,
            }

        # transformers in main station
        for trafo in district.mv_grid.station().transformers():
            trafos_idx+=1
            trafos_dict[trafos_idx] = {
                'grid_id':district.mv_grid.id_db,
                's_max_a':trafo.s_max_a}

        # Generators and other MV special nodes
        cd_count = 0
        LVs_count = 0
        cb_count =  0
        lv_trafo_count = 0
        lv_trafo_cap = 0

        for node in district.mv_grid._graph.nodes():
            mv_path_length = 0
            mvlv_path_length = 0

            if isinstance(node, GeneratorDing0):
                gen_idx+=1
                isolation = not node in nodes_in_rings
                subtype = node.subtype
                if subtype==None:
                    subtype = 'other'
                generators_dict[gen_idx] = {
                    'grid_id':district.mv_grid.id_db,
                    'type':node.type,
                    'sub_type':node.type+'/'+subtype,
                    'gen_cap':node.capacity,
                    'v_level':node.v_level,
                    'isolation': isolation,
                }
                mv_path_length = district.mv_grid.graph_path_length(
                    node_source=root,
                    node_target=node)

            elif isinstance(node, MVCableDistributorDing0):
                cd_count+=1
            elif isinstance(node, LVStationDing0):
                LVs_count+=1
                lv_trafo_count += len([trafo for trafo in node.transformers()])
                lv_trafo_cap += np.sum([trafo.s_max_a for trafo in node.transformers()])

                if not node.lv_load_area.is_aggregated:
                    mv_path_length = district.mv_grid.graph_path_length(
                        node_source=root,
                        node_target=node)
                    max_lv_path = 0
                    for lv_LA in district.lv_load_areas():
                        for lv_dist in lv_LA.lv_grid_districts():
                            if lv_dist.lv_grid._station == node:
                                for lv_node in lv_dist.lv_grid._graph.nodes():
                                    lv_path_length = lv_dist.lv_grid.graph_path_length(
                                        node_source=node,
                                        node_target=lv_node)
                                    max_lv_path = max(max_lv_path,lv_path_length)
                    mvlv_path_length = mv_path_length + max_lv_path

            elif isinstance(node, CircuitBreakerDing0):
                cb_count+=1

            max_mv_path = max(max_mv_path,mv_path_length/1000)
            max_mvlv_path = max(max_mvlv_path,mvlv_path_length/1000)

        other_nodes_dict[district.mv_grid.id_db] = {
            'CD_count':cd_count,
            'LV_count':LVs_count,
            'CB_count':cb_count,
            'MVLV_trafo_count':lv_trafo_count,
            'MVLV_trafo_cap':lv_trafo_cap,
            'max_mv_path':max_mv_path,
            'max_mvlv_path':max_mvlv_path,
            'mean_impedance' : mean_impedance,
            'mean_thermal_limit' : mean_thermal_limit,
            'mean_thermal_limit_LV' : mean_thermal_limit_LV,
            'mean_path_length' : mean_path_length / 1.e3,
            'number_outgoing_LV' : number_outgoing_LV,
            'number_outgoing_MV' : number_outgoing_MV
        }

        # branches
        for branch in district.mv_grid.graph_edges():
            branch_idx+=1
            br_in_ring = branch['branch'] in branches_in_rings
            branches_dict[branch_idx] = {
                'grid_id':district.mv_grid.id_db,
                'length': branch['branch'].length / 1e3,
                'type_name': branch['branch'].type['name'],
                'type_kind': branch['branch'].kind,
                'in_ring': br_in_ring,
            }
        # Load Areas
        for LA in district.lv_load_areas():
            LA_idx += 1
            LA_dict[LA_idx] = {
                'grid_id':district.mv_grid.id_db,
                'is_agg': LA.is_aggregated,
                'is_sat': LA.is_satellite,
                #'peak_gen':LA.peak_generation,
            }
            LA_pop = 0
            residential_peak_load = 0
            retail_peak_load = 0
            industrial_peak_load = 0
            agricultural_peak_load = 0
            lv_gen_level_6 = 0
            lv_gen_level_7 = 0
            for lv_district in LA.lv_grid_districts():
                LA_pop =+ lv_district.population
                residential_peak_load += lv_district.peak_load_residential
                retail_peak_load += lv_district.peak_load_retail
                industrial_peak_load += lv_district.peak_load_industrial
                agricultural_peak_load += lv_district.peak_load_agricultural

                #generation capacity
                for g in lv_district.lv_grid.generators():
                    if g.v_level == 6:
                        lv_gen_level_6 += g.capacity
                    elif g.v_level == 7:
                        lv_gen_level_7 += g.capacity

                #branches lengths
                for br in lv_district.lv_grid.graph_edges():
                    lv_branches_idx+=1
                    lv_branches_dict[lv_branches_idx] = {
                        'grid_id':district.mv_grid.id_db,
                        'length': br['branch'].length / 1e3,
                        'type_name': br['branch'].type.to_frame().columns[0], #why is it different as for MV grids?
                        'type_kind': br['branch'].kind,
                    }

            LA_dict[LA_idx].update({
                'population': LA_pop,
                'residential_peak_load': residential_peak_load,
                'retail_peak_load': retail_peak_load,
                'industrial_peak_load': industrial_peak_load,
                'agricultural_peak_load': agricultural_peak_load,
                'total_peak_load' : residential_peak_load + retail_peak_load + \
                                    industrial_peak_load + agricultural_peak_load,
                'lv_generation': lv_gen_level_6 + lv_gen_level_7,
                'lv_gens_lvl_6': lv_gen_level_6,
                'lv_gens_lvl_7': lv_gen_level_7,
            })

        # geographic
        #  ETRS (equidistant) to WGS84 (conformal) projection
        proj = partial(
            pyproj.transform,
            #pyproj.Proj(init='epsg:3035'),  # source coordinate system
            #pyproj.Proj(init='epsg:4326'))  # destination coordinate system
            pyproj.Proj(init='epsg:4326'),  # source coordinate system
            pyproj.Proj(init='epsg:3035'))  # destination coordinate system
        district_geo = transform(proj, district.geo_data)
        other_nodes_dict[district.mv_grid.id_db].update({'Dist_area': district_geo.area})

    mvgd_stats = pd.DataFrame.from_dict({}, orient='index')
    ###################################
    #built dataframes from dictionaries
    trafos_df = pd.DataFrame.from_dict(trafos_dict, orient='index')
    generators_df = pd.DataFrame.from_dict(generators_dict, orient='index')
    other_nodes_df = pd.DataFrame.from_dict(other_nodes_dict, orient='index')
    branches_df = pd.DataFrame.from_dict(branches_dict, orient='index')
    lv_branches_df = pd.DataFrame.from_dict(lv_branches_dict, orient='index')
    ring_df = pd.DataFrame.from_dict(ring_dict, orient='index')
    LA_df = pd.DataFrame.from_dict(LA_dict, orient='index')

    ###################################
    #Aggregated data HV/MV Trafos
    if not trafos_df.empty:
        mvgd_stats = pd.concat([mvgd_stats, trafos_df.groupby('grid_id').count()['s_max_a']], axis=1)
        mvgd_stats = pd.concat([mvgd_stats, trafos_df.groupby('grid_id').sum()[['s_max_a']]], axis=1)
        mvgd_stats.columns = ['N° of HV/MV Trafos','Trafos HV/MV Acc s_max_a']

    ###################################
    #Aggregated data Generators
    if not generators_df.empty:
        #MV generation per sub_type
        mv_generation = generators_df.groupby(['grid_id', 'sub_type'])['gen_cap'].sum().to_frame().unstack(level=-1)
        mv_generation.columns = ['Gen. Cap. of MV '+_[1] if isinstance(_, tuple) else _
                             for _ in mv_generation.columns]
        mvgd_stats = pd.concat([mvgd_stats, mv_generation], axis=1)

        #MV generation at V levels
        mv_generation = generators_df.groupby(
            ['grid_id', 'v_level'])['gen_cap'].sum().to_frame().unstack(level=-1)
        mv_generation.columns = ['Gen. Cap. of MV at v_level '+str(_[1])
                             if isinstance(_, tuple) else _
                             for _ in mv_generation.columns]
        mvgd_stats = pd.concat([mvgd_stats, mv_generation], axis=1)
        #Isolated generators
        mv_generation = generators_df[generators_df['isolation']].groupby(
            ['grid_id'])['gen_cap'].count().to_frame()#.unstack(level=-1)
        mv_generation.columns = ['N° of isolated MV Generators']
        mvgd_stats = pd.concat([mvgd_stats, mv_generation], axis=1)

    ###################################
    #Aggregated data of other nodes
    if not other_nodes_df.empty:
        #print(other_nodes_df['CD_count'].to_frame())
        mvgd_stats['N° of Cable Distr'] = other_nodes_df['CD_count'].to_frame().astype(int)
        mvgd_stats['N° of LV Stations'] = other_nodes_df['LV_count'].to_frame().astype(int)
        mvgd_stats['N° of Circuit Breakers'] = other_nodes_df['CB_count'].to_frame().astype(int)
        mvgd_stats['District Area'] = other_nodes_df['Dist_area'].to_frame()
        mvgd_stats['N° of MV/LV Trafos'] = other_nodes_df['MVLV_trafo_count'].to_frame().astype(int)
        mvgd_stats['Trafos MV/LV Acc s_max_a'] = other_nodes_df['MVLV_trafo_cap'].to_frame()
        mvgd_stats['Length of MV max path'] = other_nodes_df['max_mv_path'].to_frame()
        mvgd_stats['Length of MVLV max path'] = other_nodes_df['max_mvlv_path'].to_frame()
        mvgd_stats['Impedance Z of path to terminal node (mean value)'] = \
                                            other_nodes_df['mean_impedance'].to_frame()
        mvgd_stats['I_max of first segment of path from MV station to terminal node (mean value)'] = \
                                            other_nodes_df['mean_thermal_limit'].to_frame()
        mvgd_stats['I_max of first segment of path from LV station to terminal node (mean value)'] = \
                                            other_nodes_df['mean_thermal_limit_LV'].to_frame()
        mvgd_stats['Length of path from MV station to terminal node (mean value)'] = \
                                            other_nodes_df['mean_path_length'].to_frame()
        mvgd_stats['Number of lines and cables going out from LV stations'] = \
                                            other_nodes_df['number_outgoing_LV'].to_frame()
        mvgd_stats['Number of lines and cables going out from MV stations'] = \
                                            other_nodes_df['number_outgoing_MV'].to_frame()

    ###################################
    #Aggregated data of MV Branches
    if not branches_df.empty:
        #km of underground cable
        branches_data = branches_df[branches_df['type_kind']=='cable'].groupby(
            ['grid_id'])['length'].sum().to_frame()
        branches_data.columns = ['Length of MV underground cables']
        mvgd_stats = pd.concat([mvgd_stats, branches_data], axis=1)

        #km of overhead lines
        branches_data = branches_df[branches_df['type_kind']=='line'].groupby(
            ['grid_id'])['length'].sum().to_frame()
        branches_data.columns = ['Length of MV overhead lines']
        mvgd_stats = pd.concat([mvgd_stats, branches_data], axis=1)

        #km of different wire types
        branches_data = branches_df.groupby(
            ['grid_id', 'type_name'])['length'].sum().to_frame().unstack(level=-1)
        branches_data.columns = ['Length of MV type '+_[1] if isinstance(_, tuple) else _
                             for _ in branches_data.columns]
        mvgd_stats = pd.concat([mvgd_stats, branches_data], axis=1)

        #branches not in ring
        total_br = branches_df.groupby(['grid_id'])['length'].count().to_frame()
        ring_br = branches_df[branches_df['in_ring']].groupby(
            ['grid_id'])['length'].count().to_frame()
        branches_data = total_br - ring_br
        total_br.columns = ['N° of MV branches']
        mvgd_stats = pd.concat([mvgd_stats, total_br], axis=1)
        branches_data.columns = ['N° of MV branches not in a ring']
        mvgd_stats = pd.concat([mvgd_stats, branches_data], axis=1)

    ###################################
    #Aggregated data of LV Branches
    if not lv_branches_df.empty:
        #km of underground cable
        lv_branches_data = lv_branches_df[lv_branches_df['type_kind']=='cable'].groupby(
            ['grid_id'])['length'].sum().to_frame()
        lv_branches_data.columns = ['Length of LV underground cables']
        mvgd_stats = pd.concat([mvgd_stats, lv_branches_data], axis=1)

        #km of overhead lines
        lv_branches_data = lv_branches_df[lv_branches_df['type_kind']=='line'].groupby(
            ['grid_id'])['length'].sum().to_frame()
        lv_branches_data.columns = ['Length of LV overhead lines']
        mvgd_stats = pd.concat([mvgd_stats, lv_branches_data], axis=1)

        #km of different wire types
        lv_branches_data = lv_branches_df.groupby(
            ['grid_id', 'type_name'])['length'].sum().to_frame().unstack(level=-1)
        lv_branches_data.columns = ['Length of LV type '+_[1] if isinstance(_, tuple) else _
                             for _ in lv_branches_data.columns]
        mvgd_stats = pd.concat([mvgd_stats, lv_branches_data], axis=1)

        #n° of branches
        total_lv_br = lv_branches_df.groupby(['grid_id'])['length'].count().to_frame()
        total_lv_br.columns = ['N° of LV branches']
        mvgd_stats = pd.concat([mvgd_stats, total_lv_br], axis=1)


    ###################################
    #Aggregated data of Rings
    if not ring_df.empty:
        #N° of rings
        ring_data = ring_df.groupby(['grid_id'])['grid_id'].count().to_frame()
        ring_data.columns = ['N° of MV Rings']
        mvgd_stats = pd.concat([mvgd_stats, ring_data], axis=1)

        #min,max,mean km of all rings
        ring_data = ring_df.groupby(['grid_id'])['ring_length'].min().to_frame()
        ring_data.columns = ['Length of MV Ring min']
        mvgd_stats = pd.concat([mvgd_stats, ring_data], axis=1)
        ring_data = ring_df.groupby(['grid_id'])['ring_length'].max().to_frame()
        ring_data.columns = ['Length of MV Ring max']
        mvgd_stats = pd.concat([mvgd_stats, ring_data], axis=1)
        ring_data = ring_df.groupby(['grid_id'])['ring_length'].mean().to_frame()
        ring_data.columns = ['Length of MV Ring mean']
        mvgd_stats = pd.concat([mvgd_stats, ring_data], axis=1)

        #km of all rings
        ring_data = ring_df.groupby(['grid_id'])['ring_length'].sum().to_frame()
        ring_data.columns = ['Length of MV Rings total']
        mvgd_stats = pd.concat([mvgd_stats, ring_data], axis=1)

        #km of non-ring
        non_ring_data = branches_df.groupby(['grid_id'])['length'].sum().to_frame()
        non_ring_data.columns = ['Length of MV Rings total']
        ring_data  = non_ring_data - ring_data
        ring_data.columns = ['Length of MV Non-Rings total']
        mvgd_stats = pd.concat([mvgd_stats, ring_data.round(1).abs()], axis=1)

        #rings generation capacity
        ring_data = ring_df.groupby(['grid_id'])['ring_capacity'].sum().to_frame()
        ring_data.columns = ['Gen. Cap. Connected to MV Rings']
        mvgd_stats = pd.concat([mvgd_stats, ring_data], axis=1)
    ###################################
    #Aggregated data of Load Areas
    if not LA_df.empty:
        LA_data = LA_df.groupby(['grid_id'])['population'].count().to_frame()
        LA_data.columns = ['N° of Load Areas']

        mvgd_stats = pd.concat([mvgd_stats, LA_data], axis=1)

        LA_data = LA_df.groupby(['grid_id'])['population',
                                             'residential_peak_load',
                                             'retail_peak_load',
                                             'industrial_peak_load',
                                             'agricultural_peak_load',
                                             'total_peak_load',
                                             'lv_generation',
                                             'lv_gens_lvl_6',
                                             'lv_gens_lvl_7'
                                            ].sum()
        LA_data.columns = ['LA Total Population',
                           'LA Total LV Peak Load Residential',
                           'LA Total LV Peak Load Retail',
                           'LA Total LV Peak Load Industrial',
                           'LA Total LV Peak Load Agricultural',
                           'LA Total LV Peak Load total',
                           'LA Total LV Gen. Cap.',
                           'Gen. Cap. of LV at v_level 6',
                           'Gen. Cap. of LV at v_level 7',
                           ]
        mvgd_stats = pd.concat([mvgd_stats, LA_data], axis=1)

    ###################################
    #Aggregated data of Aggregated Load Areas
    if not LA_df.empty:
        agg_LA_data = LA_df[LA_df['is_agg']].groupby(
            ['grid_id'])['population'].count().to_frame()
        agg_LA_data.columns = ['N° of Load Areas - Aggregated']
        mvgd_stats = pd.concat([mvgd_stats, agg_LA_data], axis=1)

        sat_LA_data = LA_df[LA_df['is_sat']].groupby(
            ['grid_id'])['population'].count().to_frame()
        sat_LA_data.columns = ['N° of Load Areas - Satellite']
        mvgd_stats = pd.concat([mvgd_stats, sat_LA_data], axis=1)

        agg_LA_data = LA_df[LA_df['is_agg']].groupby(['grid_id'])['population',
                                                              'lv_generation', 'total_peak_load'
                                                             ].sum()
        agg_LA_data.columns = ['LA Aggregated Population',
                                             'LA Aggregated LV Gen. Cap.', 'LA Aggregated LV Peak Load total'
                                            ]
        mvgd_stats = pd.concat([mvgd_stats, agg_LA_data], axis=1)

    ###################################
    mvgd_stats=mvgd_stats.fillna(0)
    mvgd_stats = mvgd_stats[sorted(mvgd_stats.columns.tolist())]
    return mvgd_stats


####################################################
def calculate_mvgd_voltage_current_stats(nw):
    """
    MV Voltage and Current Statistics for an arbitrary network

    Parameters
    ----------
    nw: :any:`list` of NetworkDing0
        The MV grid(s) to be studied

    Returns
    -------
    pandas.DataFrame
        nodes_df : Dataframe containing voltage statistics for every node in the MVGD
    pandas.DataFrame
        edges_df : Dataframe containing voltage statistics for every edge in the MVGD
    """
    ##############################
    #close circuit breakers
    nw.control_circuit_breakers(mode='close')
    ##############################
    nodes_idx = 0
    nodes_dict = {}
    branches_idx = 0
    branches_dict = {}
    for district in nw.mv_grid_districts():
        #nodes voltage
        for node in district.mv_grid.graph_nodes_sorted():
            nodes_idx+=1
            if hasattr(node, 'voltage_res'):
                Vres0 = node.voltage_res[0]
                Vres1 = node.voltage_res[1]
            else:
                Vres0 = 'Not available'
                Vres1 = 'Not available'
            nodes_dict[nodes_idx] = {
                    'MV_grid_id': district.mv_grid.id_db,
                    'node id': node.__repr__(),
                    'V_res_0': Vres0,
                    'V_res_1': Vres1,
                    'V nominal': district.mv_grid.v_level,
                }
        # branches currents
        for branch in district.mv_grid.graph_edges():
            branches_idx+=1
            if hasattr(branch['branch'], 's_res'):
                s_res0 = branch['branch'].s_res[0]
                s_res1 = branch['branch'].s_res[1]
            else:
                s_res0 = 'Not available'
                s_res1 = 'Not available'

            branches_dict[branches_idx] = {
                    'MV_grid_id': district.mv_grid.id_db,
                    'branch id': branch['branch'].__repr__(), #.id_db
                    's_res_0': s_res0,
                    's_res_1': s_res1,
                    #'length': branch['branch'].length / 1e3,
                }
    nodes_df = pd.DataFrame.from_dict(nodes_dict, orient='index')
    branches_df = pd.DataFrame.from_dict(branches_dict, orient='index')

    if not nodes_df.empty:
        nodes_df = nodes_df.set_index('node id')
        nodes_df = nodes_df.fillna(0)
        nodes_df = nodes_df[sorted(nodes_df.columns.tolist())]
        nodes_df.sort_index(inplace=True)

    if not branches_df.empty:
        branches_df = branches_df.set_index('branch id')
        branches_df = branches_df.fillna(0)
        branches_df = branches_df[sorted(branches_df.columns.tolist())]
        branches_df.sort_index(inplace=True)

    return(nodes_df, branches_df)

####################################################
def calculate_lvgd_voltage_current_stats(nw):
    """
    LV Voltage and Current Statistics for an arbitrary network

    Note
    ----
    Aggregated Load Areas are excluded.

    Parameters
    ----------
    nw: :any:`list` of NetworkDing0
        The MV grid(s) to be studied

    Returns
    -------
    pandas.DataFrame
        nodes_df : Dataframe containing voltage, respectively current, statis
        for every critical node, resp. every critical station, in every LV grid
        in nw.
    pandas.DataFrame
        edges_df : Dataframe containing current statistics for every critical
        edge,  in every LV grid in nw.
    """
    ##############################
    #close circuit breakers
    nw.control_circuit_breakers(mode='close')
    ##############################
    nodes_idx = 0
    nodes_dict = {}
    branches_idx = 0
    branches_dict = {}
    for mv_district in nw.mv_grid_districts():
        for LA in mv_district.lv_load_areas():
            if not LA.is_aggregated:
                for lv_district in LA.lv_grid_districts():
                    #nodes voltage
                    crit_nodes = get_critical_voltage_at_nodes(lv_district.lv_grid)
                    for node in crit_nodes:
                        nodes_idx+=1
                        nodes_dict[nodes_idx] = {
                            'MV_grid_id': mv_district.mv_grid.id_db,
                            'LV_grid_id': lv_district.lv_grid.id_db,
                            'LA_id':LA.id_db,
                            'node id': node['node'].__repr__(),
                            'v_diff_0': node['v_diff'][0],
                            'v_diff_1': node['v_diff'][1],
                            's_max_0': 'NA',
                            's_max_1': 'NA',
                            'V nominal': lv_district.lv_grid.v_level,
                        }
                    # branches currents
                    critical_branches, critical_stations = get_critical_line_loading(lv_district.lv_grid)
                    for branch in critical_branches:
                        branches_idx += 1
                        branches_dict[branches_idx] = {
                            'MV_grid_id': mv_district.mv_grid.id_db,
                            'LV_grid_id': lv_district.lv_grid.id_db,
                            'LA_id':LA.id_db,
                            'branch id': branch['branch'].__repr__(),
                            's_max_0': branch['s_max'][0],
                            's_max_1': branch['s_max'][1],
                        }
                    # stations
                    for node in critical_stations:
                        nodes_idx+=1
                        nodes_dict[nodes_idx] = {
                            'MV_grid_id': mv_district.mv_grid.id_db,
                            'LV_grid_id': lv_district.lv_grid.id_db,
                            'LA_id':LA.id_db,
                            'node id': node['station'].__repr__(),
                            's_max_0': node['s_max'][0],
                            's_max_1': node['s_max'][1],
                            'v_diff_0': 'NA',
                            'v_diff_1': 'NA',
                        }
    nodes_df = pd.DataFrame.from_dict(nodes_dict, orient='index')
    branches_df = pd.DataFrame.from_dict(branches_dict, orient='index')

    if not nodes_df.empty:
        nodes_df = nodes_df.set_index('node id')
        nodes_df = nodes_df.fillna(0)
        nodes_df = nodes_df[sorted(nodes_df.columns.tolist())]
        nodes_df.sort_index(inplace=True)

    if not branches_df.empty:
        branches_df = branches_df.set_index('branch id')
        branches_df = branches_df.fillna(0)
        branches_df = branches_df[sorted(branches_df.columns.tolist())]
        branches_df.sort_index(inplace=True)

    return nodes_df, branches_df

########################################################
def init_mv_grid(mv_grid_districts=[3545], filename='ding0_tests_grids_1.pkl'):
    '''Runs ding0 over the districtis selected in mv_grid_districts

    It also writes the result in filename. If filename = False,
    then the network is not saved.

    Parameters
    ----------
    mv_grid_districts: :any:`list` of :obj:`int`
        Districts IDs: Defaults to [3545]
    filename: str
        Defaults to 'ding0_tests_grids_1.pkl'
        If filename=False, then the network is not saved

    Returns
    -------
    NetworkDing0
        The created MV network.

    '''
    print('\n########################################')
    print('  Running ding0 for district', mv_grid_districts)
    # database connection/ session
    engine = db.connection(section='oedb')
    session = sessionmaker(bind=engine)()

    # instantiate new ding0 network object
    nd = NetworkDing0(name='network')

    # run DINGO on selected MV Grid District
    nd.run_ding0(session=session, mv_grid_districts_no=mv_grid_districts)

    # export grid to file (pickle)
    if filename:
        print('\n########################################')
        print('  Saving result in ', filename)
        save_nd_to_pickle(nd, filename=filename)

    print('\n########################################')
    return nd

########################################################
def process_stats(mv_districts,
                  n_of_districts,
                  source,
                  mode,
                  critical,
                  filename,
                  output):
    '''Generates stats dataframes for districts in mv_districts.

    If source=='ding0', then runned districts are saved to a pickle named
    filename+str(n_of_districts[0])+'_to_'+str(n_of_districts[-1])+'.pkl'

    Parameters
    ----------
    districts_list: list of int
        List with all districts to be run.
    n_of_districts: int
        Number of districts to be run in each cluster
    source: str
        If 'pkl', pickle files are read.
        If 'ding0', ding0 is run over the districts.
    mode: str
        If 'MV', medium voltage stats are calculated.
        If 'LV', low voltage stats are calculated.
        If empty, medium and low voltage stats are calculated.
    critical: bool
        If True, critical nodes and branches are returned
    filename: str
        filename prefix for saving pickles
    output:
        outer variable where the output is stored as a tuple of 6 lists::

        * mv_stats: MV stats DataFrames.
          If mode=='LV', then DataFrame is empty.

        * lv_stats: LV stats DataFrames.
          If mode=='MV', then DataFrame is empty.

        * mv_crit_nodes: MV critical nodes stats DataFrames.
          If mode=='LV', then DataFrame is empty.
          If critical==False, then DataFrame is empty.

        * mv_crit_edges: MV critical edges stats DataFrames.
          If mode=='LV', then DataFrame is empty.
          If critical==False, then DataFrame is empty.

        * lv_crit_nodes: LV critical nodes stats DataFrames.
          If mode=='MV', then DataFrame is empty.
          If critical==False, then DataFrame is empty.

        * lv_crit_edges: LV critical edges stats DataFrames.
          If mode=='MV', then DataFrame is empty.
          If critical==False, then DataFrame is empty.
    '''
    #######################################################################
    # decide what exactly to do with MV LV
    if mode == 'MV':
        calc_mv = True
        calc_lv = False
    elif mode == 'LV':
        calc_mv = False
        calc_lv = True
    else:
        calc_mv = True
        calc_lv = True
    #######################################################################
    clusters = [mv_districts[x:x + n_of_districts] for x in range(0, len(mv_districts), n_of_districts)]

    mv_stats      = []
    lv_stats      = []
    mv_crit_nodes = []
    mv_crit_edges = []
    lv_crit_nodes = []
    lv_crit_edges = []
    #######################################################################
    for cl in clusters:
        nw_name = filename + str(cl[0])
        if not cl[0] == cl[-1]:
            nw_name = nw_name + '_to_' + str(cl[-1])

        nw = NetworkDing0(name=nw_name)
        if source == 'pkl':
            print('\n########################################')
            print('  Reading data from pickle district', cl)
            print('########################################')
            try:
                nw = load_nd_from_pickle(nw_name+'.pkl')
            except Exception:
                continue
        else:
            # database connection/ session
            engine = db.connection(section='oedb')
            session = sessionmaker(bind=engine)()

            print('\n########################################')
            print('  Running ding0 for district', cl)
            print('########################################')
            try:
                nw.run_ding0(session=session, mv_grid_districts_no=cl)
                try:
                    save_nd_to_pickle(nw, filename=nw_name+'.pkl')
                except Exception:
                    continue
            except Exception:
                continue

            # Close database connection
        if calc_mv:
            stats = calculate_mvgd_stats(nw)
            mv_stats.append(stats)
        if calc_lv:
            stats = calculate_lvgd_stats(nw)
            lv_stats.append(stats)
        if critical and calc_mv:
            stats = calculate_mvgd_voltage_current_stats(nw)
            mv_crit_nodes.append(stats[0])
            mv_crit_edges.append(stats[1])
        if critical and calc_lv:
            stats = calculate_lvgd_voltage_current_stats(nw)
            lv_crit_nodes.append(stats[0])
            lv_crit_edges.append(stats[1])
    #######################################################################
    salida = (mv_stats,lv_stats,mv_crit_nodes,mv_crit_edges,lv_crit_nodes,lv_crit_edges)
    output.put(salida)

def parallel_running_stats(districts_list,
                           n_of_processes,
                           n_of_districts=1,
                           source='pkl',
                           mode='',
                           critical = False,
                           save_csv = False,
                           save_path = ''):
    '''Organize parallel runs of ding0 to calculate stats

    The function take all districts in a list and divide them into
    n_of_processes parallel processes. For each process, the assigned districts
    are given to the function process_runs() with arguments n_of_districts,
    source, mode, and critical

    Parameters
    ----------
    districts_list: list of int
        List with all districts to be run.
    n_of_processes: int
        Number of processes to run in parallel
    n_of_districts: int
        Number of districts to be run in each cluster given as argument to
        process_stats()
    source: str
        If 'pkl', pickle files are read. Otherwise, ding0 is run over the districts.
    mode: str
        If 'MV', medium voltage stats are calculated.
        If 'LV', low voltage stats are calculated.
        If empty, medium and low voltage stats are calculated.
    critical: bool
        If True, critical nodes and branches are returned
    path: str
        path to save the pkl and csv files

    Returns
    -------
    DataFrame
        mv_stats: MV stats in a DataFrame.
        If mode=='LV', then DataFrame is empty.
    DataFrame
        lv_stats: LV stats in a DataFrame.
        If mode=='MV', then DataFrame is empty.
    DataFrame
        mv_crit_nodes: MV critical nodes stats in a DataFrame.
        If mode=='LV', then DataFrame is empty.
        If critical==False, then DataFrame is empty.
    DataFrame
        mv_crit_edges: MV critical edges stats in a DataFrame.
        If mode=='LV', then DataFrame is empty.
        If critical==False, then DataFrame is empty.
    DataFrame
        lv_crit_nodes: LV critical nodes stats in a DataFrame.
        If mode=='MV', then DataFrame is empty.
        If critical==False, then DataFrame is empty.
    DataFrame
        lv_crit_edges: LV critical edges stats in a DataFrame.
        If mode=='MV', then DataFrame is empty.
        If critical==False, then DataFrame is empty.

    See Also
    --------
    process_stats
    '''
    start = time.time()

    nw_name = os.path.join(save_path, 'ding0_grids__') #name of files prefix

    #######################################################################
    # Define an output queue
    output_stats = mp.Queue()
    #######################################################################
    # Setup a list of processes that we want to run
    max_dist = len(districts_list)
    threat_long = floor(max_dist / n_of_processes)

    if threat_long == 0:
        threat_long = 1

    threats = [districts_list[x:x + threat_long] for x in
               range(0, len(districts_list), threat_long)]

    processes = []
    for districts in threats:
        args = (districts, n_of_districts, source, mode, critical, nw_name, output_stats)
        processes.append(mp.Process(target=process_stats, args=args))
    #######################################################################
    # Run processes
    for p in processes:
        p.start()
    # Resque output_stats from processes
    output = [output_stats.get() for p in processes]
    # Exit the completed processes
    for p in processes:
        p.join()
    #######################################################################
    #create outputs
    # Name of files
    if save_csv:
        nw_name = nw_name + str(districts_list[0])
        if not districts_list[0] == districts_list[-1]:
            nw_name = nw_name + '_to_' + str(districts_list[-1])

    #concatenate all dataframes
    try:
        mv_stats = pd.concat(
            [df for p in range(0, len(processes)) for df in output[p][0]],
            axis=0)
    except:
        mv_stats = pd.DataFrame.from_dict({})
    try:
        lv_stats = pd.concat(
            [df for p in range(0, len(processes)) for df in output[p][1]],
             axis=0)
    except:
        lv_stats = pd.DataFrame.from_dict({})
    try:
        mv_crit_nodes = pd.concat(
            [df for p in range(0, len(processes)) for df in output[p][2]],
            axis=0)
    except:
        mv_crit_nodes = pd.DataFrame.from_dict({})
    try:
        mv_crit_edges = pd.concat(
            [df for p in range(0, len(processes)) for df in output[p][3]],
            axis=0)
    except:
        mv_crit_edges = pd.DataFrame.from_dict({})
    try:
        lv_crit_nodes = pd.concat(
            [df for p in range(0, len(processes)) for df in output[p][4]],
            axis=0)
    except:
        lv_crit_nodes = pd.DataFrame.from_dict({})
    try:
        lv_crit_edges = pd.concat(
            [df for p in range(0, len(processes)) for df in output[p][5]],
            axis=0)
    except:
        lv_crit_edges = pd.DataFrame.from_dict({})

    # format concatenated Dataframes
    if not mv_stats.empty:
        mv_stats = mv_stats.fillna(0)
        mv_stats = mv_stats[sorted(mv_stats.columns.tolist())]
        mv_stats.sort_index(inplace=True)
        if save_csv:
            mv_stats.to_csv(nw_name+ '_mv_stats.csv')

    if not lv_stats.empty:
        lv_stats = lv_stats.fillna(0)
        lv_stats = lv_stats[sorted(lv_stats.columns.tolist())]
        lv_stats.sort_index(inplace=True)
        if save_csv:
            lv_stats.to_csv(nw_name+ '_lv_stats.csv')

    if not mv_crit_nodes.empty:
        mv_crit_nodes = mv_crit_nodes.fillna(0)
        mv_crit_nodes = mv_crit_nodes[sorted(mv_crit_nodes.columns.tolist())]
        mv_crit_nodes.sort_index(inplace=True)
        if save_csv:
            mv_crit_nodes.to_csv(nw_name + '_mv_crit_nodes.csv')

    if not mv_crit_edges.empty:
        mv_crit_edges = mv_crit_edges.fillna(0)
        mv_crit_edges = mv_crit_edges[sorted(mv_crit_edges.columns.tolist())]
        mv_crit_edges.sort_index(inplace=True)
        if save_csv:
            mv_crit_edges.to_csv(nw_name + '_mv_crit_edges.csv')

    if not lv_crit_nodes.empty:
        lv_crit_nodes = lv_crit_nodes.fillna(0)
        lv_crit_nodes = lv_crit_nodes[sorted(lv_crit_nodes.columns.tolist())]
        lv_crit_nodes.sort_index(inplace=True)
        if save_csv:
            lv_crit_nodes.to_csv(nw_name + '_lv_crit_nodes.csv')

    if not lv_crit_edges.empty:
        lv_crit_edges = lv_crit_edges.fillna(0)
        lv_crit_edges = lv_crit_edges[sorted(lv_crit_edges.columns.tolist())]
        lv_crit_edges.sort_index(inplace=True)
        if save_csv:
            lv_crit_edges.to_csv(nw_name + '_lv_crit_edges.csv')

    #######################################################################
    print('\n########################################')
    print('  Elapsed time for', str(max_dist),
          'MV grid districts (seconds): {}'.format(time.time() - start))
    print('\n########################################')
    #######################################################################
    return mv_stats, lv_stats, mv_crit_nodes, mv_crit_edges, lv_crit_nodes, lv_crit_edges

########################################################
def export_network(nw, mode=''):
    """
    Export all nodes and edges of the network nw as DataFrames

    Parameters
    ----------
    nw: :any:`list` of NetworkDing0
        The MV grid(s) to be studied
    mode: str
        If 'MV' export only medium voltage nodes and edges
        If 'LV' export only low voltage nodes and edges
        else, exports MV and LV nodes and edges

    Returns
    -------
    pandas.DataFrame
        nodes_df : Dataframe containing nodes and its attributes
    pandas.DataFrame
        edges_df : Dataframe containing edges and its attributes
    """


    #close circuit breakers
    nw.control_circuit_breakers(mode='close')
    #srid
    srid = str(int(nw.config['geo']['srid']))
    ##############################
    #check what to do
    lv_info = True
    mv_info = True
    if mode=='LV':
        mv_info = False
    if mode=='MV':
        lv_info = False
    ##############################
    # from datetime import datetime
    run_id = nw.metadata['run_id']  # datetime.now().strftime("%Y%m%d%H%M%S")
    ##############################
    #############################
    #go through the grid collecting info
    lvgrid_idx = 0
    lvgrid_dict = {}
    lvloads_idx = 0
    lvloads_dict = {}
    mvgrid_idx = 0
    mvgrid_dict = {}
    mvloads_idx = 0
    mvloads_dict = {}
    mvgen_idx = 0
    mvgen_dict = {}
    mvcb_idx = 0
    mvcb_dict = {}
    mvcd_idx = 0
    mvcd_dict = {}
    mvstations_idx = 0
    mvstations_dict = {}
    mvtrafos_idx = 0
    mvtrafos_dict = {}
    lvgen_idx = 0
    lvgen_dict = {}
    lvcd_idx = 0
    lvcd_dict = {}
    lvstations_idx = 0
    lvstations_dict = {}
    lvtrafos_idx = 0
    lvtrafos_dict = {}
    areacenter_idx = 0
    areacenter_dict = {}
    edges_idx = 0
    edges_dict = {}
    LVMVmapping_idx = 0
    LVMVmapping_dict = {}

    def aggregate_generators(gen, aggr):
        """Aggregate generation capacity per voltage level
        Parameters
        ----------
        gen: ding0.core.GeneratorDing0
            Ding0 Generator object
        aggr: dict
            Aggregated generation capacity. For structure see
            `_determine_aggregated_nodes()`.
        Returns
        -------
        """

        if gen.v_level not in aggr['generation']:
            aggr['generation'][gen.v_level] = {}
        if gen.type not in aggr['generation'][gen.v_level]:
            aggr['generation'][gen.v_level][gen.type] = {}
        if gen.subtype not in aggr['generation'][gen.v_level][gen.type]:
            aggr['generation'][gen.v_level][gen.type].update(
                {gen.subtype: {'ids': [gen.id_db],
                               'capacity': gen.capacity}})
        else:
            aggr['generation'][gen.v_level][gen.type][gen.subtype][
                'ids'].append(gen.id_db)
            aggr['generation'][gen.v_level][gen.type][gen.subtype][
                'capacity'] += gen.capacity

        return aggr

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


        #for s in ['retail', 'industrial', 'agricultural', 'residential']:
            for t in ['nominal','peak']:
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
        mv_grid_id_db = '_'.join(
            [str(mv_district.mv_grid.__class__.__name__), 'MV', str(mv_grid_id), str(mv_district.mv_grid.id_db)])

        if mv_info:
            lv_grid_id = 0

            # MV-grid
            #ToDo: geom <- Polygon
            mvgrid_idx += 1
            mvgrid_dict[mvgrid_idx] = {
                'MV_grid_id': mv_district.mv_grid.id_db,
                'id_db': '_'.join([str(mv_district.mv_grid.__class__.__name__), 'MV', str(mv_grid_id),  str(mv_district.mv_grid.id_db)]),
                #'network': mv_district.mv_grid.network,
                'geom': wkt_dumps(mv_district.geo_data),
                'population': #None,
                    sum([_.zensus_sum
                         for _ in
                         mv_district._lv_load_areas #ding0_grid.grid_district._lv_load_areas
                         if not np.isnan(_.zensus_sum)]),
                'voltage_nom': mv_district.mv_grid.v_level, # in kV
                'run_id': run_id
            }

            #id_db: Classname_MV/LV_mvgridid/lvgridid_id
            #excemptions: class LVStations: LVStationDing0_MV_mvgridid_id(=lvgridid)

            #MVGrid
            for node in mv_district.mv_grid.graph_nodes_sorted():
                geom = wkt_dumps(node.geo_data)
                #geom = from_shape(Point(node.geo_data), srid=srid)
                db_id = node.id_db

                #LVStation
                if isinstance(node, LVStationDing0):
                    if not node.lv_load_area.is_aggregated:
                        lvstations_idx+=1
                        lvstations_dict[lvstations_idx] = {
                            'id_db': '_'.join([str(node.__class__.__name__), 'MV', str(mv_grid_id), str(node.id_db)]),
                            'LV_grid_id_db': '_'.join(['LVGridDing0', 'LV', str(node.id_db), str(node.id_db)]),
                            'geom':geom,
                            'run_id': run_id,
                        }

                        #LV-MV mapping
                        LVMVmapping_idx+=1
                        LVMVmapping_dict[LVMVmapping_idx] = {
                            'MV_grid_id':mv_grid_id,
                            'MV_grid_id_db': mv_grid_id_db,
                            'LV_grid_id':node.id_db,
                            'LV_grid_id_db': '_'.join(['LVGridDing0', 'LV', str(node.id_db), str(node.id_db)]),
                            'run_id': run_id,
                        }

                        #Trafos LV
                        for t in node.transformers():
                            lvtrafos_idx+=1
                            lvtrafos_dict[lvtrafos_idx] = {
                                'id_db':'_'.join([str(t.__class__.__name__), 'LV', str(mv_grid_id), str(node.id_db)]),
                                'geom':geom,
                                'LV_grid_id_db': '_'.join(['LVGridDing0', 'LV', str(node.id_db), str(node.id_db)]),
                                'voltage_op':t.v_level,
                                'S_nom':t.s_max_a,
                                'X':t.x,
                                'R':t.r,
                                'run_id': run_id,
                            }

                #MVStation
                elif isinstance(node, MVStationDing0):
                    mvstations_idx+=1
                    mvstations_dict[mvstations_idx] = {
                        'id_db': '_'.join([str(node.__class__.__name__), 'MV', str(mv_grid_id), str(node.id_db)]),
                        'MV_grid_id_db': mv_grid_id_db,
                        'geom':geom,
                        'run_id': run_id,
                    }

                    #Trafos MV
                    for t in node.transformers():
                        mvtrafos_idx+=1
                        mvtrafos_dict[mvtrafos_idx] = {
                            'id_db': '_'.join([str(t.__class__.__name__), 'MV', str(mv_grid_id), str(node.id_db)]),
                            'geom':geom,
                            'MV_grid_id_db': mv_grid_id_db,
                            'voltage_op':t.v_level,
                            'S_nom':t.s_max_a,
                            'X':t.x,
                            'R':t.r,
                            'run_id': run_id,
                        }

                #MVGenerator
                elif isinstance(node, GeneratorDing0):
                    if node.subtype==None:
                        subtype = 'other'
                    else:
                        subtype = node.subtype
                    type = node.type
                    mvgen_idx+=1
                    mvgen_dict[mvgen_idx] = {
                        'id_db': '_'.join([str(node.__class__.__name__), 'MV', str(mv_grid_id), str(node.id_db)]),
                        'MV_grid_id_db': mv_grid_id_db,
                        'geom':geom,
                        'type':type,
                        'subtype':subtype,
                        'v_level':node.v_level,
                        'nominal_capacity':node.capacity,
                        'run_id': run_id,
                        'is_aggregated': False,
                    }

                #MVBranchTees
                elif isinstance(node, MVCableDistributorDing0):
                    mvcd_idx+=1
                    mvcd_dict[mvcd_idx] = {
                        'id_db': '_'.join([str(node.__class__.__name__), 'MV', str(mv_grid_id), str(node.id_db)]),
                        'MV_grid_id_db': mv_grid_id_db,
                        'geom': geom,
                        'run_id': run_id,
                    }

                #LoadAreaCentre
                elif isinstance(node, LVLoadAreaCentreDing0):

                    #type = 'Load area center of aggregated load area'

                    areacenter_idx+=1
                    aggr_edges = 0

                    aggr = {'generation': {}, 'load': {}, 'aggregates': []}

                    # Determine aggregated generation in LV grid
                    for lvgd in node.lv_load_area._lv_grid_districts:

                        for aggr_gen in lvgd.lv_grid.generators():
                            aggr = aggregate_generators(aggr_gen, aggr)

                            if aggr_gen.subtype==None:
                                subtype = 'other'
                            else:
                                subtype = aggr_gen.subtype
                            type = aggr_gen.type

                    # Determine aggregated load in MV grid
                    # -> Implement once loads in Ding0 MV grids exist

                    # Determine aggregated load in LV grid
                    aggr = aggregate_loads(node, aggr)

                    # Collect metadata of aggregated load areas
                    aggr['aggregates'] = {
                        'population': node.lv_load_area.zensus_sum,
                        'geom': node.lv_load_area.geo_area}
                    aggr_line_type = nw._static_data['MV_cables'].iloc[
                        nw._static_data['MV_cables']['I_max_th'].idxmax()]
                    geom = wkt_dumps(node.lv_load_area.geo_area)

                    for aggr_node in aggr:
                        if aggr_node == 'generation':
                            mvgenaggr_idx = 0

                            for v_level in aggr['generation']:
                                for type in aggr['generation'][v_level]:
                                    for subtype in aggr['generation'][v_level][type]:
                                        mvgen_idx += 1
                                        mvgenaggr_idx +=1
                                        mvgen_dict[mvgen_idx] = {
                                            'id_db': '_'.join(
                                                [str(aggr_gen.__class__.__name__), 'MV', str(mv_grid_id),
                                                 str(aggr_gen.id_db), str(mvgenaggr_idx)]), #, str(mvgen_idx)
                                            'MV_grid_id_db': mv_grid_id_db,
                                            'geom': geom,#from_shape(Point(mv_district.mv_grid.station().geo_data), srid=srid),#lv_load_area.geo_area,#geom, #?? Polygon # mvstations_dict[mvstations_idx]['geom'], #
                                            'type': type,
                                            'subtype': subtype,
                                            'v_level': v_level,
                                            'nominal_capacity': aggr['generation'][v_level][type][subtype]['capacity'],
                                            'is_aggregated': True,
                                            'run_id': run_id,
                                        }

                                        edges_idx += 1
                                        aggr_edges += 1
                                        edges_dict[edges_idx] = {
                                            #ToDo: Rename edge_name
                                            'edge_name': '_'.join([str(mv_grid_id), 'aggr', str(node.lv_load_area.id_db), str(aggr_edges)]),#, 'vlevel', str(v_level), 'subtype', str(subtype)]),#}'.format(v_level=v_level, subtype=subtype),
                                            'grid_id_db': mv_grid_id_db,
                                            # ToDo: read type_name from aggr_line_type
                                            'type_name': 'NA2XS2Y 3x1x500 RM/35',#aggr_line_type.name,
                                            'type_kind': 'cable',#branch['branch'].kind,
                                            'length': 1,
                                            'U_n': aggr_line_type.U_n,
                                            'I_max_th': aggr_line_type.I_max_th,
                                            'R': aggr_line_type.R,
                                            'L': aggr_line_type.L,
                                            'C': aggr_line_type.C,
                                            'node1': '_'.join(
                                                [str(aggr_gen.__class__.__name__), 'MV', str(mv_grid_id),
                                                 str(aggr_gen.id_db), str(mvgenaggr_idx)]),
                                            'node2': '_'.join([
                                                'MVStationDing0', 'MV', str(mv_grid_id), str(mv_grid_id)]),
                                            'run_id': run_id,
                                        }

                        elif aggr_node == 'load':
                            for type in aggr['load']:
                                mvloads_idx += 1
                                mvloads_dict[mvloads_idx] = {
                                    'id_db': '_'.join(
                                        ['AggregatedLoad', 'MV', str(mv_grid_id), str(mvloads_idx)]),
                                    'MV_grid_id_db': mv_grid_id_db,
                                    'geom': geom,#from_shape(Point(mv_district.mv_grid.station().geo_data), srid=srid),
                                    'consumption': json.dumps({type: aggr['load'][type]['nominal']}),
                                    'is_aggregated': True,
                                    'run_id': run_id,
                                }

                                edges_idx += 1
                                aggr_edges += 1
                                edges_dict[edges_idx] = {
                                    # ToDo: Rename edge_name
                                    'edge_name': '_'.join(
                                        [str(mv_grid_id), 'aggr', str(node.lv_load_area.id_db), str(aggr_edges)]),
                                    #'edge_name': '_'.join(
                                    #    ['line_aggr_load', str(node.lv_load_area), 'vlevel', str(v_level),
                                    #     'subtype', str(subtype)]),  # }'.format(v_level=v_level, subtype=subtype),
                                    'grid_id_db': mv_grid_id_db,
                                    #ToDo: read type_name from aggr_line_type
                                    'type_name': 'NA2XS2Y 3x1x500 RM/35',#aggr_line_type.name,
                                    'type_kind': 'cable',  # branch['branch'].kind,
                                    #'type': aggr_line_type,
                                    'length': 1e-3, # in km
                                    'U_n': aggr_line_type.U_n,
                                    'I_max_th': aggr_line_type.I_max_th,
                                    'R': aggr_line_type.R,
                                    'L': aggr_line_type.L,
                                    'C': aggr_line_type.C,
                                    'node1': '_'.join(
                                        ['AggregatedLoad', 'MV', str(mv_grid_id), str(mvloads_idx)]),
                                    'node2': '_'.join([
                                        'MVStationDing0', 'MV', str(mv_grid_id), str(mv_grid_id)]),
                                    'run_id': run_id,
                                }


                    #areacenter_dict[areacenter_idx] = {
                    #    'id_db': '_'.join([str(node.__class__.__name__), 'MV', str(mv_grid_id), str(node.id_db)]),#node.id_db,
                    #    'MV_grid_id':node.grid,
                    #    'geom':node.geo_data,
                    #    'lv_load_area': node.lv_load_area,
                    #    'run_id': run_id,#

                     #}

                #DisconnectingPoints
                elif isinstance(node, CircuitBreakerDing0):
                    mvcb_idx+=1
                    mvcb_dict[mvcb_idx] = {
                        'id_db': '_'.join([str(node.__class__.__name__), 'MV', str(mv_grid_id), str(node.id_db)]),
                        'MV_grid_id':mv_grid_id,
                        'MV_grid_id_db': mv_grid_id_db,
                        'geom':geom,
                        'status':node.status,
                        'run_id': run_id,
                    }
                else:
                    type = 'Unknown'

            #MVedges
            for branch in mv_district.mv_grid.graph_edges():
                #geom = wkt_dumps(node.geo_data)
                geom = from_shape(LineString([branch['adj_nodes'][0].geo_data,branch['adj_nodes'][1].geo_data]),srid=srid)
                if not any([isinstance(branch['adj_nodes'][0], LVLoadAreaCentreDing0),
                            isinstance(branch['adj_nodes'][1], LVLoadAreaCentreDing0)]):
                    edges_idx +=1
                    edges_dict[edges_idx] = {
                        'edge_name': branch['branch'].id_db,
                        'grid_id_db': mv_grid_id_db,
                        'type_name': branch['branch'].type['name'],
                        'type_kind': branch['branch'].kind,
                        'length': branch['branch'].length/1e3,
                        'U_n':branch['branch'].type['U_n'],
                        'I_max_th':branch['branch'].type['I_max_th'],
                        'R': branch['branch'].type['R'],
                        'L': branch['branch'].type['L'],
                        'C': branch['branch'].type['C'],
                        'node1': '_'.join([str(branch['adj_nodes'][0].__class__.__name__), 'MV', str(mv_grid_id), str(branch['adj_nodes'][0].id_db)]),
                        'node2': '_'.join([str(branch['adj_nodes'][1].__class__.__name__), 'MV', str(mv_grid_id), str(branch['adj_nodes'][1].id_db)]),
                        'run_id': run_id,
                    }

        if lv_info:
            for LA in mv_district.lv_load_areas():
                for lv_district in LA.lv_grid_districts():
                    if not lv_district.lv_grid.grid_district.lv_load_area.is_aggregated:

                        # ding0_grid.grid_district._lv_load_areas._lv_grid_districts    _.lv_grid
                        # LV-grid
                        # ToDo: geom <- Polygon
                        lvgrid_idx += 1
                        lvgrid_dict[lvgrid_idx] = {
                            'LV_grid_id': lv_district.lv_grid.id_db,
                            'id_db': '_'.join([str(lv_district.lv_grid.__class__.__name__), 'LV', str(lv_district.lv_grid.id_db), str(lv_district.lv_grid.id_db)]),
                            'geom': wkt_dumps(lv_district.geo_data),
                            'population': lv_district.population,
                            'voltage_nom': lv_district.lv_grid.v_level/1e3,
                            'run_id': run_id
                        }

                        lv_grid_id = lv_district.lv_grid.id_db
                        lv_grid_id_db = '_'.join([str(lv_district.lv_grid.__class__.__name__), 'LV', str(lv_district.lv_grid.id_db), str(lv_district.lv_grid.id_db)])

                        #geom = from_shape(Point(lv_district.lv_grid.station().geo_data), srid=srid)
                        #geom = wkt_dumps(lv_district.geo_data)# lv_grid.station() #ding0_lv_grid.grid_district.geo_data
                        for node in lv_district.lv_grid.graph_nodes_sorted():
                            #geom = wkt_dumps(node.geo_data)

                            #LVGenerator
                            if isinstance(node, GeneratorDing0):
                                if node.subtype == None:
                                    subtype = 'other'
                                else:
                                    subtype = node.subtype
                                type = node.type
                                lvgen_idx += 1
                                lvgen_dict[lvgen_idx] = {
                                    'id_db': '_'.join([str(node.__class__.__name__), 'LV', str(lv_grid_id), str(node.id_db)]),
                                    'LV_grid_id_db': lv_grid_id_db,
                                    'geom': wkt_dumps(node.geo_data),
                                    'type': type,
                                    'subtype': subtype,
                                    'v_level': node.v_level,
                                    'nominal_capacity': node.capacity,
                                    'run_id': run_id,
                                }

                            #LVcd
                            elif isinstance(node, LVCableDistributorDing0):
                                lvcd_idx += 1
                                lvcd_dict[lvcd_idx] = {
                                    'id_db': '_'.join([str(node.__class__.__name__), 'LV', str(lv_grid_id), str(node.id_db)]),
                                    'LV_grid_id_db': lv_grid_id_db,
                                    'geom': None,#wkt_dumps(lv_district.geo_data),#wkt_dumps(node.geo_data), Todo: why no geo_data?
                                    'run_id': run_id,
                                }

                            #LVload
                            elif isinstance(node, LVLoadDing0):
                                lvloads_idx += 1
                                lvloads_dict[lvloads_idx] = {
                                    'id_db': '_'.join([str(node.__class__.__name__), 'LV', str(lv_grid_id), str(node.id_db)]),
                                    'LV_grid_id_db': lv_grid_id_db,
                                    'geom': None,#wkt_dumps(lv_district.geo_data),#wkt_dumps(node.geo_data), Todo: why no geo_data?
                                    'consumption': json.dumps(node.consumption),
                                    'run_id': run_id,
                                   }

                            else:
                                type = 'Unknown'

                        #LVedges
                        for branch in lv_district.lv_grid.graph_edges():
    #                        geom = from_shape(
    #                            LineString([branch['adj_nodes'][0].geo_data, branch['adj_nodes'][1].geo_data]), srid=srid)
                            if not any([isinstance(branch['adj_nodes'][0], LVLoadAreaCentreDing0),
                                            isinstance(branch['adj_nodes'][1], LVLoadAreaCentreDing0)]):
                                 edges_idx +=1
                                 edges_dict[edges_idx] = {
                                    'edge_name': branch['branch'].id_db,
                                    'grid_id_db': lv_grid_id_db,
                                    'type_name': branch['branch'].type.to_frame().columns[0],
                                    'type_kind': branch['branch'].kind,
                                    'length': branch['branch'].length/1e3, # length in km
                                    'U_n':branch['branch'].type['U_n']/1e3, # U_n in kV
                                    'I_max_th':branch['branch'].type['I_max_th'],
                                    'R':branch['branch'].type['R'],
                                    'L':branch['branch'].type['L'],
                                    'C':branch['branch'].type['C'],
                                    'node1': '_'.join([str(branch['adj_nodes'][0].__class__.__name__), 'LV', str(lv_grid_id),
                                                       str(branch['adj_nodes'][0].id_db)])
                                            if not isinstance(branch['adj_nodes'][0], LVStationDing0) else '_'.join([str(branch['adj_nodes'][0].__class__.__name__), 'MV', str(mv_grid_id),
                                                       str(branch['adj_nodes'][0].id_db)]),
                                    'node2': '_'.join([str(branch['adj_nodes'][1].__class__.__name__), 'LV', str(lv_grid_id),
                                                       str(branch['adj_nodes'][1].id_db)])
                                            if not isinstance(branch['adj_nodes'][1], LVStationDing0) else '_'.join([str(branch['adj_nodes'][1].__class__.__name__), 'MV', str(mv_grid_id),
                                                       str(branch['adj_nodes'][1].id_db)]),
                                    'run_id': run_id,
                                }

    lv_grid       = pd.DataFrame.from_dict(lvgrid_dict, orient='index')
    lv_gen        = pd.DataFrame.from_dict(lvgen_dict, orient='index')
    lv_cd         = pd.DataFrame.from_dict(lvcd_dict, orient='index')
    lv_stations   = pd.DataFrame.from_dict(lvstations_dict, orient='index')
    lv_trafos     = pd.DataFrame.from_dict(lvtrafos_dict, orient='index')
    lv_loads      = pd.DataFrame.from_dict(lvloads_dict, orient='index')
    mv_grid       = pd.DataFrame.from_dict(mvgrid_dict, orient='index')
    mv_gen        = pd.DataFrame.from_dict(mvgen_dict, orient='index')
    mv_cb         = pd.DataFrame.from_dict(mvcb_dict, orient='index')
    mv_cd         = pd.DataFrame.from_dict(mvcd_dict, orient='index')
    mv_stations   = pd.DataFrame.from_dict(mvstations_dict, orient='index')
    #mv_areacenter = pd.DataFrame.from_dict(areacenter_dict, orient='index')
    mv_trafos     = pd.DataFrame.from_dict(mvtrafos_dict, orient='index')
    mv_loads      = pd.DataFrame.from_dict(mvloads_dict, orient='index')
    edges      = pd.DataFrame.from_dict(edges_dict, orient='index')
    lvmv_mapping = pd.DataFrame.from_dict(LVMVmapping_dict, orient='index')

    edges = edges[sorted(edges.columns.tolist())]

    return run_id, lv_grid, lv_gen, lv_cd, lv_stations, lv_trafos, lv_loads, mv_grid, mv_gen, mv_cb, mv_cd, mv_stations, mv_trafos, mv_loads, edges, lvmv_mapping # mv_areacenter,

#######################################################

def export_data_tocsv(path, run_id, lv_grid, lv_gen, lv_cd, lv_stations, lv_trafos, lv_loads, mv_grid, mv_gen, mv_cb, mv_cd, mv_stations, mv_trafos, mv_loads, edges, mapping ):
    # Exports data to csv
    def export_network_tocsv(path, table, tablename):
        return table.to_csv(''.join([path, '/', run_id, '/', tablename, '.csv']), ';')

    export_network_tocsv(path, lv_grid, 'lv_grid')
    export_network_tocsv(path, lv_gen, 'lv_generator')
    export_network_tocsv(path, lv_cd, 'lv_branchtee')
    export_network_tocsv(path, lv_stations, 'lvmv_station')
    export_network_tocsv(path, lv_trafos, 'lv_transformer')
    export_network_tocsv(path, lv_loads, 'lv_load')
    export_network_tocsv(path, mv_grid, 'mv_grid')
    export_network_tocsv(path, mv_gen, 'mv_generator')
    export_network_tocsv(path, mv_cd, 'mv_branchtee')
    export_network_tocsv(path, mv_stations, 'mvhv_station')
    export_network_tocsv(path, mv_trafos, 'mv_transformer')
    export_network_tocsv(path, mv_cb, 'mv_circuitbreaker')
    export_network_tocsv(path, mv_loads, 'mv_load')
    export_network_tocsv(path, edges, 'line')
    export_network_tocsv(path, mapping, 'mvlv_mapping')
    #export_network_tocsv(path, areacenter, 'areacenter')

########################################################

from sqlalchemy import create_engine
from egoio.db_tables import model_draft as md


def export_network_to_oedb(session, table, tabletype):
    dataset = []
    engine = create_engine("sqlite:///myexample.db")
    if tabletype == 'edges':
        table.apply(lambda row:
                    session.add(md.egoGridLine(
                        run_id=row['run_id'],
                        edge_name=row['edge_name'],
                        grid_id_db=row['grid_id_db'],
                        node1=row['node1'],
                        node2=row['node2'],
                        type_kind=row['type_kind'],
                        type_name=row['type_name'],
                        length=row['length'],
                        U_n=row['U_n'],
                        C=row['C'],
                        L=row['L'],
                        R=row['R'],
                        I_max_th=row['I_max_th'],
                    ))
                    , axis=1)

    elif tabletype == 'lv_cd':
        table.apply(lambda row:
                    session.add(md.EgoGridLvBranchtee(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        lv_grid_id_db=row['LV_grid_id_db'],
                        geom=row['geom'],
                    ))
                    , axis=1)

    elif tabletype == 'lv_gen':
        table.apply(lambda row:
                    session.add(md.EgoGridLvGenerator(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        lv_grid_id_db=row['LV_grid_id_db'],
                        geom=row['geom'],
                        type=row['type'],
                        subtype=row['subtype'],
                        v_level=row['v_level'],
                        nominal_capacity=row['nominal_capacity'],
                    ))
                    , axis=1)

    elif tabletype == 'lv_loads':
        table.apply(lambda row:
                    session.add(md.EgoGridLvLoad(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        lv_grid_id_db=row['LV_grid_id_db'],
                        geom=row['geom'],
                        consumption=row['consumption'],
                    ))
                    , axis=1)

    elif tabletype == 'lv_grid':
        table.apply(lambda row:
                    session.add(md.EgoGridLvGrid(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        lv_grid_id_db=row['LV_grid_id_db'],
                        geom=row['geom'],
                        population=row['population'],
                        voltage_nom=row['voltage_nom'],
                    ))
                    , axis=1)

    elif tabletype == 'lv_stations':
        table.apply(lambda row:
                    session.add(md.EgoGridLvmvStation(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        lv_grid_id_db=row['LV_grid_id_db'],
                        geom=row['geom'],
                    ))
                    , axis=1)

    elif tabletype == 'lv_trafos':
        table.apply(lambda row:
                    session.add(md.EgoGridLvTransformer(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        lv_grid_id_db=row['LV_grid_id_db'],
                        geom=row['geom'],
                        voltage_op=row['voltage_op'],
                        S_nom=row['S_nom'],
                        X=row['X'],
                        R=row['R'],
                    ))
                    , axis=1)

    elif tabletype == 'mapping':
        table.apply(lambda row:
                    session.add(md.EgoGridLvmvMapping(
                        run_id=row['run_id'],
                        lv_grid_id=row['LV_grid_id'],
                        lv_grid_id_db=row['LV_grid_id_db'],
                        mv_grid_id=row['MV_grid_id'],
                        mv_grid_id_db=row['MV_grid_id_db'],
                    ))
                    , axis=1)

    elif tabletype == 'mv_cd':
        table.apply(lambda row:
                    session.add(md.EgoGridMvBranchtee(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        mv_grid_id_db=row['MV_grid_id_db'],
                        geom=row['geom'],
                    ))
                    , axis=1)

    elif tabletype == 'mv_gen':
        table.apply(lambda row:
                    session.add(md.EgoGridMvGenerator(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        mv_grid_id_db=row['MV_grid_id_db'],
                        geom=row['geom'],
                        type=row['type'],
                        subtype=row['subtype'],
                        v_level=row['v_level'],
                        nominal_capacity=row['nominal_capacity'],
                    ))
                    , axis=1)

    elif tabletype == 'mv_loads':
        table.apply(lambda row:
                    session.add(md.EgoGridMvLoad(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        mv_grid_id_db=row['MV_grid_id_db'],
                        geom=row['geom'],
                        is_aggregated=row['is_aggregated'],
                        consumption=row['consumption'],
                    ))
                    , axis=1)

    elif tabletype == 'mv_grid':
        table.apply(lambda row:
                    session.add(md.EgoGridMvGrid(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        mv_grid_id=row['LV_grid_id'],
                        geom=row['geom'],
                        population=row['population'],
                        voltage_nom=row['voltage_nom'],
                    ))
                    , axis=1)

    elif tabletype == 'mv_stations':
        table.apply(lambda row:
                    session.add(md.EgoGridMvhvStation(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        mv_grid_id_db=row['MV_grid_id_db'],
                        geom=row['geom'],
                    ))
                    , axis=1)

    elif tabletype == 'mv_trafos':
        table.apply(lambda row:
                    session.add(md.EgoGridMvTransformer(
                        run_id=row['run_id'],
                        id_db=row['id_db'],
                        mv_grid_id_db=row['MV_grid_id_db'],
                        geom=row['geom'],
                        voltage_op=row['voltage_op'],
                        S_nom=row['S_nom'],
                        X=row['X'],
                        R=row['R'],
                    ))
                    , axis=1)
        if not engine.dialect.has_table(engine, 'ego_grid_mv_transformer'):
            print('helloworld')

    session.commit()

def export_data_to_oedb(session, lv_grid, lv_gen, lv_cd, lv_stations, lv_trafos, lv_loads, mv_grid, mv_gen, mv_cb,
                      mv_cd, mv_stations, mv_trafos, mv_loads, edges, mapping):

    #engine = create_engine('sqlite:///:memory:')
    export_network_to_oedb(session, lv_gen, 'lv_gen')
    export_network_to_oedb(session, lv_grid, 'lv_grid')
    export_network_to_oedb(session, lv_cd, 'lv_cd')
    export_network_to_oedb(session, lv_stations, 'lv_stations')
    export_network_to_oedb(session, lv_trafos, 'lv_trafos')
    export_network_to_oedb(session, lv_loads, 'lv_loads')
    export_network_to_oedb(session, mv_grid, 'mv_grid')
    export_network_to_oedb(session, mv_gen, 'mv_gen')
    #export_network_to_oedb(session, mv_cb, 'mv_cb')
    export_network_to_oedb(session, mv_cd, 'mv_cd')
    export_network_to_oedb(session, mv_stations, 'mv_stations')
    export_network_to_oedb(session, mv_trafos, 'mv_trafos')
    export_network_to_oedb(session, mv_loads, 'mv_loads')
    export_network_to_oedb(session, edges, 'edges')
    export_network_to_oedb(session, mapping, 'mapping')



def create_ding0_db_tables(engine):

    tables = [md.EgoGridLine, md.EgoGridLvBranchtee,
              md.EgoGridLvGenerator, md.EgoGridLvLoad, md.EgoGridLvGrid,
              md.EgoGridLvmvStation, md.EgoGridLvTransformer,
              md.EgoGridLvmvMapping, md.EgoGridMvBranchtee,
              md.EgoGridMvGenerator, md.EgoGridMvLoad, md.EgoGridMvGrid,
              md.EgoGridMvhvStation, md.EgoGridMvTransformer]

    for tab in tables:
        tab().__table__.create(bind=engine, checkfirst=True)

def drop_ding0_db_tables(engine):

    tables = [md.EgoGridLine, md.EgoGridLvBranchtee,
              md.EgoGridLvGenerator, md.EgoGridLvLoad, md.EgoGridLvGrid,
              md.EgoGridLvmvStation, md.EgoGridLvTransformer,
              md.EgoGridLvmvMapping, md.EgoGridMvBranchtee,
              md.EgoGridMvGenerator, md.EgoGridMvLoad, md.EgoGridMvGrid,
              md.EgoGridMvhvStation, md.EgoGridMvTransformer]

    print("Please confirm that you would like to drop the following tables:")
    for n, tab in enumerate(tables):
        print("{: 3d}. {}".format(n, tab))

    print("Please confirm with either of the choices below:\n" +\
          "- yes\n" +\
          "- no\n" +\
          "- the indexes to drop in the format 0, 2, 3, 5")
    confirmation = input("Please type the choice completely as there is no default choice.")
    if re.fullmatch('[Yy]es', confirmation):
        for tab in tables:
            tab().__table__.drop(bind=engine, checkfirst=True)
    elif re.fullmatch('[Nn]o', confirmation):
        print("Cancelled dropping of tables")
    else:
        try:
            indlist = confirmation.split(',')
            indlist = list(map(int, indlist))
            print("Please confirm deletion of the following tables:")
            tablist = np.array(tables)[indlist].tolist()
            for n, tab in enumerate(tablist):
                print("{: 3d}. {}".format(n, tab))
            con2 = input("Please confirm with either of the choices below:\n" +\
                         "- yes\n" +\
                         "- no")
            if re.fullmatch('[Yy]es', con2):
                for tab in tablist:
                    tab().__table__.drop(bind=engine, checkfirst=True)
            elif re.fullmatch('[Nn]o', con2):
                print("Cancelled dropping of tables")
            else:
                print("The input is unclear, no action taken")
        except ValueError:
            print("Confirmation unclear, no action taken")


########################################################
if __name__ == "__main__":
    #nw = init_mv_grid(mv_grid_districts=[3544, 3545])
    #init_mv_grid(mv_grid_districts=list(range(1, 4500, 200)),filename='ding0_tests_grids_1_4500_200.pkl')
    #nw = load_nd_from_pickle(filename='ding0_tests_grids_1.pkl')
    #nw = load_nd_from_pickle(filename='ding0_tests_grids_SevenDistricts.pkl')
    #nw = load_nd_from_pickle(filename='ding0_tests_grids_1_4500_200.pkl')
    #nw = init_mv_grid(mv_grid_districts=[2370],filename=False)
    #stats = calculate_mvgd_stats(nw)
    #print(stats)
    #print(stats.T)
    #stats.to_csv('stats_1_4500_200.csv')

    #############################################
    # generate stats in parallel
    mv_grid_districts = list(range(1728, 1755))
    n_of_processes = mp.cpu_count() #number of parallel threaths
    n_of_districts = 1 #n° of districts in each cluster
    mv_stats = parallel_running_stats(districts_list = mv_grid_districts,
                                      n_of_processes = n_of_processes,
                                      n_of_districts = n_of_districts,
                                      source = 'pkl',#'ding0', #
                                      mode = '',
                                      critical = True,
                                      save_csv = True)
    print('#################\nMV STATS:')
    print(mv_stats[0].T)
    #print('#################\nLV STATS:')
    #print(mv_stats[1].T)
    #print('#################\nMV Crit Nodes STATS:')
    #print(mv_stats[2].T)
    #print('#################\nMV Crit Edges STATS:')
    #print(mv_stats[3].T)
    #print('#################\nLV Crit Nodes STATS:')
    #print(mv_stats[4].T)
    #print('#################\nLV Crit Edges STATS:')
    #print(mv_stats[5].T)

    #############################################
    #nw = load_nd_from_pickle(filename='ding0_tests_grids_1567_567.pkl')
    #nw = load_nd_from_pickle(filename='ding0_grids_1729.pkl')
    #nw = init_mv_grid(mv_grid_districts=[1567, 567],filename='ding0_tests_grids_1567_567.pkl')
    #stats = calculate_lvgd_stats(nw)
    #print(stats.iloc[1:3].T)
    #print(stats[stats['Load Area is Aggregated']].T)
    #stats = calculate_mvgd_stats(nw)
    #print(stats.T)
    #print(stats.iloc[1:3].T)
    #stats = calculate_lvgd_voltage_current_stats(nw)
    #print(stats)
    #print(stats[0][1:3].T)
    #print(stats[1].T)
    #stats = calculate_mvgd_voltage_current_stats(nw)
    #print(stats[0])#.index.tolist())#[1:3].T)#nodes
    #print(stats[1][1:20])#edges
