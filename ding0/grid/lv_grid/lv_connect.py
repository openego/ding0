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

import pandas as pd

from ding0.core.network import BranchDing0
from ding0.core.network.cable_distributors import LVCableDistributorDing0
from ding0.core.network.loads import LVLoadDing0

from ding0.tools import config as cfg_ding0
from ding0.tools.geo import calc_geo_dist, calc_edge_geometry
from ding0.grid.tools import cable_type
import logging
import random

from scipy.spatial.distance import cdist
import numpy as np

logger = logging.getLogger(__name__)


def lv_connect_generators(mv_grid, graph, debug=False):

    """ Connect LV generators to LV grid
    
    Args
    ----
    lv_grid_district: LVGridDistrictDing0
        LVGridDistrictDing0 object for which the connection process has to be done
    graph: :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes
    debug: bool, defaults to False
        If True, information is printed during process

    Returns
    -------
    :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes and newly created branches
    """

    cable_lf = cfg_ding0.get('assumptions',
                             'load_factor_lv_cable_fc_normal')
    cos_phi_gen = cfg_ding0.get('assumptions',
                                'cos_phi_gen')
    connect_lv_nodes_by_voltage_level(
        mv_grid, mv_grid.lv_generators_to_connect, cable_lf, cos_phi_gen,
        add_to_grid=lambda grid, generator: grid.add_generator(generator))


def lv_connect_components(mv_grid, components, debug=False):
    """ Connect loads and storage units at locations of their own to LV grids,
    like generators of the same voltage level (see
    :func:`lv_connect_generators`)

    Args
    ----
    mv_grid: MVGridDing0
    components: :obj:`list`
        :class:`~.ding0.core.network.components.GridComponentDing0` objects
        of voltage levels 6 and 7
    debug: bool, defaults to False
    """
    cable_lf = cfg_ding0.get('assumptions',
                             'load_factor_lv_cable_lc_normal')
    cos_phi_load = cfg_ding0.get('assumptions',
                                 'cos_phi_load')
    connect_lv_nodes_by_voltage_level(
        mv_grid, components, cable_lf, cos_phi_load,
        add_to_grid=lambda grid, component: grid.add_component(component))


def connect_lv_nodes_by_voltage_level(mv_grid, nodes_to_connect, cable_lf,
                                      cos_phi, add_to_grid):
    """ Connect generators or components to LV grids by their voltage level:
    level 6 to the nearest LV station by a cable of its own, level 7 to the
    load with the same building id or else to the nearest LV load.

    Args
    ----
    mv_grid: MVGridDing0
    nodes_to_connect: :obj:`list`
        Objects with attributes v_level, geo_data, capacity (kW) and
        building_id
    cable_lf: :obj:`float`
        Load factor of cables of voltage level 6
    cos_phi: :obj:`float`
        Power factor to size cables of voltage level 6
    add_to_grid: function
        Adds a node to the LV grid it is connected to, called with the grid
        and the node
    """
    v_nom = cfg_ding0.get('assumptions', 'lv_nominal_voltage') / 1e3  # v_nom in kV
    seed = int(cfg_ding0.get('random', 'seed'))
    random.seed(a=seed)

    # Get positions of all LV nodes
    lv_nodes_df = pd.DataFrame(columns=["node", "type", "building_id", "x", "y"])
    for mv_grid_district in mv_grid.network.mv_grid_districts():
        for load_area in mv_grid_district.lv_load_areas():
            for lv_grid_district in load_area.lv_grid_districts():
                for node in lv_grid_district.lv_grid.graph.nodes():
                    lv_nodes_df.loc[str(node)] = [
                        node,
                        type(node).__name__,
                        node.building_id if isinstance(node, LVLoadDing0) else None,
                        node.geo_data.x,
                        node.geo_data.y
                    ]
    lv_stations_df = lv_nodes_df[lv_nodes_df["type"] == "LVStationDing0"]
    lv_loads_df = lv_nodes_df[lv_nodes_df["type"] == "LVLoadDing0"]

    def add_node_to_station(node, station):
        grid = station.grid
        graph = grid.graph

        branch_shp, branch_length = calc_edge_geometry(node, station)
        # Calculate length with detour factor
        branch_length = calc_geo_dist(node, station)
        branch_type = cable_type(
            node.capacity / (cable_lf * cos_phi),
            v_nom,
            grid.network.static_data['LV_cables'])

        branch = BranchDing0(length=branch_length,
                             kind='cable',
                             grid=grid,
                             type=branch_type,
                             geometry=branch_shp)

        node.lv_grid = grid
        node.lv_load_area = grid.grid_district.lv_load_area

        graph.add_edge(node, station, branch=branch)
        add_to_grid(grid, node)

    def add_node_to_node(node, node_to_connect):
        grid = node_to_connect.grid
        graph = grid.graph

        # Add helper cable
        branch = BranchDing0(
            grid=grid,
            id_db=f"helper_{node}",
            helper_component=True,
        )

        if isinstance(node_to_connect, LVLoadDing0):
            cable_distributor = list(graph.neighbors(node_to_connect))[0]
        elif isinstance(node_to_connect, LVCableDistributorDing0):
            cable_distributor = node_to_connect
        else:
            raise ValueError("False node as connection target!")

        node.lv_grid = grid
        node.lv_load_area = grid.grid_district.lv_load_area
        node.geo_data = cable_distributor.geo_data

        add_to_grid(grid, node)
        graph.add_edge(cable_distributor, node, branch=branch)

    def get_nearest_nodes_of_node(nearest_node, node):
        distance = cdist(
            nearest_node.loc[:, ["x", "y"]],
            [(node.geo_data.x, node.geo_data.y)],
            'euclidean'
        )
        nearest_node = nearest_node.iloc[np.where(distance == distance.min())[0][0]]["node"]
        return nearest_node

    for node in sorted(nodes_to_connect, key=lambda x: repr(x)):
        if node.v_level == 6:
            nearest_station = get_nearest_nodes_of_node(lv_stations_df, node)
            add_node_to_station(node, nearest_station)
        elif node.v_level == 7:
            if (
                    node.building_id
                    and not lv_loads_df[lv_loads_df["building_id"] == node.building_id].empty
            ):
                node_to_connect = lv_loads_df[lv_loads_df["building_id"] == node.building_id]["node"].values[0]
            else:
                node_to_connect = get_nearest_nodes_of_node(lv_loads_df, node)
            add_node_to_node(node, node_to_connect)
        else:
            raise ValueError("False Voltage Level!")
