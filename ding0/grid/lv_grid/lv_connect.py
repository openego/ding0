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

    def add_generator_to_station(generator, station):
        grid = station.grid
        graph = grid.graph

        branch_shp, branch_length = calc_edge_geometry(generator, station)
        # Calculate length with detour factor
        branch_length = calc_geo_dist(generator, station)
        branch_type = cable_type(
            generator.capacity / (cable_lf * cos_phi_gen),
            v_nom,
            lv_grid_district.lv_grid.network.static_data['LV_cables'])

        branch = BranchDing0(length=branch_length,
                             kind='cable',
                             grid=grid,
                             type=branch_type,
                             geometry=branch_shp)

        generator.lv_grid = grid
        generator.lv_load_area = grid.grid_district.lv_load_area

        graph.add_edge(generator, station, branch=branch)
        grid.add_generator(generator)

    def add_generator_to_node(generator, node_to_connect):
        grid = node_to_connect.grid
        graph = grid.graph

        # Add helper cable
        branch = BranchDing0(
            grid=grid,
            id_db=f"helper_{generator}",
            helper_component=True,
        )

        if isinstance(node_to_connect, LVLoadDing0):
            cable_distributor = list(graph.neighbors(node_to_connect))[0]
        elif isinstance(node_to_connect, LVCableDistributorDing0):
            cable_distributor = node_to_connect
        else:
            raise ValueError("False node as connection target!")

        generator.lv_grid = grid
        generator.lv_load_area = grid.grid_district.lv_load_area
        generator.geo_data = cable_distributor.geo_data

        grid.add_generator(generator)
        graph.add_edge(cable_distributor, generator, branch=branch)

    def get_nearest_nodes_of_generator(nearest_node, generator):
        distance = cdist(
            nearest_node.loc[:, ["x", "y"]],
            [(generator.geo_data.x, generator.geo_data.y)],
            'euclidean'
        )
        nearest_node = nearest_node.iloc[np.where(distance == distance.min())[0][0]]["node"]
        return nearest_node

    for generator in sorted(mv_grid.lv_generators_to_connect, key=lambda x: repr(x)):
        if generator.v_level == 6:
            nearest_station = get_nearest_nodes_of_generator(lv_stations_df, generator)
            add_generator_to_station(generator, nearest_station)
        elif generator.v_level == 7:
            if (
                    generator.building_id
                    and not lv_loads_df[lv_loads_df["building_id"] == generator.building_id].empty
            ):
                node_to_connect = lv_loads_df[lv_loads_df["building_id"] == generator.building_id]["node"].values[0]
            else:
                node_to_connect = get_nearest_nodes_of_generator(lv_loads_df, generator)
            add_generator_to_node(generator, node_to_connect)
        else:
            raise ValueError("False Voltage Level!")
