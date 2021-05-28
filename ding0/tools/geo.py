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


import os
from geopy.distance import geodesic
from pyproj import Transformer

from ding0.tools import config as cfg_ding0
import logging

if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString
    from shapely.ops import transform

logger = logging.getLogger('ding0')


def calc_geo_branches_in_polygon(mv_grid, polygon, mode, proj):
    """ Calculate geographical branches in polygon.

    For a given `mv_grid` all branches (edges in the graph of the grid) are
    tested if they are in the given `polygon`. You can choose different modes
    and projections for this operation.

    Parameters
    ----------
    mv_grid : :class:`~.ding0.core.network.grids.MVGridDing0`
        MV Grid object. Edges contained in `mv_grid.graph_edges()` are taken
        for the test.
    polygon : :shapely:`Shapely Point object<points>`
        Polygon that contains edges.
    mode : :obj:`str`
        Choose between 'intersects' or 'contains'.
    proj : :obj:`int`
        EPSG code to specify projection

    Returns
    -------
    :obj:`list` of :any:`BranchDing0` objects
        List of branches

    """

    branches = []
    polygon_shp = transform(proj, polygon)
    for branch in mv_grid.graph_edges():
        nodes = branch['adj_nodes']
        branch_shp = transform(proj, LineString([nodes[0].geo_data, nodes[1].geo_data]))

        # check if branches intersect with polygon if mode = 'intersects'
        if mode == 'intersects':
            if polygon_shp.intersects(branch_shp):
                branches.append(branch)
        # check if polygon contains branches if mode = 'contains'
        elif mode == 'contains':
            if polygon_shp.contains(branch_shp):
                branches.append(branch)
        # error
        else:
            raise ValueError('Mode is invalid!')
    return branches


def calc_geo_branches_in_buffer(node, mv_grid, radius, radius_inc, proj):
    """ Determines branches in nodes' associated graph that are at least partly
    within buffer of `radius` from `node`.
    
    If there are no nodes, the buffer is successively extended by `radius_inc`
    until nodes are found.

    Parameters
    ----------
    node : LVStationDing0, GeneratorDing0, or CableDistributorDing0
        origin node (e.g. LVStationDing0 object) with associated shapely object
        (attribute `geo_data`) in any CRS (e.g. WGS84)
    radius : float
        buffer radius in m
    radius_inc : float
        radius increment in m
    proj : :obj:`int`
        pyproj projection object: nodes' CRS to equidistant CRS
        (e.g. WGS84 -> ETRS)

    Returns
    -------
    :obj:`list` of :networkx:`NetworkX Graph Obj< >`
        List of branches (NetworkX branch objects)

    """

    branches = []

    while not branches:
        node_shp = transform(proj, node.geo_data)
        buffer_zone_shp = node_shp.buffer(radius)
        for branch in mv_grid.graph_edges():
            nodes = branch['adj_nodes']
            branch_shp = transform(proj, LineString([nodes[0].geo_data, nodes[1].geo_data]))
            if buffer_zone_shp.intersects(branch_shp):
                branches.append(branch)
        radius += radius_inc

    return branches


def calc_geo_dist(node_source, node_target):
    """ Calculates the geodesic distance between `node_source` and `node_target`
    incorporating the detour factor specified in :file:`ding0/ding0/config/config_calc.cfg`.

    Parameters
    ----------
    node_source: LVStationDing0, GeneratorDing0, or CableDistributorDing0
        source node, member of GridDing0.graph
    node_target: LVStationDing0, GeneratorDing0, or CableDistributorDing0
        target node, member of GridDing0.graph

    Returns
    -------
    :any:`float`
        Distance in m
    """

    branch_detour_factor = cfg_ding0.get('assumptions', 'branch_detour_factor')

    # notice: geodesic takes (lat,lon)
    branch_length = branch_detour_factor * geodesic((node_source.geo_data.y, node_source.geo_data.x),
                                                    (node_target.geo_data.y, node_target.geo_data.x)).m

    # ========= BUG: LINE LENGTH=0 WHEN CONNECTING GENERATORS ===========
    # When importing generators, the geom_new field is used as position. If it is empty, EnergyMap's geom
    # is used and so there are a couple of generators at the same position => length of interconnecting
    # line is 0. See issue #76
    if branch_length == 0:
        branch_length = 1
        logger.warning('Geo distance is zero, check objects\' positions. '
                       'Distance is set to 1m')
    # ===================================================================

    return branch_length


def calc_geo_dist_matrix(nodes_pos):
    """ Calculates the geodesic distance between all nodes in `nodes_pos` incorporating the detour factor in config_calc.cfg.
        
    For every two points/coord it uses geopy's geodesic function. As default ellipsoidal model of the earth WGS-84 is used.
    For more options see
    
    https://geopy.readthedocs.io/en/stable/index.html?highlight=geodesic#geopy.distance.geodesic

    Parameters
    ----------
    nodes_pos: dict 
        dictionary of nodes with positions, with x=longitude, y=latitude, and the following format::
        
        {
            'node_1': (x_1, y_1), 
            ...,
            'node_n': (x_n, y_n)
        }
   
    Returns
    -------
    :obj:`dict`
        dictionary with distances between all nodes (in km), with the following format::
        
        {
            'node_1': {'node_1': dist_11, ..., 'node_n': dist_1n},
            ..., 
            'node_n': {'node_1': dist_n1, ..., 'node_n': dist_nn}
        }

    """

    branch_detour_factor = cfg_ding0.get('assumptions', 'branch_detour_factor')

    matrix = {}

    for i in nodes_pos:
        pos_origin = tuple(nodes_pos[i])

        matrix[i] = {}

        for j in nodes_pos:
            pos_dest = tuple(nodes_pos[j])
            # notice: geodesic takes (lat,lon), thus the (x,y)/(lon,lat) tuple is reversed
            distance = branch_detour_factor * geodesic(tuple(reversed(pos_origin)), tuple(reversed(pos_dest))).km
            matrix[i][j] = distance

    return matrix


def calc_geo_centre_point(node_source, node_target):
    """ Calculates the geodesic distance between `node_source` and `node_target`
    incorporating the detour factor specified in config_calc.cfg.
    
    Parameters
    ----------
    node_source: LVStationDing0, GeneratorDing0, or CableDistributorDing0
        source node, member of GridDing0.graph
    node_target: LVStationDing0, GeneratorDing0, or CableDistributorDing0
        target node, member of GridDing0.graph

    Returns
    -------
    :any:`float`
        Distance in m.
    """

    # WGS84 (conformal) to ETRS (equidistant) projection
    proj_source = Transformer.from_crs("epsg:4326", "epsg:3035", always_xy=True).transform
    # ETRS (equidistant) to WGS84 (conformal) projection
    proj_target = Transformer.from_crs("epsg:3035", "epsg:4326", always_xy=True).transform

    branch_shp = transform(proj_source, LineString([node_source.geo_data, node_target.geo_data]))

    distance = geodesic((node_source.geo_data.y, node_source.geo_data.x),
                        (node_target.geo_data.y, node_target.geo_data.x)).m

    centre_point_shp = transform(proj_target, branch_shp.interpolate(distance/2))

    return centre_point_shp
