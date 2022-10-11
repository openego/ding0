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

logger = logging.getLogger(__name__)


def calc_geo_branches_in_polygon(mv_grid, polygon, mode, proj, srid=3035):
    """
    NEW PARAM srid=3035 to calculate calc_geo_branches_in_buffer.
    Before srid=4326 was assumed.
    
    Calculate geographical branches in polygon.

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
    if srid == 3035:
        branches = []
        for branch in mv_grid.graph_edges():
            branch_shp = branch['branch'].geometry
            # check if branches intersect with polygon if mode = 'intersects'
            if mode == 'intersects':
                if polygon.intersects(branch_shp):
                    branches.append(branch)
            # check if polygon contains branches if mode = 'contains'
            elif mode == 'contains':
                if polygon.contains(branch_shp):
                    branches.append(branch)
            # error
            else:
                raise ValueError('Mode is invalid!')
    else:
        branches = []
        polygon_shp = transform(proj, polygon)
        for branch in mv_grid.graph_edges():
            branch_shp = transform(proj, branch['branch'].geometry)

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


def calc_geo_branches_in_buffer(node, mv_grid, radius, radius_inc, proj, srid=3035):
    """
    NEW PARAM srid=3035 to calculate calc_geo_branches_in_buffer.
    Before srid=4326 was assumed.
    
    Determines branches in nodes' associated graph that are at least partly
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

    if srid == 3035:
        branches = []

        while not branches:
            node_shp = node.geo_data
            buffer_zone_shp = node_shp.buffer(radius)
            for branch in mv_grid.graph_edges():
                branch_shp = branch['branch'].geometry
                if buffer_zone_shp.intersects(branch_shp):
                    branches.append(branch)
            radius += radius_inc

    else:
        branches = []

        while not branches:
            node_shp = transform(proj, node.geo_data)
            buffer_zone_shp = node_shp.buffer(radius)
            for branch in mv_grid.graph_edges():
                branch_shp = transform(proj, branch['branch'].geometry)
                if buffer_zone_shp.intersects(branch_shp):
                    branches.append(branch)
            radius += radius_inc

    return branches


def calc_geo_dist(node_source, node_target, srid=3035):
    """ Calculates the geodesic distance between `node_source` and `node_target`
    incorporating the detour factor specified in :file:`ding0/ding0/config/config_calc.cfg`.

    Parameters
    ----------
    node_source: LVStationDing0, GeneratorDing0, or CableDistributorDing0
        source node, member of GridDing0.graph
    node_target: LVStationDing0, GeneratorDing0, or CableDistributorDing0
        target node, member of GridDing0.graph
    srid: defines crs. 4326 by default check config_misc.cfg
    Returns
    -------
    :any:`float`
        Distance in m
    """

    branch_detour_factor = cfg_ding0.get('assumptions', 'branch_detour_factor')

    if srid == 4326:
        # notice: geodesic takes (lat,lon)
        branch_length = branch_detour_factor * geodesic((node_source.geo_data.y, node_source.geo_data.x),
                                                        (node_target.geo_data.y, node_target.geo_data.x)).m
    elif srid == 3035:
        # NEU weil srid=4326 in config.misc.cfg
        branch_length = LineString([node_source.geo_data, node_target.geo_data]).length * branch_detour_factor

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


def calc_geo_dist_matrix(nodes_pos, srid=3035):
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
    
    if srid == 3035:
        
        for i in nodes_pos:
            pos_origin = tuple(nodes_pos[i])  # pos_origin to pos_dest

            matrix[i] = {}

            for j in nodes_pos:
                pos_dest = tuple(nodes_pos[j])
                distance = LineString([pos_origin, pos_dest]).length * branch_detour_factor / 1000  # km
                matrix[i][j] = distance
        
    else:  # ding0 default old

        for i in nodes_pos:
            pos_origin = tuple(nodes_pos[i])

            matrix[i] = {}

            for j in nodes_pos:
                pos_dest = tuple(nodes_pos[j])
                # notice: geodesic takes (lat,lon), thus the (x,y)/(lon,lat) tuple is reversed
                distance = branch_detour_factor * geodesic(tuple(reversed(pos_origin)), tuple(reversed(pos_dest))).km
                matrix[i][j] = distance

    return matrix


def calc_geo_centre_point(node_source, node_target, srid=3035):
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

    if srid == 3035:  # calc new approach
        branch_shp = LineString([node_source.geo_data, node_target.geo_data])
        distance = branch_shp.length
        centre_point_shp = branch_shp.interpolate(distance/2)
    else:
        # WGS84 (conformal) to ETRS (equidistant) projection
        proj_source = Transformer.from_crs("epsg:4326", "epsg:3035", always_xy=True).transform
        # ETRS (equidistant) to WGS84 (conformal) projection
        proj_target = Transformer.from_crs("epsg:3035", "epsg:4326", always_xy=True).transform

        branch_shp = transform(proj_source, LineString([node_source.geo_data, node_target.geo_data]))

        distance = geodesic((node_source.geo_data.y, node_source.geo_data.x),
                            (node_target.geo_data.y, node_target.geo_data.x)).m

        centre_point_shp = transform(proj_target, branch_shp.interpolate(distance/2))

    return centre_point_shp


def calc_edge_geometry(node_source, node_target, srid=3035):
    """ returns straight edge geometry and related length between two nodes as LineString 

    Parameters
    ----------
    node_source: LVStationDing0, GeneratorDing0, or CableDistributorDing0
        source node, member of GridDing0.graph
    node_target: LVStationDing0, GeneratorDing0, or CableDistributorDing0
        target node, member of GridDing0.graph
    srid: defines crs. 4326 by default check config_misc.cfg
    Returns
    -------
    geometry: LineString,
    :any:`float`
        Distance in m
    """

    if srid == 3035:
        branch_geometry = LineString([node_source.geo_data, node_target.geo_data])
        branch_length = branch_geometry.length
        
    # ========= BUG: LINE LENGTH=0 WHEN CONNECTING GENERATORS ===========
    # When importing generators, the geom_new field is used as position. If it is empty, EnergyMap's geom
    # is used and so there are a couple of generators at the same position => length of interconnecting
    # line is 0. See issue #76
    if branch_length == 0:
        branch_length = 1
        logger.warning('Geo distance is zero, check objects\' positions. '
                       'Distance is set to 1m')
    # ===================================================================

    return branch_geometry, branch_length
