from geopy.distance import vincenty


def calc_geo_dist_point_buffer(node_orig, nodes_dest, radius):
    """ Determines nodes in `nodes_dest` that are within buffer of `radius` from `node_orig`, sorted ascending
        by distance.

    Args:
        node_orig: origin node (shapely Point object)
        nodes_dest: destination nodes (shapely MultiPoint object)
        radius: buffer radius in m

    Returns:
        shapely MultiPoint object
        OLD:
        dictionary with origin nodes and nodes that are within buffer of `radius`, sorted ascending by distance.
        Format: {node_orig_1: {node_dest_x, ..., node_dest_y},
                 ...,
                 node_orig_n: {node_dest_x, ..., node_dest_y}
                }
    """

    buffer_zone = node_orig.buffer(radius)
    return nodes_dest.intersection(buffer_zone)


def calc_geo_dist_vincenty(nodes_orig_pos, nodes_dest_pos):
    """ Calculates the geodesic distance between all nodes in `nodes_orig_pos` to nodes in `nodes_dest_pos`. For every
    two points/coord it uses geopy's vincenty function (formula devised by Thaddeus Vincenty, with an accurate
    ellipsoidal model of the earth). As default ellipsoidal model of the earth WGS-84 is used. For more options see
    https://geopy.readthedocs.org/en/1.10.0/index.html?highlight=vincenty#geopy.distance.vincenty

    Args:
        nodes_orig_pos: dictionary of origin nodes with positions
        nodes_dest_pos: dictionary of destination nodes with positions

    Returns:
        dictionary with distances between origin nodes to destination nodes
    """
    # TODO: REVISE!

    matrix = {}

    for i in nodes_orig_pos:
        pos_origin = tuple(nodes_orig_pos[i])

        matrix[i] = {}

        for j in nodes_dest_pos:
            pos_dest = tuple(nodes_dest_pos[j])
            distance = vincenty(pos_origin, pos_dest).km
            matrix[i][j] = distance

    return matrix


def calc_geo_dist_matrix_vincenty(nodes_pos):
    """ Calculates the geodesic distance between all nodes in `nodes_pos`. For every two points/coord it uses geopy's
    vincenty function (formula devised by Thaddeus Vincenty, with an accurate ellipsoidal model of the earth). As
    default ellipsoidal model of the earth WGS-84 is used. For more options see
    https://geopy.readthedocs.org/en/1.10.0/index.html?highlight=vincenty#geopy.distance.vincenty

    Args:
        nodes_pos: dictionary of nodes with positions,
                   Format: {'node_1': (x_1, y_1),
                            ...,
                            'node_n': (x_n, y_n)
                           }

    Returns:
        dictionary with distances between all nodes,
        Format: {'node_1': {'node_1': dist_11, ..., 'node_n': dist_1n},
                 ...,
                 'node_n': {'node_1': dist_n1, ..., 'node_n': dist_nn
                }
    """

    matrix = {}

    for i in nodes_pos:
        pos_origin = tuple(nodes_pos[i])

        matrix[i] = {}

        for j in nodes_pos:
            pos_dest = tuple(nodes_pos[j])
            distance = vincenty(pos_origin, pos_dest).km
            matrix[i][j] = distance

    return matrix