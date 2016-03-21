from geopy.distance import vincenty


def calc_geo_distance_vincenty(nodes_pos):
    """ Calculates the geodesic distance between all nodes. For every two points/coord it uses geopy's vincenty function
    (formula devised by Thaddeus Vincenty, with an accurate ellipsoidal model of the earth). As default ellipsoidal
    model of the earth WGS-84 is used. For more options see
    https://geopy.readthedocs.org/en/1.10.0/index.html?highlight=vincenty#geopy.distance.vincenty

    Args:
        nodes_pos:

    Returns:

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