"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DINGO lives at github: https://github.com/openego/dingo/
The documentation is available on RTD: http://dingo.readthedocs.io

Based on code by Romulo Oliveira copyright (C) 2015,
https://github.com/RomuloOliveira/monte-carlo-cvrp
Originally licensed under the Apache License, Version 2.0. You may obtain a
copy of the license at http://www.apache.org/licenses/LICENSE-2.0
"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/dingo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


import re
import math

from os import path
from dingo.grid.mv_grid.models.models import Graph


class ParseException(Exception):
    """Exception raised when something unexpected occurs in a TSPLIB file parsing"""
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def strip(line):
    """Removes any \r or \n from line and remove trailing whitespaces"""
    return line.replace('\r\n', '').strip() # remove new lines and trailing whitespaces


def sanitize(filename):
    """Returns a sanitized file name with absolut path

    Example: ~/input.txt -> /home/<your_home/input.txt
    """
    return path.abspath(path.expanduser(path.expandvars(filename)))


def _parse_depot_section(f):
    """Parse TSPLIB DEPOT_SECTION data part from file descriptor f

    Returns an array of depots
    """
    depots = []

    for line in f:
        line = strip(line)
        if line == '-1' or line == 'EOF': # End of section
            break
        else:
            depots.append(line)

    if len(depots) != 1:
        raise ParseException('One and only one depot is supported')

    return int(depots[0])


def _parse_nodes_section(f, current_section, nodes):
    """Parse TSPLIB NODE_COORD_SECTION or DEMAND_SECTION from file descript f

    Returns a dict containing the node as key
    """
    section = {}
    dimensions = None

    if current_section == 'NODE_COORD_SECTION':
        dimensions = 3 # i: (i, j)
    elif current_section == 'DEMAND_SECTION':
        dimensions = 2 # i: q
    else:
        raise ParseException('Invalid section {}'.format(current_section))

    n = 0
    for line in f:
        line = strip(line)

        # Check dimensions
        definitions = re.split('\s*', line)
        if len(definitions) != dimensions:
            raise ParseException('Invalid dimensions from section {}. Expected: {}'.format(current_section, dimensions))

        node = int(definitions[0])
        values = [int(v) for v in definitions[1:]]

        if len(values) == 1:
            values = values[0]

        section[node] = values

        n = n + 1
        if n == nodes:
            break

    # Assert all nodes were read
    if n != nodes:
        raise ParseException('Missing {} nodes definition from section {}'.format(nodes - n, current_section))

    return section


def _parse_edge_weight(f, nodes):
    """Parse TSPLIB EDGE_WEIGHT_SECTION from file f

    Supports only FULL_MATRIX for now
    """
    matrix = []

    n = 0

    for line in f:
        line = strip(line)

        regex = re.compile('\s+')

        row = regex.split(line)

        matrix.append(row)

        n = n + 1

        if n == nodes:
            break

    if n != nodes:
        raise ParseException('Missing {} nodes definition from section EDGE_WEIGHT_SECTION'.format(nodes - n))

    return matrix


def calculate_euc_distance(a, b):
    """Calculates Eclidian distances from two points a and b

    Points are two-dimension tuples
    """
    x1, y1 = a
    x2, y2 = b

    return int(round(math.sqrt(((x1 - x2) ** 2) + (((y1 - y2) ** 2))))) # HIER NICHT RUNDEN


def _post_process_specs(specs):
    """Post-process specs after pure parsing

    Casts any number expected values into integers

    Remarks: Modifies the specs object
    """
    integer_specs = ['DIMENSION', 'CAPACITY']

    for s in integer_specs:
        specs[s] = int(specs[s])


def _create_node_matrix_from_coord_section(specs):
    """Transformed parsed data from NODE_COORD_SECTION into an upper triangular matrix

    Calculates distances between nodes
    'MATRIX' key added to `specs`
    """
    distances = specs['NODE_COORD_SECTION']

    specs['MATRIX'] = {}

    for i in distances:
        origin = tuple(distances[i])

        specs['MATRIX'][i] = {}

        for j in specs['NODE_COORD_SECTION']:
            destination = tuple(distances[j])

            distance = calculate_euc_distance(origin, destination)

            #
            # Upper triangular matrix
            # if i > j, ij = 0
            #
            #if i > j:
            #    continue

            specs['MATRIX'][i][j] = distance
            #specs['MATRIX'][i][i] = distance


def _create_node_matrix_from_full_matrix(specs):
    """Transform parsed data from EDGE_WEIGHT_SECTION into an upper triangular matrix

    'MATRIX' key added to `specs`
    """
    old_matrix = specs['EDGE_WEIGHT_SECTION']
    nodes = specs['DIMENSION']

    specs['MATRIX'] = {}

    for i in range(nodes):
        specs['MATRIX'][i + 1] = {}

        for j in range(nodes):
            if i > j:
                continue

            specs['MATRIX'][i + 1][j + 1] = int(old_matrix[i][j])


def _create_node_matrix(specs):
    """Transform parsed data into an upper triangular matrix

    'MATRIX' key added to `specs`
    """
    if specs['EDGE_WEIGHT_TYPE'] == 'EUC_2D':
        _create_node_matrix_from_coord_section(specs)
    elif specs['EDGE_WEIGHT_FORMAT'] == 'FULL_MATRIX':
        _create_node_matrix_from_full_matrix(specs)
    else:
        raise ParseException('Could not create node matrix: Invalid EDGE_WEIGHT_TYPE or EDGE_WEIGHT_FORMAT')


def _setup_depot(specs):
    """Setup depot model

    'DEPOT' key added to `specs`
    """
    specs['DEPOT'] = specs['DEPOT_SECTION']


def _setup_demands(specs):
    """Setup demand model"""
    specs['DEMAND'] = specs['DEMAND_SECTION']


def _post_process_data(specs):
    """Post-process specs data after complete parsing

    Processes:
        - Calculates distances and model it in a matrix
        - Setup depot model
        - Setup demand model
    """

    _create_node_matrix(specs)
    _setup_depot(specs)
    _setup_demands(specs)


def _parse_tsplib(f):
    """Parses a TSPLIB file descriptor and returns a dict containing the problem definition"""
    line = ''

    specs = {}

    used_specs = ['NAME', 'COMMENT', 'DIMENSION', 'CAPACITY', 'TYPE', 'EDGE_WEIGHT_TYPE']
    used_data = ['DEMAND_SECTION', 'DEPOT_SECTION']

    # Parse specs part
    for line in f:
        line = strip(line)

        # Arbitrary sort, so we test everything out
        s = None
        for s in used_specs:
            if line.startswith(s):
                specs[s] = line.split('{} :'.format(s))[-1].strip() # get value data part
                break

        if s == 'EDGE_WEIGHT_TYPE' and s in specs and specs[s] == 'EXPLICIT':
            used_specs.append('EDGE_WEIGHT_FORMAT')

        # All specs read
        if len(specs) == len(used_specs):
            break

    if len(specs) != len(used_specs):
        missing_specs = set(used_specs).symmetric_difference(set(specs))
        raise ParseException('Error parsing TSPLIB data: specs {} missing'.format(missing_specs))

    print(specs)

    if specs['EDGE_WEIGHT_TYPE'] == 'EUC_2D':
        used_data.append('NODE_COORD_SECTION')
    elif specs['EDGE_WEIGHT_FORMAT'] == 'FULL_MATRIX':
        used_data.append('EDGE_WEIGHT_SECTION')
    else:
        raise ParseException('EDGE_WEIGHT_TYPE or EDGE_WEIGHT_FORMAT not supported')

    _post_process_specs(specs)

    # Parse data part
    for line in f:
        line = strip(line)

        for d in used_data:
            if line.startswith(d):
                if d == 'DEPOT_SECTION':
                    specs[d] = _parse_depot_section(f)
                elif d in ['NODE_COORD_SECTION', 'DEMAND_SECTION']:
                    specs[d] = _parse_nodes_section(f, d, specs['DIMENSION'])
                elif d == 'EDGE_WEIGHT_SECTION':
                    specs[d] = _parse_edge_weight(f, specs['DIMENSION'])

        if len(specs) == len(used_specs) + len(used_data):
            break

    if len(specs) != len(used_specs) + len(used_data):
        missing_specs = set(specs).symmetric_difference(set(used_specs).union(set(used_data)))
        raise ParseException('Error parsing TSPLIB data: specs {} missing'.format(missing_specs))

    _post_process_data(specs)

    return specs


def read_file(filename):
    """Reads a TSPLIB file and returns the problem data"""
    sanitized_filename = sanitize(filename)

    f = open(sanitized_filename)

    specs = None

    try:
        specs = _parse_tsplib(f)
    except ParseException:
        raise
    finally: # 'finally' is executed even when we re-raise exceptions
        f.close()

    if specs['TYPE'] != 'CVRP':
        raise Exception('Not a CVRP TSPLIB problem. Found: {}'.format(specs['TYPE']))
    
    #additional params for graph/network (temporary)
    specs['VOLTAGE'] = 20000
    specs['CABLETYPE'] = 1

    #return (Graph(specs), specs)
    return Graph(specs)
