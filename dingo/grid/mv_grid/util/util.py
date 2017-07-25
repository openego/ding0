"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io

Based on code by Romulo Oliveira copyright (C) 2015,
https://github.com/RomuloOliveira/monte-carlo-cvrp
Originally licensed under the Apache License, Version 2.0. You may obtain a
copy of the license at http://www.apache.org/licenses/LICENSE-2.0
"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


def print_upper_triangular_matrix(matrix):
    """Prints a CVRP data dict matrix"""

    # Print column header
    # Assumes first row contains all needed headers
    first = sorted(matrix.keys())[0]
    print('\t', end=' ')
    for i in matrix[first]:
        print('{}\t'.format(i), end=' ')
    print()

    indent_count = 0

    for i in matrix:
        # Print line header
        print('{}\t'.format(i), end=' ')

        if indent_count:
            print('\t' * indent_count, end=' ')

        for j in sorted(matrix[i]): # required because dict doesn't guarantee insertion order
            print('{}\t'.format(matrix[i][j]), end=' ')

        print()

        indent_count = indent_count + 1

def print_upper_triangular_matrix_as_complete(matrix):
    """Prints a CVRP data dict upper triangular matrix as a normal matrix

    Doesn't print header"""
    for i in sorted(matrix.keys()):
        for j in sorted(matrix.keys()):
            a, b = i, j
            if a > b:
                a, b = b, a

            print(matrix[a][b], end=' ')

        print()

def print_solution(solution):
    """Prints a solution

    Solution is an instance of project.solvers.BaseSolution

    Example:
        [8, 9, 10, 7]: 160
        [5, 6]: 131
        [3, 4, 2]: 154
        Total cost: 445
    """
    total_cost = 0
    for solution in solution.routes():
        cost = solution.length()
        total_cost = total_cost + cost
        print('{}: {}'.format(solution, cost))
        #print('xxx')
    print('Total cost: {}'.format(total_cost))
