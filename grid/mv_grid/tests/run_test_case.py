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
import sys

# workaround: add ding0 to sys.path to allow imports
PACKAGE_PARENT = '../../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import time
import matplotlib.pyplot as plt

from ding0.grid.mv_grid.util import util, data_input
from ding0.grid.mv_grid.solvers import savings, local_search


def main():
    """ 
    Description of Test Case
    """
    plt.close('all')

    graph = data_input.read_file('./testcases/Augerat/A-n32-k5.vrp')
    #graph = data_input.read_file('./material/monte-carlo-cvrp-master/input/Augerat/A-n80-k10.vrp')
    #graph = data_input.read_file('./material/monte-carlo-cvrp-master/input/Augerat/A-n69-k9.vrp')

    timeout = 30000

    savings_solver = savings.ClarkeWrightSolver()
    local_search_solver = local_search.LocalSearchSolver()

    start = time.time()

    savings_solution = savings_solver.solve(graph, timeout)
    elapsed = time.time() - start

    #if not savings_solution.is_complete():
    #    print('=== Solution is not a complete solution! ===')

    print('ClarkeWrightSolver solution:')
    util.print_solution(savings_solution)
    #print('Elapsed time (seconds): {}'.format(elapsed))

    savings_solution.draw_network()

    local_search_solution = local_search_solver.solve(graph, savings_solution, timeout)
    print('Local Search solution:')
    util.print_solution(local_search_solution)
    local_search_solution.draw_network()


if __name__ == '__main__':
    main()
