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


import time
import itertools as it

from ding0.grid.mv_grid.models import models
from ding0.grid.mv_grid.solvers.base import BaseSolution, BaseSolver

from ding0.tools import config as cfg_ding0
import logging


logger = logging.getLogger('ding0')


class LocalSearchSolution(BaseSolution):
    """Solution class for Local Search metaheuristic
    
    Parameters
    ----------
    cvrp_problem : type
        Descr
    solution : BaseSolution
        Descr
    """

    def __init__(self, cvrp_problem, solution):
        super(LocalSearchSolution, self).__init__(cvrp_problem)
        
        self._nodes = solution._nodes
        self._routes = solution._routes

    def clone(self):
        """Returns a deep copy of self

        Function clones:
        
        * route
        * allocation
        * nodes
        
        Returns
        -------
        LocalSearchSolution
            Deep copy of self
        """

        new_solution = self.__class__(self._problem, self._vehicles)

        # Clone routes
        for index, r in enumerate(self._routes):
            new_route = new_solution._routes[index] = models.Route(self._problem)
            for node in r.nodes():
                # Insert new node on new route
                new_node = new_solution._nodes[node.name()]
                new_route.allocate([new_node])

        # remove empty routes from new solution
        new_solution._routes = [route for route in new_solution._routes if route._nodes]

        return new_solution


class LocalSearchSolver(BaseSolver):
    """ Improve initial savings solution using local search

    The implementation of the local searach algorithm founds on the following
    publications [#]_, [#]_, [#]_, [#]_
    
    Graph operators::

        Or-Opt (intra-route)
        Relocate (inter-route)
        Exchange (inter-route)

    Todo
    ----
        * Cross (inter-route) - to remove crossing edges between two routes

    References
    ----------
    .. [#] W. Wenger, "Multikriterielle Tourenplanung", Dissertation, 2009
    .. [#] M. Kämpf, "Probleme der Tourenbildung", Chemnitzer Informatik-Berichte, 2006
    .. [#] O. Bräysy, M. Gendreau, "Vehicle Routing Problem with Time Windows,
        Part I: Route Construction and Local Search Algorithms",
        Transportation Science, vol. 39, Issue 1, pp. 104-118, 2005
    .. [#] C. Boomgaarden, "Dynamische Tourenplanung und -steuerung",
        Dissertation, 2007

    """
    # TODO: Cross (inter-route), see above
    
    def operator_oropt(self, graph, solution, op_diff_round_digits, anim=None):
        # TODO: check docstring
        """Applies Or-Opt intra-route operator to solution
        
        Takes chains of nodes (length=3..1 consecutive nodes) from a given
        route and calculates savings when inserted into another position on the
        same route (all possible positions). Performes best move (max. saving)
        and starts over again with new route until no improvement is found.
        
        Parameters
        ----------
        graph: :networkx:`NetworkX Graph Obj< >`
            A NetworkX graaph is used.
        solution: BaseSolution
            BaseSolution instance
        op_diff_round_digits: float
            Precision (floating point digits) for rounding route length differences.
            
            *Details*: In some cases when an exchange is performed on two routes with one node each,
            the difference between the both solutions (before and after the exchange) is not zero.
            This is due to internal rounding errors of float type. So the loop won't break
            (alternating between these two solutions), we need an additional criterion to avoid
            this behaviour: A threshold to handle values very close to zero as if they were zero
            (for a more detailed description of the matter see http://floating-point-gui.de or
            https://docs.python.org/3.5/tutorial/floatingpoint.html)
        anim: AnimationDing0
            AnimationDing0 object
        
        Returns
        -------
        LocalSearchSolution
           A solution (LocalSearchSolution class)
        
        Note
        -----
        Since Or-Opt is an intra-route operator, it has not to be checked if route can allocate (Route's method
        can_allocate()) nodes during relocation regarding max. peak load/current because the line/cable type is the
        same along the entire route. However, node order within a route has an impact on the voltage stability
        so the check would be actually required. Due to large line capacity (load factor of lines/cables ~60 %)
        the voltage stability issues are neglected.

        (Inner) Loop variables:
        
        * s: length (count of consecutive nodes) of the chain that is moved. Values: 3..1
        * i: node that precedes the chain before moving (position in the route `tour`, not node name)
        * j: node that precedes the chain after moving (position in the route `tour`, not node name)
        
        Todo
        ----
        * insert literature reference for Or-algorithm here
        * Remove ugly nested loops, convert to more efficient matrix operations
        """
        no_ctr = 100
        # shorter var names for loop
        dm = graph._matrix
        dn = graph._nodes
        
        for route in solution.routes():

            # exclude routes with single high-demand nodes (Load Areas)
            if len(route._nodes) == 1:
                if solution._problem._is_aggregated[str(route._nodes[0])]:
                    continue

            n = len(route._nodes)+1

            # create tour by adding depot at start and end
            tour = [graph._depot] + route._nodes + [graph._depot]
            
            # Or-Opt: Search better solutions by checking possible chain moves
            while True:
                length = route.length()
                length_best = length
                
                for s in range(3,0,-1):
                    for i in range(1,n-s):
                        length_diff = (length -
                                       dm[dn[tour[i-1].name()]][dn[tour[i].name()]] -
                                       dm[dn[tour[i+s-1].name()]][dn[tour[i+s].name()]] +
                                       dm[dn[tour[i-1].name()]][dn[tour[i+s].name()]])
                        for j in range(i+s+1,n+1):
                            if j == n:
                                j2 = 1
                            else:
                                j2 = j+1
                            length_new = (length_diff +
                                          dm[dn[tour[j-1].name()]][dn[tour[i].name()]] +
                                          dm[dn[tour[i+s-1].name()]][dn[tour[j2-1].name()]] -
                                          dm[dn[tour[j-1].name()]][dn[tour[j2-1].name()]])
                            if length_new < length_best:
                                length_best = length_new
                                s_best, i_best, j_best = s, i, j
                if length_best < length:
                    tour = tour[0:i_best] + tour[i_best+s_best:j_best] + tour[i_best:i_best+s_best] + tour[j_best:n+1]

                    if anim is not None:
                        solution.draw_network(anim)

                # no improvement found
                if length_best == length:
                    # replace old route by new (same arg for allocation and deallocation since node order is considered at allocation)
                    solution._routes[solution._routes.index(route)].deallocate(tour[1:-1])
                    solution._routes[solution._routes.index(route)].allocate(tour[1:-1])
                    
                    break
        
        #solution = LocalSearchSolution(solution, graph, new_routes)
        return solution
    
    def operator_relocate(self, graph, solution, op_diff_round_digits, anim):
        """applies Relocate inter-route operator to solution
        
        Takes every node from every route and calculates savings when inserted
        into all possible positions in other routes. Insertion is done at
        position with max. saving and procedure starts over again with newly
        created graph as input. Stops when no improvement is found.
        
        Parameters
        ----------
        graph: :networkx:`NetworkX Graph Obj< >`
            A NetworkX graaph is used.
        solution: BaseSolution
            BaseSolution instance
        op_diff_round_digits: float
            Precision (floating point digits) for rounding route length differences.
            
            *Details*: In some cases when an exchange is performed on two routes with one node each,
            the difference between the both solutions (before and after the exchange) is not zero.
            This is due to internal rounding errors of float type. So the loop won't break
            (alternating between these two solutions), we need an additional criterion to avoid
            this behaviour: A threshold to handle values very close to zero as if they were zero
            (for a more detailed description of the matter see http://floating-point-gui.de or
            https://docs.python.org/3.5/tutorial/floatingpoint.html)
        anim: AnimationDing0
            AnimationDing0 object
        
        Returns
        -------
        LocalSearchSolution
           A solution (LocalSearchSolution class)
        
        Note
        -----
        (Inner) Loop variables:
        
        * i: node that is checked for possible moves (position in the route `tour`, not node name)
        * j: node that precedes the insert position in target route (position in the route `target_tour`, not node name)
            
        Todo
        ----
        * Remove ugly nested loops, convert to more efficient matrix operations
        """
        # shorter var names for loop
        dm = graph._matrix
        dn = graph._nodes        
        
        # Relocate: Search better solutions by checking possible node moves
        while True:
            length_diff_best = 0
            
            for route in solution.routes():

                # exclude origin routes with single high-demand nodes (Load Areas)
                if len(route._nodes) == 1:
                    if solution._problem._is_aggregated[str(route._nodes[0])]:
                        continue

                # create tour by adding depot at start and end
                tour = [graph._depot] + route._nodes + [graph._depot]
                
                for target_route in solution.routes():

                    # exclude (origin+target) routes with single high-demand nodes (Load Areas)
                    if len(target_route._nodes) == 1:
                        if solution._problem._is_aggregated[str(target_route._nodes[0])]:
                            continue

                    target_tour = [graph._depot] + target_route._nodes + [graph._depot]
                    
                    if route == target_route:
                        continue
                    
                    n = len(route._nodes)
                    nt = len(target_route._nodes)+1

                    for i in range(0,n):
                        node = route._nodes[i]
                        for j in range(0,nt):
                            #target_node = target_route._nodes[j]
                            
                            if target_route.can_allocate([node]):
                                length_diff = (-dm[dn[tour[i].name()]][dn[tour[i+1].name()]] -
                                                dm[dn[tour[i+1].name()]][dn[tour[i+2].name()]] +
                                                dm[dn[tour[i].name()]][dn[tour[i+2].name()]] +
                                                dm[dn[target_tour[j].name()]][dn[tour[i+1].name()]] +
                                                dm[dn[tour[i+1].name()]][dn[target_tour[j+1].name()]] -
                                                dm[dn[target_tour[j].name()]][dn[target_tour[j+1].name()]])

                                if length_diff < length_diff_best:
                                    length_diff_best = length_diff
                                    node_best, target_route_best, j_best = node, target_route, j
                                        
            if length_diff_best < 0:
                # insert new node
                target_route_best.insert([node_best], j_best)
                # remove empty routes from solution
                solution._routes = [route for route in solution._routes if route._nodes]

                if anim is not None:
                    solution.draw_network(anim)
                
                #print('Bessere Loesung gefunden:', node_best, target_node_best, target_route_best, length_diff_best)
            
            # no improvement found
            if round(length_diff_best, op_diff_round_digits) == 0:
                break

            
        return solution
        
    def operator_exchange(self, graph, solution, op_diff_round_digits, anim):
        """applies Exchange inter-route operator to solution
        
        Takes every node from every route and calculates savings when exchanged
        with another one of all possible nodes in other routes. Insertion is done at
        position with max. saving and procedure starts over again with newly
        created graph as input. Stops when no improvement is found.
        
        Parameters
        ----------
        graph: :networkx:`NetworkX Graph Obj< >`
            A NetworkX graaph is used.
        solution: BaseSolution
            BaseSolution instance
        op_diff_round_digits: float
            Precision (floating point digits) for rounding route length differences.
            
            *Details*: In some cases when an exchange is performed on two routes with one node each,
            the difference between the both solutions (before and after the exchange) is not zero.
            This is due to internal rounding errors of float type. So the loop won't break
            (alternating between these two solutions), we need an additional criterion to avoid
            this behaviour: A threshold to handle values very close to zero as if they were zero
            (for a more detailed description of the matter see http://floating-point-gui.de or
            https://docs.python.org/3.5/tutorial/floatingpoint.html)
        anim: AnimationDing0
            AnimationDing0 object
        
        Returns
        -------
        LocalSearchSolution
           A solution (LocalSearchSolution class)
        
        Note
        -----
        (Inner) Loop variables:
        
        * i: node that is checked for possible moves (position in the route `tour`, not node name)
        * j: node that precedes the insert position in target route (position in the route `target_tour`, not node name)
            
        Todo
        ----
        * allow moves of a 2-node chain
        * Remove ugly nested loops, convert to more efficient matrix operations
        """

        # shorter var names for loop
        dm = graph._matrix
        dn = graph._nodes

        exchange_step = []
        
        # Exchange: Search better solutions by checking possible node exchanges
        while True:
            length_diff_best = 0
            
            for route in solution.routes():

                # exclude origin routes with single high-demand nodes (Load Areas)
                if len(route._nodes) == 1:
                    if solution._problem._is_aggregated[str(route._nodes[0])]:
                        continue

                # create tour by adding depot at start and end
                tour = [graph._depot] + route._nodes + [graph._depot]
                
                for target_route in solution.routes():

                    if route == target_route:
                        continue

                    # exclude (origin+target) routes with single high-demand nodes (Load Areas)
                    if len(target_route._nodes) == 1:
                        if solution._problem._is_aggregated[str(target_route._nodes[0])]:
                            continue

                    target_tour = [graph._depot] + target_route._nodes + [graph._depot]

                    n = len(route._nodes)
                    nt = len(target_route._nodes)

                    for i in range(0,n):
                        node = route._nodes[i]
                        for j in range(0,nt):
                            target_node = target_route._nodes[j]

                            length_diff = (-dm[dn[tour[i].name()]][dn[tour[i+1].name()]] -
                                            dm[dn[tour[i+1].name()]][dn[tour[i+2].name()]] -
                                            dm[dn[target_tour[j].name()]][dn[target_tour[j+1].name()]] -
                                            dm[dn[target_tour[j+1].name()]][dn[target_tour[j+2].name()]] +
                                            dm[dn[tour[i].name()]][dn[target_tour[j+1].name()]] +
                                            dm[dn[target_tour[j+1].name()]][dn[tour[i+2].name()]] +
                                            dm[dn[target_tour[j].name()]][dn[tour[i+1].name()]] +
                                            dm[dn[tour[i+1].name()]][dn[target_tour[j+2].name()]])
    
                            if length_diff < length_diff_best:
                                length_diff_best = length_diff                           
                                i_best, j_best = i, j
                                node_best, target_node_best, route_best, target_route_best = node, target_node, route, target_route
                                        
            if length_diff_best < 0:
                if route_best.can_allocate([target_node_best], i_best) and \
                        route_best.can_allocate([node_best], j_best):
                    # insert new node
                    target_route_best.insert([node_best], j_best)
                    route_best.insert([target_node_best], i_best)
                    # remove empty routes from solution
                    solution._routes = [route for route in solution._routes if route._nodes]
                else:
                    # This block is required for the following reason:
                    # If there's exactly one better solution (length_diff_best < 0) but the routes (route_best,
                    # route_best) cannot exchange the concerned nodes (node_best <-> target_node_best) due to a
                    # validation of tech. constraints the exchange won't be done. So the exit condition below
                    # (length_diff_best==0) isn't fulfilled and it never can be since there're no more better
                    # solutions.
                    # This is why the current (best) exchange configuration has to be compared to the former best.
                    # If it has not changed, stop the exchange operator by setting length_diff_best = 0.
                    if exchange_step == [node_best, route_best,
                                         target_node_best, target_route_best,
                                         round(length_diff_best, 5)]:
                        length_diff_best = 0
                    else:
                        # save current exchange configuration
                        exchange_step = [node_best, route_best,
                                         target_node_best, target_route_best,
                                         round(length_diff_best, 5)]

                if anim is not None:
                    solution.draw_network(anim)
            
            # no improvement found
            if round(length_diff_best, op_diff_round_digits) == 0:
                break

            
        return solution

    def operator_cross(self, graph, solution, op_diff_round_digits):
        # TODO: check docstring
        """applies Cross inter-route operator to solution

        Takes every node from every route and calculates savings when inserted
        into all possible positions in other routes. Insertion is done at
        position with max. saving and procedure starts over again with newly
        created graph as input. Stops when no improvement is found.
        
        Parameters
        ----------
        graph: :networkx:`NetworkX Graph Obj< >`
            Descr
        solution: BaseSolution
            Descr
        op_diff_round_digits: float
            Precision (floating point digits) for rounding route length differences.
            
            *Details*: In some cases when an exchange is performed on two routes with one node each,
            the difference between the both solutions (before and after the exchange) is not zero.
            This is due to internal rounding errors of float type. So the loop won't break
            (alternating between these two solutions), we need an additional criterion to avoid
            this behaviour: A threshold to handle values very close to zero as if they were zero
            (for a more detaisled description of the matter see http://floating-point-gui.de or
            https://docs.python.org/3.5/tutorial/floatingpoint.html)
        
        Returns
        -------
        LocalSearchSolution
           A solution (LocalSearchSolution class)

        Todo
        ----
        * allow moves of a 2-node chain
        * Remove ugly nested loops, convert to more efficient matrix operations
        """

        # shorter var names for loop
        dm = graph._matrix
        dn = graph._nodes

    def benchmark_operator_order(self, graph, solution, op_diff_round_digits):
        """performs all possible permutations of route improvement and prints graph length
        
        Parameters
        ----------
        graph: :networkx:`NetworkX Graph Obj< >`
            A NetworkX graaph is used.
        solution: BaseSolution
            BaseSolution instance
        op_diff_round_digits: float
            Precision (floating point digits) for rounding route length differences.
            
            *Details*: In some cases when an exchange is performed on two routes with one node each,
            the difference between the both solutions (before and after the exchange) is not zero.
            This is due to internal rounding errors of float type. So the loop won't break
            (alternating between these two solutions), we need an additional criterion to avoid
            this behaviour: A threshold to handle values very close to zero as if they were zero
            (for a more detailed description of the matter see http://floating-point-gui.de or
            https://docs.python.org/3.5/tutorial/floatingpoint.html)
        """
        
        operators = {self.operator_exchange: 'exchange',
                     self.operator_relocate: 'relocate',
                     self.operator_oropt: 'oropt'}
                     
        for op in it.permutations(operators):
            solution = solution.clone()
            solution = op[0](graph, solution, op_diff_round_digits)
            solution = op[1](graph, solution, op_diff_round_digits)
            solution = op[2](graph, solution, op_diff_round_digits)
            logger.info("{0} {1} {2} => Length: {3}".format(
                operators[op[0]],
                operators[op[1]],
                operators[op[2]],
                solution.length()))

    def solve(self, graph, savings_solution, timeout, debug=False, anim=None):
        """Improve initial savings solution using local search

        Parameters
        ----------
        graph: :networkx:`NetworkX Graph Obj< >`
            Graph instance
        savings_solution: SavingsSolution
            initial solution of CVRP problem (instance of `SavingsSolution` class)
        timeout: :obj:`int`
            max processing time in seconds
        debug: bool, defaults to False
            If True, information is printed while routing
        anim: AnimationDing0
            AnimationDing0 object
        
        Returns
        -------
        LocalSearchSolution
           A solution (LocalSearchSolution class)

        """
        # TODO: If necessary, use timeout to set max processing time of local search

        # load threshold for operator (see exchange or relocate operator's description for more information)
        op_diff_round_digits = int(cfg_ding0.get('mv_routing', 'operator_diff_round_digits'))

        solution = LocalSearchSolution(graph, savings_solution)

        # FOR BENCHMARKING OF OPERATOR'S ORDER:
        #self.benchmark_operator_order(graph, savings_solution, op_diff_round_digits)

        for run in range(10):
            start = time.time()
            solution = self.operator_exchange(graph, solution, op_diff_round_digits, anim)
            time1 = time.time()
            if debug:
                logger.debug('Elapsed time (exchange, run {1}): {0}, '
                             'Solution\'s length: {2}'.format(
                    time1 - start, str(run), solution.length()))

            solution = self.operator_relocate(graph, solution, op_diff_round_digits, anim)
            time2 = time.time()
            if debug:
                logger.debug('Elapsed time (relocate, run {1}): {0}, '
                             'Solution\'s length: {2}'.format(
                    time2 - time1, str(run), solution.length()))

            solution = self.operator_oropt(graph, solution, op_diff_round_digits, anim)
            time3 = time.time()
            if debug:
                logger.debug('Elapsed time (oropt, run {1}): {0}, '
                             'Solution\'s length: {2}'.format(
                    time3 - time2, str(run), solution.length()))

        return solution
