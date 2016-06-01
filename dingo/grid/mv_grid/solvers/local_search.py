"""
    Copyright 2016 openego development group
    Licensed under GNU General Public License 3.0. See the LICENSE file at the
    top-level directory of this distribution or obtain a copy of the license at
    http://www.gnu.org/licenses/gpl-3.0.txt
    
    Based on code by Romulo Oliveira copyright (C) 2015,
    https://github.com/RomuloOliveira/monte-carlo-cvrp
    Originally licensed under the Apache License, Version 2.0. You may obtain a
    copy of the license at http://www.apache.org/licenses/LICENSE-2.0
"""

#import operator
import time
import itertools as it

from dingo.grid.mv_grid.models import models
from dingo.grid.mv_grid.solvers.base import BaseSolution, BaseSolver

from dingo.tools import config as cfg_dingo

class LocalSearchSolution(BaseSolution):
    """Solution class for Local Search metaheuristic"""

    def __init__(self, cvrp_problem, solution):
        super(LocalSearchSolution, self).__init__(cvrp_problem)
        
        self._nodes = solution._nodes
        self._routes = solution._routes

    def clone(self):
        """Returns a deep copy of self

        Clones:
            routes
            allocation
            nodes
        """

        new_solution = self.__class__(self._problem, self._vehicles)

        # Clone routes
        for index, r in enumerate(self._routes):
            new_route = new_solution._routes[index] = models.Route(self._problem, self._problem.capacity())
            for node in r.nodes():
                # Insere new node on new route
                new_node = new_solution._nodes[node.name()]
                new_route.allocate([new_node])

        # remove empty routes from new solution
        new_solution._routes = [route for route in new_solution._routes if route._nodes]

        return new_solution

    def is_complete(self):
        """Returns True if this is a complete solution, i.e, all nodes are allocated
        TO BE REVIEWED, CURRENTY NOT IN USE        
        """
        allocated = all(
            [node.route_allocation() is not None for node in list(self._nodes.values()) if node.name() != self._problem.depot().name()]
        )

        valid_routes = len(self._routes) == self._vehicles

        valid_demands = all([route.demand() <= route.capacity() for route in self._routes])

        #return allocated and valid_routes and valid_demands
        return 0

class LocalSearchSolver(BaseSolver):
    """ Improve initial savings solution using local search
    
        Graph operators:
            Or-Opt (intra-route)
            Relocate (inter-route)
            Exchange (inter-route)
           
        ToDo:
        -----
            * Cross (inter-route) - to remove crossing edges between two routes

        References
        ----------
        .. [1] W. Wenger, "Multikriterielle Tourenplanung", Dissertation, 2009
        .. [2] M. Kämpf, "Probleme der Tourenbildung", Chemnitzer Informatik-Berichte, 2006
        .. [3] O. Bräysy, M. Gendreau, "Vehicle Routing Problem with Time Windows,
            Part I: Route Construction and Local Search Algorithms",
            Transportation Science, vol. 39, Issue 1, pp. 104-118, 2005
        .. [4] C. Boomgaarden, "Dynamische Tourenplanung und -steuerung",
            Dissertation, 2007

    """
    # TODO: Cross (inter-route), see above
    
    def operator_oropt(self, graph, solution, op_diff_round_digits):
        """applies Or-Opt intra-route operator to solution
        
        Takes chains of nodes (length=3..1 consecutive nodes) from a given
        route and calculates savings when inserted into another position on the
        same route (all possible positions). Performes best move (max. saving)
        and starts over again with new route until no improvement is found.
        
        Returns a solution (LocalSearchSolution class))

        Args:
            op_diff_round_digits: Precision (floating point digits) for rounding route length differences.
                                  Details: In some cases when an exchange is performed on two routes with one node each,
                                  the difference between the both solutions (before and after the exchange) is not zero.
                                  This is due to internal rounding errors of float type. So the loop won't break
                                  (alternating between these two solutions), we need an additional criterion to avoid
                                  this behaviour: A threshold to handle values very close to zero as if they were zero
                                  (for a more detailed description of the matter see http://floating-point-gui.de or
                                  https://docs.python.org/3.5/tutorial/floatingpoint.html)

        (Inner) Loop variables:
            s: length (count of consecutive nodes) of the chain that is moved. Values: 3..1
            i: node that precedes the chain before moving (position in the route `tour`, not node name)
            j: node that precedes the chain after moving (position in the route `tour`, not node name)
        
        ToDo:
            * insert literature reference for Or-algorithm here
            * Remove ugly nested loops, convert to more efficient matrix operations
        """
        
        # shorter var names for loop
        dm = graph._matrix
        dn = graph._nodes
        
        for route in solution.routes():
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
                # no improvement found
                if length_best == length:
                    # replace old route by new (same arg for allocation and deallocation since node order is considered at allocation)
                    solution._routes[solution._routes.index(route)].deallocate(tour[1:-1])
                    solution._routes[solution._routes.index(route)].allocate(tour[1:-1])
                    
                    break
        
        #solution = LocalSearchSolution(solution, graph, new_routes)
        return solution
    
    def operator_relocate(self, graph, solution, op_diff_round_digits):
        """applies Relocate inter-route operator to solution
        
        Takes every node from every route and calculates savings when inserted
        into all possible positions in other routes. Insertion is done at
        position with max. saving and procedure starts over again with newly
        created graph as input. Stops when no improvement is found.
        
        Returns a solution (LocalSearchSolution class))

        Args:
            op_diff_round_digits: Precision (floating point digits) for rounding route length differences.
                                  Details: In some cases when an exchange is performed on two routes with one node each,
                                  the difference between the both solutions (before and after the exchange) is not zero.
                                  This is due to internal rounding errors of float type. So the loop won't break
                                  (alternating between these two solutions), we need an additional criterion to avoid
                                  this behaviour: A threshold to handle values very close to zero as if they were zero
                                  (for a more detailed description of the matter see http://floating-point-gui.de or
                                  https://docs.python.org/3.5/tutorial/floatingpoint.html)
        
        (Inner) Loop variables:
            i: node that is checked for possible moves (position in the route `tour`, not node name)
            j: node that precedes the insert position in target route (position in the route `target_tour`, not node name)
            
        ToDo:
            * Remove ugly nested loops, convert to more efficient matrix operations
        """
        # shorter var names for loop
        dm = graph._matrix
        dn = graph._nodes        
        
        # Relocate: Search better solutions by checking possible node moves
        while True:
            length_diff_best = 0
            
            for route in solution.routes():
                # create tour by adding depot at start and end
                tour = [graph._depot] + route._nodes + [graph._depot]
                
                for target_route in solution.routes():
                    target_tour = [graph._depot] + target_route._nodes + [graph._depot]
                    
                    if route == target_route:
                        continue
                    
                    n = len(route._nodes)
                    nt = len(target_route._nodes)+1

                    for i in range(0,n):
                        node = route._nodes[i]
                        for j in range(0,nt):
                            #target_node = target_route._nodes[j]
                            
                            #if target_route.can_allocate([node]):
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
                
                #print('Bessere Loesung gefunden:', node_best, target_node_best, target_route_best, length_diff_best)
            
            # no improvement found
            if round(length_diff_best, op_diff_round_digits) == 0:
                break

            
        return solution
        
    def operator_exchange(self, graph, solution, op_diff_round_digits):
        """applies Exchange inter-route operator to solution
        
        Takes every node from every route and calculates savings when inserted
        into all possible positions in other routes. Insertion is done at
        position with max. saving and procedure starts over again with newly
        created graph as input. Stops when no improvement is found.
        
        Returns a solution (LocalSearchSolution class))

        Args:
            op_diff_round_digits: Precision (floating point digits) for rounding route length differences.
                                  Details: In some cases when an exchange is performed on two routes with one node each,
                                  the difference between the both solutions (before and after the exchange) is not zero.
                                  This is due to internal rounding errors of float type. So the loop won't break
                                  (alternating between these two solutions), we need an additional criterion to avoid
                                  this behaviour: A threshold to handle values very close to zero as if they were zero
                                  (for a more detailed description of the matter see http://floating-point-gui.de or
                                  https://docs.python.org/3.5/tutorial/floatingpoint.html)
        
        (Inner) Loop variables:
            i: node that is checked for possible moves (position in the route `tour`, not node name)
            j: node that precedes the insert position in target route (position in the route `target_tour`, not node name)
            
        ToDo:
            * allow moves of a 2-node chain
            * Remove ugly nested loops, convert to more efficient matrix operations
        """

        # shorter var names for loop
        dm = graph._matrix
        dn = graph._nodes        
        
        # Exchange: Search better solutions by checking possible node exchanges
        while True:
            length_diff_best = 0
            
            for route in solution.routes():
                # create tour by adding depot at start and end
                tour = [graph._depot] + route._nodes + [graph._depot]
                
                for target_route in solution.routes():
                    target_tour = [graph._depot] + target_route._nodes + [graph._depot]
                    
                    if route == target_route:
                        continue
                    
                    n = len(route._nodes)
                    nt = len(target_route._nodes)

                    for i in range(0,n):
                        node = route._nodes[i]
                        for j in range(0,nt):
                            target_node = target_route._nodes[j]
                            
                            #if route.can_exchange_nodes(target_route, [node], [target_node]):
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
                # insert new node
                target_route_best.insert([node_best], j_best)
                route_best.insert([target_node_best], i_best)
                # remove empty routes from solution
                solution._routes = [route for route in solution._routes if route._nodes]
            
            # no improvement found
            if round(length_diff_best, op_diff_round_digits) == 0:
                break

            
        return solution

    def operator_cross(self, graph, solution, op_diff_round_digits):
        """applies Cross inter-route operator to solution

        Takes every node from every route and calculates savings when inserted
        into all possible positions in other routes. Insertion is done at
        position with max. saving and procedure starts over again with newly
        created graph as input. Stops when no improvement is found.

        Returns a solution (LocalSearchSolution class))

        Args:
            op_diff_round_digits: Precision (floating point digits) for rounding route length differences.
                                  Details: In some cases when an exchange is performed on two routes with one node each,
                                  the difference between the both solutions (before and after the exchange) is not zero.
                                  This is due to internal rounding errors of float type. So the loop won't break
                                  (alternating between these two solutions), we need an additional criterion to avoid
                                  this behaviour: A threshold to handle values very close to zero as if they were zero
                                  (for a more detailed description of the matter see http://floating-point-gui.de or
                                  https://docs.python.org/3.5/tutorial/floatingpoint.html)


        ToDo:
            * allow moves of a 2-node chain
            * Remove ugly nested loops, convert to more efficient matrix operations
        """

        # shorter var names for loop
        dm = graph._matrix
        dn = graph._nodes




    def benchmark_operator_order(self, graph, solution, op_diff_round_digits):
        """performs all possible permutations of route improvement and prints graph length"""
        
        operators = {self.operator_exchange: 'exchange',
                     self.operator_relocate: 'relocate',
                     self.operator_oropt: 'oropt'}
                     
        for op in it.permutations(operators):
            solution = solution.clone()
            solution = op[0](graph, solution, op_diff_round_digits)
            solution = op[1](graph, solution, op_diff_round_digits)
            solution = op[2](graph, solution, op_diff_round_digits)
            print(operators[op[0]], '+', operators[op[1]], '+', operators[op[2]], '=> Length:', solution.length())

    def solve(self, graph, savings_solution, timeout):
        """Improve initial savings solution using local search

        Parameters:
            graph: Graph instance
            savings_solution: initial solution of CVRP problem (instance of `SavingsSolution` class)
            timeout: max processing time in seconds

        Returns a solution (LocalSearchSolution class))
        """
        # TODO: If necessary, use timeout to set max processing time of local search

        # load threshold for operator (see exchange or relocate operator's description for more information)
        op_diff_round_digits = int(cfg_dingo.get('mv_routing', 'operator_diff_round_digits'))

        solution = LocalSearchSolution(graph, savings_solution)

        # FOR BENCHMARKING:
        #self.benchmark_operator_order(graph, savings_solution, op_diff_round_digits)

        start = time.time()

        #solution = self.operator_oropt(graph, solution, op_diff_round_digits)

        for i in range(10):
            solution = self.operator_exchange(graph, solution, op_diff_round_digits)
            time1 = time.time()
            print('Elapsed time (exchange): {}'.format(time1 - start))

            solution = self.operator_relocate(graph, solution, op_diff_round_digits)
            time2 = time.time()
            print('Elapsed time (relocate): {}'.format(time2 - time1))

            solution = self.operator_oropt(graph, solution, op_diff_round_digits)
            time3 = time.time()
            print('Elapsed time (oropt): {}'.format(time3 - time2))

        return solution
