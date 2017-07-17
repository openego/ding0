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


import networkx as nx
import matplotlib.pyplot as plt


class BaseSolution(object):
    """Base abstract class for a CVRP solution"""

    def __init__(self, cvrp_problem):
        """Initialize class

        Parameters:
            cvrp_problem: Graph instance
        """
        self._problem = cvrp_problem
        self._allocated = 0

    def get_pair(self, pair):
        i, j = pair
        return (self._nodes[i.name()], self._nodes[j.name()])

    def is_complete(self):
        """Returns True if this is a complete solution, i.e, all nodes are allocated"""
        return all(
            [node.route_allocation() is not None for node in list(self._nodes.values()) if node != self._problem.depot()]
        )

    def clone(self):
        """Returns a deep copy of self

        Clones:
            routes
            allocation
            nodes
        """

        new_solution = self.__class__(self._problem, len(self._routes))

        # Clone routes
        for index, r in enumerate(self._routes):
            new_route = new_solution._routes[index]
            for node in r.nodes():
                # Insere new node on new route
                new_node = new_solution._nodes[node]
                new_route.allocate([new_node])

        return new_solution

    def routes(self):
        """Returns a generator for iterating over solution routes"""
        for r in self._routes:
            yield r

    def length(self):
        """Returns the solution length (or cost)"""
        length = 0
        for r in self._routes:
            length = length + r.length()

        return length

    def can_process(self, pairs):
        """Returns True if this solution can process `pairs`

        Parameters:
            pairs: List of pairs
        """
        raise NotImplementedError()

    def process(self, node_or_pair):
        """Processes a node or a pair of nodes into the current solution

        MUST CREATE A NEW INSTANCE, NOT CHANGE ANY INSTANCE ATTRIBUTES

        Returns a new instance (deep copy) of self object
        """
        raise NotImplementedError()

    def draw_network(self, anim):
        """draws solution's graph using networkx"""

        g = nx.Graph()
        ntemp = []
        nodes_pos = {}
        demands = {}
        demands_pos = {}
        for no, node in self._nodes.items():
            g.add_node(node)
            ntemp.append(node)
            coord = self._problem._coord[no]
            nodes_pos[node] = tuple(coord)
            demands[node] = 'd=' + str(node.demand())
            demands_pos[node] = tuple([a+b for a, b in zip(coord, [2.5]*len(coord))])

        depot = self._nodes[self._problem._depot._name]
        for r in self.routes():
            n1 = r._nodes[0:len(r._nodes)-1]
            n2 = r._nodes[1:len(r._nodes)]
            e = list(zip(n1, n2))
            e.append((depot, r._nodes[0]))
            e.append((r._nodes[-1], depot))
            g.add_edges_from(e)

        plt.figure()

        if anim is not None:
            nx.draw_networkx(g, nodes_pos, with_labels=False, node_size=50)
            plt.savefig(anim.file_path + anim.file_prefix + (4 - len(str(anim.counter))) * '0' + str(anim.counter) + '.png')
            anim.counter += 1
            plt.close()
        else:
            nx.draw_networkx(g, nodes_pos)
            nx.draw_networkx_labels(g, demands_pos, labels=demands)
            plt.show()


class BaseSolver(object):
    """Base algorithm solver class"""

    def solve(self, data, vehicles, timeout):
        """Must solves the CVRP problem

        Parameters:
            data: Graph instance
            vehicles: Vehicles number
            timeout: max processing time in seconds

        Must return BEFORE timeout

        Must returns a solution (BaseSolution class derived)
        """
        raise NotImplementedError()
