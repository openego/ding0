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

from dingo.tools import config as cfg_dingo

from math import pi, tan, acos


class Route(object):
    """
    CVRP route, consists of consecutive nodes
    -----------------------------------------
    bla
    """

    def __init__(self, cvrp_problem):
        """Class constructor

        Initialize route

        Parameters:

        """
        self._problem = cvrp_problem
        self._demand = 0
        self._nodes = []

    def clone(self):
        """Returns a deep copy of self

        Clones:
            allocation
            nodes
        """

        new_route = self.__class__(self._problem)
        for node in self.nodes():
            # Insere new node on new route
            new_node = node.__class__(node._name, node._demand)
            new_route.allocate([new_node])

        return new_route

    def demand(self):
        """Returns the current route demand"""
        return self._demand

    def nodes(self):
        """Returns a generator for iterating over nodes"""
        for node in self._nodes:
            yield node

    def length(self):
        """Returns the route length (cost)"""
        cost = 0
        depot = self._problem.depot()

        last = depot
        for i in self._nodes:
            a, b = last, i
            if a.name() > b.name():
                a, b = b, a

            cost = cost + self._problem.distance(a, b)
            last = i

        cost = cost + self._problem.distance(depot, last)

        return cost

    def can_allocate(self, nodes, pos=None):
        """Returns True if this route can allocate nodes in `nodes` list"""

        # clone route and nodes
        new_route = self.clone()
        new_nodes = [node.clone() for node in nodes]
        if pos is None:
            pos = len(self._nodes)
        new_route._nodes = new_route._nodes[:pos] + new_nodes + new_route._nodes[pos:]
        new_route._demand = sum([node.demand() for node in new_route._nodes])

        if new_route.tech_constraints_satisfied():
            return True

        return False

    def allocate(self, nodes, append=True):
        """Allocates all nodes from `nodes` list in this route"""

        nodes_demand = 0
        for node in [node for node in nodes]:
            if node._allocation:
                node._allocation.deallocate([node])

            node._allocation = self
            nodes_demand = nodes_demand + node.demand()
            if append:
                self._nodes.append(node)
            else:
                self._nodes.insert(0, node)

        self._demand = self._demand + nodes_demand

    def deallocate(self, nodes):
        """Deallocates all nodes from `nodes` list from this route"""

        nodes_demand = 0
        for node in nodes:
            self._nodes.remove(node)
            node._allocation = None
            nodes_demand = nodes_demand + node.demand()

        self._demand = self._demand - nodes_demand

        if self._demand < 0:
            raise Exception('Trying to deallocate more than previously allocated')
    
    def insert(self, nodes, pos):
        """Inserts all nodes from `nodes` list into this route at position `pos`"""
        
        node_list = []
        nodes_demand = 0
        for node in [node for node in nodes]:
            if node._allocation:
                node._allocation.deallocate([node])
            node_list.append(node)
            node._allocation = self
            nodes_demand = nodes_demand + node.demand()

        self._nodes = self._nodes[:pos] + node_list + self._nodes[pos:]
        self._demand += nodes_demand

    def is_interior(self, node):
        """Returns True if node is interior to the route, i.e., not adjascent to depot"""
        return self._nodes.index(node) != 0 and self._nodes.index(node) != len(self._nodes) - 1

    def last(self, node):
        """Returns True if node is the last node in the route"""
        return self._nodes.index(node) == len(self._nodes) - 1

    def calc_circuit_breaker_position(self, debug=False):
        """ Calculates the optimal position of a circuit breaker on route.

        Returns:
            position of circuit breaker on route (index of last node on 1st half-ring preceding the circuit breaker)

        Notes
        -----
        According to planning principles of MV grids, a MV ring is run as two strings (half-rings) separated by a
        circuit breaker which is open at normal operation.
        Assuming a ring (route which is connected to the root node at either sides), the optimal position of a circuit
        breaker is defined as the position (virtual cable) between two nodes where the conveyed current is minimal on
        the route. Instead of the peak current, the peak load is used here (assuming a constant voltage).

        The circuit breakers are used here for checking tech. constraints only and will be re-located after connection
        of satellites and stations in dingo.grid.mv_grid.tools.set_circuit_breakers

        References
        ----------

        """
        # TODO: add references (Tao)

        # set init value
        demand_diff_min = 10e6

        # check possible positions in route
        for ctr in range(len(self._nodes)):
            # split route and calc demand difference
            route_demand_part1 = sum([node.demand() for node in self._nodes[0:ctr]])
            route_demand_part2 = sum([node.demand() for node in self._nodes[ctr:len(self._nodes)]])
            demand_diff = abs(route_demand_part1 - route_demand_part2)

            if demand_diff < demand_diff_min:
                demand_diff_min = demand_diff
                position = ctr

        if debug:
            print('sum 1=', sum([node.demand() for node in self._nodes[0:position]]))
            print('sum 2=', sum([node.demand() for node in self._nodes[position:len(self._nodes)]]))
            print('Position of circuit breaker: ', self._nodes[position-1], '-', self._nodes[position],
                  '(sumdiff=', demand_diff_min, ')')

        return position
        
    def tech_constraints_satisfied(self):
        """ Check route validity according to technical constraints (voltage and current rating)
        
        Constraints:
            current rating of cable/line
            voltage stability at all nodes

        Notes:
            The validation is done for every tested MV grid configuration during CVRP algorithm. The current rating is
            checked using load factors from [1]_. Due to the high amount of steps the voltage rating cannot be checked
            using load flow calculation. Therefore we use a simple method which determines the voltage change between
            two consecutive nodes according to [2]_.
            Furthermore it is checked if new route has got more nodes than allowed (typ. 2*10 according to _[3]).

        References:
        .. [1] Deutsche Energie-Agentur GmbH (dena), "dena-Verteilnetzstudie. Ausbau- und Innovationsbedarf der
            Stromverteilnetze in Deutschland bis 2030.", 2012
        .. [2] M. Sakulin, W. Hipp, "Netzaspekte von dezentralen Erzeugungseinheiten,
            Studie im Auftrag der E-Control GmbH", TU Graz, 2004
        .. [3] Klaus Heuck et al., "Elektrische Energieversorgung", Vieweg+Teubner, Wiesbaden, 2007
        """

        # load parameters
        load_area_count_per_ring = float(cfg_dingo.get('mv_routing',
                                                       'load_area_count_per_ring'))

        if self._problem._branch_kind == 'line':
            load_factor_normal = float(cfg_dingo.get('assumptions',
                                                     'load_factor_mv_line_lc_normal'))
            load_factor_malfunc = float(cfg_dingo.get('assumptions',
                                                      'load_factor_mv_line_lc_malfunc'))
        elif self._problem._branch_kind == 'cable':
            load_factor_normal = float(cfg_dingo.get('assumptions',
                                                     'load_factor_mv_cable_lc_normal'))
            load_factor_malfunc = float(cfg_dingo.get('assumptions',
                                                      'load_factor_mv_cable_lc_malfunc'))
        else:
            raise ValueError('Grid\'s _branch_kind is invalid, could not use branch parameters.')

        mv_max_v_level_diff_normal = float(cfg_dingo.get('mv_routing_tech_constraints',
                                                         'mv_max_v_level_diff_normal'))
        mv_max_v_level_diff_malfunc = float(cfg_dingo.get('mv_routing_tech_constraints',
                                                          'mv_max_v_level_diff_malfunc'))
        mv_routing_loads_cos_phi = float(cfg_dingo.get('mv_routing_tech_constraints',
                                                       'mv_routing_loads_cos_phi'))


        # step 0: check if route has got more nodes than allowed
        if len(self._nodes) > load_area_count_per_ring:
            return False

        # step 1: calc circuit breaker position
        position = self.calc_circuit_breaker_position()

        # step 2: calc required values for checking current & voltage
        # get nodes of half-rings
        nodes_hring1 = [self._problem._depot] + self._nodes[0:position]
        nodes_hring2 = list(reversed(self._nodes[position:len(self._nodes)] + [self._problem._depot]))
        # get all nodes of full ring for both directions
        nodes_ring1 = [self._problem._depot] + self._nodes
        nodes_ring2 = list(reversed(self._nodes + [self._problem._depot]))
        # factor to calc reactive from active power
        Q_factor = tan(acos(mv_routing_loads_cos_phi))
        # line/cable params per km
        r = self._problem._branch_type['R']  # unit for r: ohm/km
        x = self._problem._branch_type['L'] * 2*pi * 50 / 1e3  # unit for x: ohm/km

        # step 3a: check if current rating of default cable/line is violated
        # (for every of the 2 half-rings using load factor for normal operation)
        demand_hring_1 = sum([node.demand() for node in self._nodes[0:position]])
        demand_hring_2 = sum([node.demand() for node in self._nodes[position:len(self._nodes)]])
        peak_current_sum_hring1 = demand_hring_1 / (3**0.5) / self._problem._v_level  # units: kVA / kV = A
        peak_current_sum_hring2 = demand_hring_2 / (3**0.5) / self._problem._v_level  # units: kVA / kV = A

        if (peak_current_sum_hring1 > (self._problem._branch_type['I_max_th'] * load_factor_normal) or
            peak_current_sum_hring2 > (self._problem._branch_type['I_max_th'] * load_factor_normal)):
            return False

        # step 3b: check if current rating of default cable/line is violated
        # (for full ring using load factor for malfunction operation)
        peak_current_sum_ring = self._demand / (3**0.5) / self._problem._v_level  # units: kVA / kV = A
        if peak_current_sum_ring > (self._problem._branch_type['I_max_th'] * load_factor_malfunc):
            return False

        # step 4a: check voltage stability at all nodes
        # (for every of the 2 half-rings using max. voltage difference for normal operation)

        # get operation voltage level from station
        v_level_hring1 =\
            v_level_hring2 =\
            v_level_ring_dir1 =\
            v_level_ring_dir2 =\
            v_level_op =\
            self._problem._v_level_operation * 1e3

        for n1, n2 in zip(nodes_hring1[0:len(nodes_hring1)-1], nodes_hring1[1:len(nodes_hring1)]):
            v_level_hring1 -= n2.demand() * 1e3 * self._problem.distance(n1, n2) * (r + x*Q_factor) / v_level_hring1
            if (v_level_op - v_level_hring1) > (v_level_op * mv_max_v_level_diff_normal):
                return False

        for n1, n2 in zip(nodes_hring2[0:len(nodes_hring2)-1], nodes_hring2[1:len(nodes_hring2)]):
            v_level_hring2 -= n2.demand() * 1e3 * self._problem.distance(n1, n2) * (r + x*Q_factor) / v_level_hring2
            if (v_level_op - v_level_hring2) > (v_level_op * mv_max_v_level_diff_normal):
                return False

        # step 4b: check voltage stability at all nodes
        # (for full ring calculating both directions simultaneously using max. voltage diff. for malfunction operation)
        for (n1, n2), (n3, n4) in zip(zip(nodes_ring1[0:len(nodes_ring1)-1], nodes_ring1[1:len(nodes_ring1)]),
                                      zip(nodes_ring2[0:len(nodes_ring2)-1], nodes_ring2[1:len(nodes_ring2)])):
            v_level_ring_dir1 -= (n2.demand() * 1e3 * self._problem.distance(n1, n2) * (r + x*Q_factor) /
                                  v_level_ring_dir1)
            v_level_ring_dir2 -= (n4.demand() * 1e3 * self._problem.distance(n3, n4) * (r + x*Q_factor) /
                                  v_level_ring_dir2)
            if ((v_level_op - v_level_ring_dir1) > (v_level_op * mv_max_v_level_diff_malfunc) or
                (v_level_op - v_level_ring_dir2) > (v_level_op * mv_max_v_level_diff_malfunc)):
                return False

        return True

        # TODO (mv_routing): create circuit breaker object, open ring after routing


    def __str__(self):
        return str(self._nodes)

    def __repr__(self):
        return str(self._nodes)


class Node(object):
    """
    CVRP node (MV transformer/customer)
    -----------------------------------
    bla
    """

    def __init__(self, name, demand):
        """Class constructor

        Initialize demand

        Parameters:
            name: Node name
            demand: Node demand
        """
        self._name = name
        self._demand = demand
        self._allocation = None

    def clone(self):
        """Returns a deep copy of self

        Clones:
            allocation
            nodes
        """

        new_node = self.__class__(self._name, self._demand)

        return new_node

    def name(self):
        """Returns node name"""
        return self._name

    def demand(self):
        """Returns the node demand"""
        return self._demand

    def route_allocation(self):
        """Returns the route which node is allocated"""
        return self._allocation

    def __str__(self):
        return str(self._name)

    def __repr__(self):
        return str(self._name)

    def __cmp__(self, other):
        if isinstance(other, Node):
            return self._name - other._name

        return self._name - other

    def __hash__(self):
        return self._name.__hash__()


class Graph(object):
    """Class for modelling a CVRP problem data"""
    """
    CVRP graph
    ----------
    bla
    """

    def __init__(self, data):
        """Class constructor

        Initialize all nodes, edges and depot

        Parameters:
            data: TSPLIB parsed data
        """
        
        self._coord = data['NODE_COORD_SECTION']
        self._nodes = {i: Node(i, data['DEMAND'][i]) for i in data['MATRIX']}
        self._matrix = {}
        self._depot = None
        self._branch_kind = data['BRANCH_KIND']
        self._branch_type = data['BRANCH_TYPE']
        self._v_level = data['V_LEVEL']
        self._v_level_operation = data['V_LEVEL_OP']
        self._is_aggregated = data['IS_AGGREGATED']

        for i in data['MATRIX']:

            x = self._nodes[i]
            self._matrix[x] = {}

            if i == data['DEPOT']:
                self._depot = x # x, not i!!

            for j in data['MATRIX']:
                y = self._nodes[j]

                self._matrix[x][y] = data['MATRIX'][i][j]

        if self._depot is None:
            raise Exception('Depot not found')

    def nodes(self):
        """Returns a generator for iterating over nodes"""
        for i in sorted(self._nodes):
            yield self._nodes[i]

    def edges(self):
        """Returns a generator for iterating over edges"""
        for i in sorted(self._matrix.keys(), key=lambda x:x.name()):
            for j in sorted(self._matrix[i].keys(), key=lambda x:x.name()):
                if i != j:
                    yield (i, j)

    def depot(self):
        """Returns the depot node"""
        return self._depot

    def distance(self, i, j):
        """Returns the distance between node i and node j"""

        a, b = i, j

        if a.name() > b.name():
            a, b = b, a
        
        return self._matrix[self._nodes[a.name()]][self._nodes[b.name()]]
