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


# This file provides some useful functions for debugging DING0

import ding0
import os.path as path
import networkx as nx


def compare_graphs(graph1, mode, graph2=None):
    """ Compares graph with saved one which is loaded via networkx' gpickle

    Parameters
    ----------
    graph1 : networkx.graph
        First Ding0 MV graph for comparison
    graph2 : networkx.graph
        Second Ding0 MV graph for comparison. If a second graph is not provided
        it will be laoded from disk with hard-coded file name.
    mode: 'write' or 'compare'

    Returns:

    """

    # get path
    package_path = ding0.__path__[0]
    file = path.join(package_path, 'output', 'debug', 'graph1.gpickle')

    if mode == 'write':
        try:
            nx.write_gpickle(graph1, file)
            print('=====> DEBUG: Graph written to', file)
        except:
            raise FileNotFoundError('Could not write to file', file)

    elif mode == 'compare':
        if graph2 is None:
            try:
                graph2 = nx.read_gpickle(file)
                print('=====> DEBUG: Graph read from', file)
            except:
                raise FileNotFoundError('File not found:', file)

        # get data
        nodes1 = sorted(graph1.nodes(), key=lambda _: repr(_))
        nodes2 = sorted(graph2.nodes(), key=lambda _: repr(_))
        edges1 = sorted(graph1.edges(), key=lambda _: repr(_))
        edges2 = sorted(graph2.edges(), key=lambda _: repr(_))

        graphs_are_isomorphic = True

        # check nodes
        if len(nodes1) > len(nodes2):
            print('Node count in graph 1 > node count in graph 2')
            print('Difference:', [node for node in nodes1 if repr(node) not in repr(nodes2)])
            graphs_are_isomorphic = False
        elif len(nodes2) > len(nodes1):
            print('Node count in graph 2 > node count in graph 1')
            print('Difference:', [node for node in nodes2 if repr(node) not in repr(nodes1)])
            graphs_are_isomorphic = False

        # check edges
        if len(edges1) > len(edges2):
            print('Edge count in graph 1 > edge count in graph 2')
            print('Difference:', [edge for edge in edges1 if (repr(edge) not in repr(edges2)) and
                                  (repr(tuple(reversed(edge))) not in repr(edges2))])
            graphs_are_isomorphic = False
        elif len(edges2) > len(edges1):
            print('Edge count in graph 2 > edge count in graph 1')
            print('Difference:', [edge for edge in edges2 if (repr(edge) not in repr(edges1)) and
                                  (repr(tuple(reversed(edge))) not in repr(edges1))])
            graphs_are_isomorphic = False
        elif (len(edges1) == len(edges1)) and (len([edge for edge in edges1 if (repr(edge) not in repr(edges2)) and
            (repr(tuple(reversed(edge))) not in repr(edges2))]) > 0):
            print('Edge count in graph 1 = edge count in graph 2')
            print('Difference:', [edge for edge in edges2 if (repr(edge) not in repr(edges1)) and
                                  (repr(tuple(reversed(edge))) not in repr(edges1))])
            graphs_are_isomorphic = False

        if graphs_are_isomorphic:
            print('=====> DEBUG: Graphs are isomorphic')
        else:
            print('=====> DEBUG: Graphs are NOT isomorphic')

    else:
        raise ValueError('Invalid value for mode, use mode=\'write\' or \'compare\'')

    exit(0)
