# This file provides some useful functions for debugging DINGO

import dingo
import os.path as path
import networkx as nx


def compare_graphs(graph, mode):
    """ Compares graph with saved one which is loaded via networkx' gpickle
    Args:
        graph:
        mode: 'write' or 'compare'

    Returns:

    """

    # get path
    package_path = dingo.__path__[0]
    file = path.join(package_path, 'output', 'debug', 'graph.gpickle')

    if mode is 'write':
        try:
            nx.write_gpickle(graph, file)
            print('=====> DEBUG: Graph written to', file)
        except:
            raise FileNotFoundError('Could not write to file', file)

    elif mode is 'compare':
        try:
            graph2 = nx.read_gpickle(file)
            print('=====> DEBUG: Graph read from', file)
        except:
            raise FileNotFoundError('File not found:', file)

        # get data
        nodes1 = sorted(graph.nodes(), key=lambda _: repr(_))
        nodes2 = sorted(graph2.nodes(), key=lambda _: repr(_))
        edges1 = sorted(graph.edges(), key=lambda _: repr(_))
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
            print('Edge count in graph 1 > egde count in graph 2')
            print('Difference:', [edge for edge in edges1 if (repr(edge) not in repr(edges2)) and (repr(tuple(reversed(edge))) not in repr(edges2))])
            graphs_are_isomorphic = False
        elif len(edges2) > len(edges1):
            print('Edge count in graph 2 > edge count in graph 1')
            print('Difference:', [edge for edge in edges2 if (repr(edge) not in repr(edges1)) and (repr(tuple(reversed(edge))) not in repr(edges1))])
            graphs_are_isomorphic = False

        if graphs_are_isomorphic:
            print('=====> DEBUG: Graphs are isomorphic')
        else:
            print('=====> DEBUG: Graphs are NOT isomorphic')

    else:
        raise ValueError('Invalid value for mode, use mode=\'write\' or \'compare\'')

    exit(0)
