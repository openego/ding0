from ding0.core.network.cable_distributors import LVCableDistributorDing0


def overloading(graph):
    """
    Check a grid for line overloading due to current exceeding I_th_max

    Parameters
    ----------
    graph : networkx.Graph
        Graph structure as container for a grid topology including its equipment

    Returns
    -------
    overloaded : tuple
        Pairwise edges of graph a maximum occuring current
    """

    # return overloaded


def get_branches(grid):
    """
    Individual graphs of sectoral loads

    :param geid:
    :return:
    """

    station = grid._station

    tree = nx.dfs_tree(grid._graph, station)

    # TODO: idea
    # 1. build tree from lv_grid station as root -> diretions should point to
    # descending leafs
    # 2. for analysis of current issues get list of descendants with
    # nx.descendants(tree, station). Sum peak load / gen capacity
    # 3. Extract nodes belonging to main route of a branch by checking all
    # successors if these are LVCalbleDistributors
    # notes and hints:
    # 1. associated edges can be accessed via grid._graph.in_edges(<node>)
    # respectively grid._graph.out_edges(<node>)
    # 2. when using nx.descendants(tree, station) make sure the order of nodes
    # is maintained as this is important to properly assess voltage and over-
    # loading issues

    # first_cbl_dists = [x for x in grid._graph.neighbors(station)
    #                    if isinstance(x, LVCableDistributorDing0)]


    # if len(first_cbl_dists) > 0:
    #     ancestors =  nx.ancestors(grid._graph, first_cbl_dists[0])
    # else:
    #     ancestors = None
    # return ancestors
    branch_heads = nx.neighbors(tree, station)

    descendants = {branch_head: list(nx.descendants(tree, branch_head)) for
                   branch_head in branch_heads}



    return descendants
