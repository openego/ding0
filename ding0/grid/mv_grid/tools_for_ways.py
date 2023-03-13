import networkx as nx
import pandas as pd

from ding0.grid.lv_grid.graph_processing import simplify_graph_adv, remove_unloaded_deadends, remove_parallels_and_loops




def reduce_graph_for_dist_matrix_calc(graph, nodes_to_keep):
    
    '''
    simplifies graph, until number of graph nodes between two iterations
    is almost the same (ratio=0.95), keeps nodes_of_interest in graph
    during simplification
    returns graph

    '''
    
    post_graph_no, pre_graph_no = 0, len(graph)
    
    while post_graph_no/pre_graph_no <= 0.95: 
        pre_graph_no = len(graph)
        graph = simplify_graph_adv(graph, nodes_to_keep)
        graph = remove_parallels_and_loops(graph)
        graph = remove_unloaded_deadends(graph, nodes_to_keep)
        post_graph_no = len(graph)
    
    return graph



def calc_street_dist_matrix(G, matrix_node_list):

    node_list = matrix_node_list.copy()
    node_list.extend(list(set(G.nodes()) - set(matrix_node_list)))

    dm_floyd = nx.floyd_warshall_numpy(G, nodelist=node_list, weight='length')
    df_floyd = pd.DataFrame(dm_floyd)
    #reduce matrix
    df_floyd = df_floyd.iloc[:len(matrix_node_list),:len(matrix_node_list)]
    df_floyd = df_floyd.div(1000) # unit is km
    df_floyd = df_floyd.round(3) # accuracy in m
    # pass osmid nodes as indices
    df_floyd = df_floyd.set_index(pd.Series(matrix_node_list))
    df_floyd.columns = matrix_node_list
    # transform matrix to dict
    matrix_dict = df_floyd.to_dict('dict')
    
    return matrix_dict


