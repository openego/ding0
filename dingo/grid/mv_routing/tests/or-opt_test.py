# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:20:18 2016

@author: jonathan

Or-Opt Test
"""

import itertools as it
import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
rnd.seed()

plt.close('all')

# =================== FUNCTIONS =======================
def calc_costs(tour, cost_matrix):
    costs_tour = 0
    for edge in zip(tour,tour[1:len(tour)]):            
        costs_tour += cost_matrix[edge]
    return costs_tour

def draw_nx(tour, pos, cost_matrix):
    g = nx.Graph()
    g.add_nodes_from(pos.keys())
    g.add_edges_from(zip(tour,tour[1:len(tour)]))
    
    plt.figure()
    nx.draw_networkx(g, pos)
    labels = {}
    for edge in g.edges():
        labels[edge] = str(cost_matrix[edge])
    nx.draw_networkx_edge_labels(g, pos, labels)
    #plt.show()

# =================== START =======================
nodes = [1,2,3,4,5,6,7,8,9]
#nodes = [1,2,3,4,5,6]
tour = [1,2,3,4,5,6,7,8,9,1]
#tour = [1,2,3,4,5,6,1]
pos = {1: (4, 2), 2: (2, 3), 3: (-1, 3), 4: (3, 1), 5: (0, -1), 6: (-0.5, 2), 7: (2, 2), 8: (2, -1), 9: (0, 0)}
#pos = {1: (4, 2), 2: (2, 3), 3: (-1, 3), 4: (3, 1), 5: (0, -1), 6: (-0.5, 2)}
#tour = {(1,2), (2,3), (3,4), (4,5), (5,6), (6,1)}
n = len(nodes)

cost_matrix = {}
for way in it.combinations(nodes, 2):
    a,b = way
    cost = round(sum((x-y)**2 for x, y in zip(pos[a], pos[b]))**0.5, 2)
    cost_matrix[a,b] = cost
    cost_matrix[b,a] = cost

draw_nx(tour, pos, cost_matrix)

ctr = 1
while True:
    cost = calc_costs(tour, cost_matrix)
    cost_best = cost
    
    for s in range(3,0,-1):
        for i in range(1,n-s):
            cost_diff = cost - cost_matrix[(tour[i-1],tour[i])] - cost_matrix[(tour[i+s-1],tour[i+s])] + cost_matrix[(tour[i-1],tour[i+s])]
            for j in range(i+s+1,n+1):
                #aaa = [s,i,j]
                #print(s,i,j)
                if j == 6:
                    j2 = 1
                else:
                    j2 = j+1
                cost_neu = cost_diff + cost_matrix[(tour[j-1],tour[i])] + cost_matrix[(tour[i+s-1],tour[j2-1])] - cost_matrix[(tour[j-1],tour[j2-1])]
                if cost_neu < cost_best:
                    cost_best = cost_neu
                    i_best = i
                    j_best = j
                    s_best = s
    if cost_best < cost:
        tour = tour[0:i_best] + tour[i_best+s_best:j_best] + tour[i_best:i_best+s_best] + tour[j_best:n+1]
        print('Tour:', tour, '=>', cost_best, '(', s_best, i_best, j_best, ')')
        ctr += 1
    if cost_best == cost:
        draw_nx(tour, pos, cost_matrix)
        break


