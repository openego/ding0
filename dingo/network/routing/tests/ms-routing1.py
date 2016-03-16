import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
rnd.seed()

ns_node_power_vals = [50,100,160,250,400,630]
ms_node_power_vals = [25,50,100,160,250]

ns_nodes_size=[]
ns_nodes_pos={}
ms_node_pos={}
ns_nodes = {}
for i in range(1,101):
    ns_nodes.update( {'ons'+str(i):{'pos':( round(rnd.uniform(1, 10),2), round(rnd.uniform(1, 10),2) ) }} )
    power = rnd.choice(ns_node_power_vals)
    ns_nodes['ons'+str(i)]['power'] = power
for n, p in ns_nodes.items():
    ns_nodes_pos[n] = ns_nodes[n]['pos']
    ns_nodes_size.append(ns_nodes[n]['power']/10)

ms_node_size = next(i for i in ms_node_power_vals if i>(sum(ns_nodes_size)/1000))*10
ms_node = {'usw':{'pos':(5,5), 'power':ms_node_size}}
ms_node_pos['usw'] = ms_node['usw']['pos']

# create nx graph, add nodes and attributes
graph = nx.Graph()

graph.add_nodes_from(ns_nodes.keys())
graph.add_nodes_from(ms_node.keys())

#nx.set_node_attributes(graph, 'power', ns_nodes.values())

# draw
nx.draw_networkx_nodes(graph, ns_nodes_pos, ns_nodes.keys(), node_shape="o", node_color="r",
                       node_size = ns_nodes_size)

nx.draw_networkx_nodes(graph, ms_node_pos, ms_node.keys(), node_shape="o", node_color="b",
                       node_size = ms_node['usw']['power'])
plt.show()
