import numpy as np 
import networkx as nx

np.random.seed(42)


for i in range(10):
    n = 10
    m = 30
    #    G = nx.complete_graph(n)
    G = nx.gnm_random_graph(n,m, seed = 42)
    nx.write_gpickle(G, path='initial_graph_edgelist_m_{}_i_{}.txt'.format(m, i))