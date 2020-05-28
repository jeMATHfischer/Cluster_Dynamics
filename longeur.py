import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

X = np.loadtxt('initial_opinions_n_10.txt')

G = nx.read_gpickle(path='initial_graph_edgelist_m_30.txt')
nx.draw(G, with_labels = True) 
plt.show()

for item in G.edges():
    print('{} has length {}'.format(item, abs(X[item[0]]-X[item[1]])))