import numpy as np
import networkx as nx
from dynamics_class import dynamics
import matplotlib.pyplot as plt

n = 3
m = 3
X = np.random.uniform(0,1,n)
G = nx.complete_graph(n)
#G = nx.gnm_random_graph(n,m)
plt.show()

theta = 0.8
mu = 0.7
p = 0.3

D = dynamics(n,X,G,theta,mu, p)

D.cluster_setup()

coloring = [agent.opinion for agent in D.population]
    
cmap = plt.cm.viridis
two_d_positions = [(agent.opinion, agent.cluster/n) for agent in D.population]
pos = dict(zip(list(range(n)), two_d_positions))

plt.figure(figsize=(7,7))
nx.draw(D.network, pos = pos, node_color=coloring, cmap = cmap, with_labels = True)
plt.axis('equal')
plt.show()



clusters = [agent.cluster for agent in D.population]
#print("labels = {}".format(X))
print("initial_clusters {}".format(clusters))

D.time_evolution(1000)
print(D.get_cluster())
coloring = [agent.opinion for agent in D.population]
    
cmap = plt.cm.viridis
two_d_positions = [(agent.opinion, agent.cluster/n) for agent in D.population]
pos = dict(zip(list(range(n)), two_d_positions))

plt.figure(figsize=(7,7))
nx.draw(D.network, pos = pos, node_color=coloring, cmap = cmap, with_labels = False)
plt.axis('equal')
plt.show()


clusters = [agent.cluster for agent in D.population]
#print("labels = {}".format(X))
print("initial_clusters {}".format(clusters))