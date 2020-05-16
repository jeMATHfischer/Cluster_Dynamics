import numpy as np
import networkx as nx
from dynamics_class import dynamics
import matplotlib.pyplot as plt

k = 10000
n = 10
m = 30
X = np.linspace(0.1,0.9,n)
#X = np.random.uniform(0,1,n)
G = nx.complete_graph(n)
#G = nx.gnm_random_graph(n,m)
plt.show()

theta = 0.4
mu = 0.1
p = 0.3

print(theta)

T_end_no_flip = []
T_end_flip = [] 

for button in [0,1]:
    for s in range(k):
        D = dynamics(n,X,G,theta,mu,p, with_flip = button)
        
        D.cluster_setup()
        
        #coloring = [agent.opinion for agent in D.population]
        #    
        #cmap = plt.cm.viridis
        #two_d_positions = [(agent.opinion, agent.cluster/len(D.existing_cluster_labels)) for agent in D.population]
        #pos = dict(zip(list(range(n)), two_d_positions))
        #
        #plt.figure(figsize=(7,7))
        #nx.draw(D.network, pos = pos, node_color=coloring, cmap = cmap, with_labels = True)
        #plt.axis('equal')
        #plt.show()
        
        clusters = [agent.cluster for agent in D.population]
        #print("labels = {}".format(X))
#        print("initial_clusters {}".format(clusters))
        
        D.time_evolution(100000)
        if button == 0:
            T_end_no_flip.append(D.t)
        else:
            T_end_flip.append(D.t)
        #coloring = [agent.opinion for agent in D.population]
        #    
        #cmap = plt.cm.viridis
        #two_d_positions = [(agent.opinion, agent.cluster/len(D.existing_cluster_labels)) for agent in D.population]
        #pos = dict(zip(list(range(n)), two_d_positions))
        #
        #plt.figure(figsize=(7,7))
        #nx.draw(D.network, pos = pos, node_color=coloring, cmap = cmap, with_labels = False)
        #plt.axis('equal')
        #plt.show()
        
        
#        clusters = [agent.cluster for agent in D.population]
        #print("labels = {}".format(X))
#        print("final_clusters {}".format(clusters))
#        
#        print(D.cluster_changes)
#        print(D.cluster_references)
#        
    print(s)
    
plt.hist(T_end_no_flip, density=True, bins=int(len(np.unique(T_end_no_flip))/3))  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.title('Dynamics without flipping')
plt.show()

plt.hist(T_end_flip, density=True, bins=int(len(np.unique(T_end_no_flip))/3))  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.title('Dynamics with flipping')
plt.show()