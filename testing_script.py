import numpy as np
import networkx as nx
from dynamics_class import dynamics
import matplotlib.pyplot as plt

k = 1000
X = np.loadtxt('initial_opinions_n_10.txt')
n = len(X)
G = nx.read_gpickle(path='initial_graph_edgelist_m_30.txt')
nx.draw(G, with_labels = True) 
plt.show()

theta = 0.4
mu = 0.1
p = 1


T_end_no_flip = []
T_end_flip = [] 

for button in [0,1]:
    for s in range(k):
        D = dynamics(n,X,G,theta,mu,p, with_flip = button)
        
        D.cluster_setup()
        D.time_evolution(100000)
        if button == 0:
            T_end_no_flip.append(D.t)
        else:
            T_end_flip.append(D.t)
    print(s)
    
plt.scatter(X,np.ones(len(X)))
plt.show()    
    
plt.hist(T_end_no_flip, density=True, bins=int(len(np.unique(T_end_no_flip))/3)+1)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.title('Dynamics without flipping')
plt.savefig("hist_dynamics_without_flipping_t_{}_m_{}.png".format(theta,mu))
plt.show()

plt.hist(T_end_flip, density=True, bins=int(len(np.unique(T_end_no_flip))/3)+1)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.title('Dynamics with flipping')
plt.savefig("hist_dynamics_with_flipping_t_{}_m_{}.png".format(theta,mu))
plt.show()