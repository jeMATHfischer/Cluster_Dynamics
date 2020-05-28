import numpy as np
import networkx as nx
from dynamics_class import dynamics
import matplotlib.pyplot as plt
from itertools import product

X = np.loadtxt('initial_opinions_n_10.txt')
n = len(X)

p = 1
k = 50

G = nx.read_gpickle(path='initial_graph_edgelist_m_30.txt')

precison = 20
theta = np.linspace(0.1,1,precison)
mu = np.linspace(0.1,0.25,precison)

def f(param):
    T_end_no_flip = []
    T_end_flip = [] 
    for button in [0,1]:
        print(button)
        for s in range(k):
            D = dynamics(n,X,G,param[0],param[1],p, with_flip = button)
            D.cluster_setup()
            D.time_evolution(100000)
            if button == 0:
                T_end_no_flip.append(D.t)
            else:
                T_end_flip.append(D.t)
        
    return sum(T_end_no_flip)/len(T_end_no_flip), sum(T_end_flip)/len(T_end_flip)


c = list(product(theta, mu))    
dt= np.dtype('float,float')     
M = np.reshape(np.array(c,dtype=dt),(precison,precison))

print(M)
Z = np.vectorize(f)
Z = Z(M)
print(Z)

plt.title('mean hitting time without flipping')
cp = plt.imshow(Z[0][::-1], cmap=plt.cm.viridis, vmin = 0)
plt.colorbar(cp)
#plt.xticks(theta)
#plt.yticks(mu)
plt.xlabel('theta index')
plt.ylabel('mu index')
plt.show()

plt.title('mean hitting time with flipping')
cp = plt.imshow(Z[1][::-1], cmap=plt.cm.viridis, vmin = 0)
plt.colorbar(cp)
plt.xlabel('theta index')
plt.ylabel('mu index')
#plt.xticks(theta)
#plt.yticks(mu)
plt.show()
