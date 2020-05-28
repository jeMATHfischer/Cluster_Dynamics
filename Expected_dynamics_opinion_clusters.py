import numpy as np
import networkx as nx
from dynamics_class import dynamics
import matplotlib.pyplot as plt

k = 1000
X = np.loadtxt('initial_opinions_n_10.txt')
n = len(X)
G = nx.read_gpickle(path='initial_graph_edgelist_m_30.txt')

nx.draw(G)
plt.show()

theta = 0.2
mu = 0.4
p = 1

T_end_no_flip = []
T_end_flip = [] 

Cluster_no_flip = []
Cluster_flip = []

for button in [0,1]:
    for s in range(k):
        D = dynamics(n,X,G,theta,mu,p, with_flip = button)
        
        D.cluster_setup()
        D.time_evolution(100000)
        if button == 0:
            T_end_no_flip.append(D.t)
        else:
            T_end_flip.append(D.t)
    
        if button == 0:
            Cluster_no_flip.append(D.cluster_changes)
        else:
            Cluster_flip.append(D.cluster_changes) 

with open("k_{}_Cluster_no_flip_t_{}_m_{}.txt".format(k,theta,mu), "w") as output:
    output.write(str(Cluster_no_flip))
    
with open("k_{}_Cluster_flip_t_{}_m_{}.txt".format(k,theta,mu), "w") as output:
    output.write(str(Cluster_flip))

T_max = max([max([time for cluster, op, time in item]) for item in (Cluster_flip + Cluster_no_flip)])
times = np.arange(0,T_max+5)

test_index = 0
for item in Cluster_no_flip:
    item_times = [time for cluster, op, time in item]
    t_val = max(item_times)
    d = np.zeros((1,len(times)))
    if test_index == 0: 
        if len(item_times)-1 == 0:
            d[0,:] = len(item[len(item_times)-1][0])
        else:
            for i in range(len(item_times)-1):
                if i == 0:
                    d[0,times < item_times[i+1]] = len(item[i][0])
                else:
                    d[0, np.logical_and(item_times[i] <= times, times < item_times[i+1])] = len(item[i][0])
            d[0,times >= t_val] = len(item[len(item_times)-1][0])
        cluster_numbers_no_flip = d 
        test_index += 1
    else:
        if len(item_times)-1 == 0:
            d[0,:] = len(item[len(item_times)-1][0])
        else:
            for i in range(len(item_times)-1):
                if i == 0:
                    d[0,times < item[i+1][2]] = len(item[i][0])
                else:
                    d[0, np.logical_and(item_times[i] <= times, times < item_times[i+1])] = len(item[i][0])
            d[0,times >= t_val] = len(item[len(item_times)-1][0])
        cluster_numbers_no_flip = np.append(cluster_numbers_no_flip, d, axis = 0)

mean_cluster_numbers_no_flip = 1/k*np.sum(cluster_numbers_no_flip, axis = 0)

test_index = 0
for item in Cluster_flip:
    item_times = [time for cluster, op, time in item]
    t_val = max(item_times)
    d = np.zeros((1,len(times)))
    if test_index == 0: 
        if len(item_times)-1 == 0:
            d[0,:] = len(item[len(item_times)-1][0])
        else:
            for i in range(len(item_times)-1):
                if i == 0:
                    d[0,times < item[i+1][2]] = len(item[i][0])
                else:
                    d[0, np.logical_and(item_times[i] <= times, times < item_times[i+1])] = len(item[i][0])
            d[0,times >= t_val] = len(item[len(item_times)-1][0])
        cluster_numbers_flip = d 
        test_index += 1
    else:
        if len(item_times)-1 == 0:
            d[0,:] = len(item[len(item_times)-1][0])
        else:
            for i in range(len(item_times)-1):
                if i == 0:
                    d[0,times <= item_times[i+1]] = len(item[i][0])
                else:
                    d[0, np.logical_and(item_times[i] <= times, times < item_times[i+1])] = len(item[i][0])
            d[0,times >= t_val] = len(item[len(item_times)-1][0])
        cluster_numbers_flip = np.append(cluster_numbers_flip, d, axis = 0)

mean_cluster_numbers_flip = 1/k*np.sum(cluster_numbers_flip, axis = 0)

print(mean_cluster_numbers_flip)

plt.step(times, mean_cluster_numbers_no_flip )
plt.title('Mean Cluster Number without Flipping')
plt.savefig("k_{}_mean_cluster_number_without_flipping_t_{}_m_{}.png".format(k,theta,mu))
plt.xlabel('Time')
plt.ylabel('Mean Number of Clusters')
plt.show()


plt.step(times, mean_cluster_numbers_flip )
plt.title('Mean Cluster Number with Flipping')
plt.savefig("k_{}_mean_cluster_number_with_flipping_t_{}_m_{}.png".format(k,theta,mu))
plt.xlabel('Time')
plt.ylabel('Mean Number of Clusters')
plt.show()

    
    