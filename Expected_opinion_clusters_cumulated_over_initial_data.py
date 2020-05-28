import numpy as np
import networkx as nx
from dynamics_class import dynamics
import matplotlib.pyplot as plt
from itertools import product

run = 0 
Tt_max = []

a = 10
b = 10

for o1,o2 in product(list(range(a)),list(range(b))):
    k = 1000
    X = np.loadtxt('initial_opinions_n_10_i_{}.txt'.format(o1))
    n = len(X)
    G = nx.read_gpickle(path='initial_graph_edgelist_m_30_i_{}.txt'.format(o2))
    
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
    Tt_max.append(T_max+5)
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
    
    if run == 0:
        run += 1
        cluster_exp_value_no_flip = np.reshape(mean_cluster_numbers_no_flip, (1,-1))
        cluster_exp_value_flip = np.reshape(mean_cluster_numbers_flip, (1,-1))
    else:
        run += 1
        values_of_interest_no_flip = np.reshape(mean_cluster_numbers_no_flip[:min(Tt_max[:run])], (1,-1))
        cluster_exp_value_no_flip  = np.append(cluster_exp_value_no_flip[:,:min(Tt_max[:run])], values_of_interest_no_flip, axis = 0)
        values_of_interest_flip = np.reshape(mean_cluster_numbers_flip[:min(Tt_max[:run])], (1,-1))
        cluster_exp_value_flip  = np.append(cluster_exp_value_flip[:,:min(Tt_max[:run])], values_of_interest_flip, axis = 0)
    
    print(run)
    
np.savetxt('cluster_exp_value_no_flip.txt', 1/(run+1)*np.sum(cluster_exp_value_no_flip, axis = 0))
np.savetxt('cluster_exp_value_flip.txt', 1/(run+1)*np.sum(cluster_exp_value_flip, axis = 0))


plt.step(np.arange(len(np.sum(cluster_exp_value_no_flip, axis = 0))), 1/(run+1)*np.sum(cluster_exp_value_no_flip, axis = 0))
plt.title('Mean Cluster Number without Flipping aggregated over initial Data')
plt.savefig("k_{}_mean_cluster_number_without_flipping_t_{}_m_{}_initial_data_{}_{}.png".format(k,theta,mu, a,b))
plt.xlabel('Time')
plt.ylabel('Mean Number of Clusters')
plt.show()


plt.step(np.arange(len(np.sum(cluster_exp_value_flip, axis = 0))), 1/(run+1)*np.sum(cluster_exp_value_flip, axis = 0))
plt.title('Mean Cluster Number with Flipping aggregated over initial Data')
plt.savefig("k_{}_mean_cluster_number_with_flipping_t_{}_m_{}_initial_data_{}_{}.png".format(k,theta,mu, a,b))
plt.xlabel('Time')
plt.ylabel('Mean Number of Clusters')
plt.show()

    
    