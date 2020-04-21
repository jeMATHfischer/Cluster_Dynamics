import numpy as np
import networkx as nx
from agent_class import agent
from cluster_class import cluster
import matplotlib.pyplot as plt
#from edge_class import edge
import random

class dynamics():
    
    def __init__(self, population_size,  labels, network, theta, mu,p):
        self.population_size = population_size
        self.labels = labels
        self.population = [agent(i,labels[i],list(nx.neighbors(network,i)),-1) for i in range(population_size)]
        self.network = network
        self.theta = theta
        self.mu = mu
        self.p = p
        self.clusters = []
        self.remaining_cluster_labels = list(range(1,population_size+1))
        self.existing_cluster_labels = []
        self.t = 0

    def cluster_setup(self):
        '''
        The cluster_setup function allows the creation of initial maximum confidence clusters
        acoording to definition in Chen2020. 
        The Void graph reduces the initial network to a possibly disconnected network of short edges,
        .i.e., edges with length smaller than theta. Paths within these graphs correspond to
        the connecting features in the definition of Chen2020. The connected components of Void
        then yield the confidence clusters. The loop simply sets the label for each individual agent in each
        connected component.
        '''
        Void = nx.empty_graph(self.population_size)
        short_edges = [edge for edge in self.network.edges() if 
                       abs(self.population[edge[0]].opinion-self.population[edge[1]].opinion) < self.theta]
        Void.add_edges_from(short_edges)
        representatives = self.population.copy()
        while len(representatives) > 0:
            agent_x = random.choice(representatives)
            label = random.choice(self.remaining_cluster_labels)
            agent_x_adjoints = [self.population[i] for i in Void.nodes() if nx.has_path(Void,agent_x.name, i)]
            for agent_a in agent_x_adjoints:
                agent_a.cluster = label
                representatives.remove(agent_a)
            
            self.remaining_cluster_labels.remove(label)
            self.existing_cluster_labels.append(label)
        print('cluster setup')
        print(self.remaining_cluster_labels)
                    
    def get_cluster(self):
        '''
        This function allows to display all clusters as sets of the contained 
        agents. Both cluster name and name of the agent are displayed as a touple.
        '''
        return [[agent.cluster for agent in self.population if agent.cluster == label] for label in self.existing_cluster_labels]
        
    def adjust_labels(self,agent_a,agent_b):
        '''
        Helper function to avoid cluttering in the evolution step. This corresponds
        to a short edge being drawn and both connected agents approach each other
        '''
        agent_a.opinion = (1-self.mu)*agent_a.opinion + self.mu*agent_b.opinion
        agent_b.opinion = (1-self.mu)*agent_b.opinion + self.mu*agent_a.opinion
            
    def split_cluster(self,label):
        '''
        In the case of an update of a short edge it is possible that a cluster
        being earlier connected by paths with segments of length shorter than theta
        does no longer satisfy this property.
        Hence the cluster has to be split into multiple ones and new labels have to be assigned. 
        The small_void network plays the same role as the Void network in the cluster
        setup but now we are only considering the network spanned within each cluster. Hence
        small void. The remaining part works identical to the cluster setup function.        
        '''
        self.existing_cluster_labels.remove(label)
        self.remaining_cluster_labels.append(label)
        concerned_agents = []
        for agent_a in self.population:
            if agent_a.cluster == label:
                agent_a.cluster = -1
                concerned_agents.append(agent_a.name)
        small_void = nx.empty_graph(len(concerned_agents))
        node_names = dict(zip(list(range(len(concerned_agents))), concerned_agents))
        nx.relabel_nodes(small_void, node_names, copy=False)
        natural_edges = [edges for edges in self.network.edges() if 
                         edges[0] in concerned_agents and edges[1] in concerned_agents and abs(self.population[edges[0]].opinion-self.population[edges[1]].opinion) < self.theta]
        small_void.add_edges_from(natural_edges)
        representatives = [self.population[i] for i in concerned_agents]
        while len(representatives) > 0:            
            agent_x = random.choice(representatives)
            label_new = random.choice(self.remaining_cluster_labels)
            agent_x_adjoints = [self.population[i] for i in small_void.nodes() if nx.has_path(small_void,agent_x.name, i)]
            for agent_a in agent_x_adjoints:
                agent_a.cluster = label_new
                representatives.remove(agent_a)
            self.remaining_cluster_labels.remove(label_new)
            self.existing_cluster_labels.append(label_new)
        
    def combine_clusters(self, label_a, label_b):
        '''
        In linking two individuals with close opinions, who did not belong the same
        cluster beforehand we may reduce the number of clusters. This is due to the fact
        that the rewiring may create new short paths containing exclusively short
        segements. 
        Therefore, we throw out the old cluster label and assign a new one identical
        for each agent within the combined cluster.
        '''
        self.existing_cluster_labels.remove(label_a)
        self.existing_cluster_labels.remove(label_b)
        self.remaining_cluster_labels = self.remaining_cluster_labels + [label_a,label_b]
        label_new = random.choice(self.remaining_cluster_labels)
        for agent_a in self.population:
            if agent_a.cluster == label_a or agent_a.cluster == label_b:
                agent_a.cluster = label_new
        self.existing_cluster_labels.append(label_new)
        self.remaining_cluster_labels.remove(label_new)
        
    
    def time_evolution(self, final_time):
        '''
        Up to the final time the flipping-approaching dynamics are simulated. 
        It is a direct implementation of the algorithmic dynamics, which can be found in 
        the paper.
        '''
        while self.t < final_time:
#            tester = len(self.existing_cluster_labels)
            e = random.choice(list(self.network.edges()))
            if abs(self.population[e[0]].opinion - self.population[e[1]].opinion) < self.theta:
                self.adjust_labels(self.population[e[0]], self.population[e[1]])
                self.split_cluster(self.population[e[0]].cluster)
            else:
                c2 = np.random.binomial(1,0.5)
                while True: 
                    x = random.choice(list(self.network.nodes()))
                    if (x,e[c2]) not in self.network.edges():
            #                 G.add_edge(x,rm[0])
                        break
                c1 = np.random.binomial(1,self.p)
                if c1 == 1:
                    self.network.remove_edge(e[0],e[1])
                    self.network.add_edge(x,e[c2]) 
                    if abs(self.population[x].opinion-self.population[e[c2]].opinion) < self.theta and self.population[x].cluster != self.population[e[c2]].cluster:
                        self.combine_clusters(self.population[x].cluster, self.population[e[c2]].cluster)
                        
#            if tester != len(self.existing_cluster_labels):
#                print(self.get_cluster())
            self.t += 1
            
#        print(self.get_cluster())
#        print([agent.opinion for agent in self.population])
#            
        
        
        
        
        
        
        
        
        
        
        
        
        
            