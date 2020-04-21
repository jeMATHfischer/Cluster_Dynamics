import numpy as np

class agent():
    
    def __init__(self, name, opinion, neighbors, cluster):
        self.name = name
        self.neighbors = neighbors
        self.opinion = opinion
        self.cluster = cluster
        
    def change_cluster(self, new_cluster):
        pass