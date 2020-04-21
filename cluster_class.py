import numpy as np

class cluster():
    
    def __init__(self, label):
        self.label = label
        
    def belongs_to_cluster(self, candidates):
        return [agent for agent in candidates if agent.cluster == self.label]
    
    def cluster_size(self, population):
        return len(self.belongs_to_cluster(population))
    
    