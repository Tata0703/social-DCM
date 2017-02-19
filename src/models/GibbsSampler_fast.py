from __future__ import division
import random
import numpy as np
import numpy.linalg as alg
import scipy as spy

import time
from itertools import *
import sys


class GibbsSampler:
    '''
    class for GibbsSampler, called during E step to calcualte the marginal latent 
    class distribution for nodes.
    '''
    def __init__(self,A,w,b,num_samples,burn_in,Lambda,display_step,num_classes,neighbors,z_0):
        self.A = A
        self.b = b   
        self.w = w
        size = A.shape
        n = size[0]
        self.num_nodes = n
        #self.num_edges = int(A.nnz/2)
        self.num_edges = int(np.count_nonzero(A)/2)
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.burn_in = burn_in
        self.num_iterations = self.num_samples + self.burn_in
        self.z = z_0
        self.Lambda = Lambda
        self.node_freq = np.zeros((self.num_nodes,self.num_classes))
        self.node_probs = np.zeros((self.num_nodes,self.num_classes))
        self.edge_freq = np.zeros((self.num_edges,self.num_classes))
        self.edge_probs = np.zeros((self.num_edges,self.num_classes))
        self.display_step = display_step
        self.neighbors = neighbors

    def sampling(self):
     
        numerators = np.zeros(self.num_classes)
        probs = np.zeros(self.num_classes)
    
        for iteration in range(self.num_iterations):
            for i in range(self.num_nodes):
                for j in self.neighbors[i]:
                    for k in range(self.num_classes):
                        if self.z[j] == k:
                            numerators[k] += (self.b[i,k] - self.b[j,k])**2 + np.sum(np.square(self.w[i,k,:] - self.w[j,k,:]))
                denominator = 0
                
                for k in range(self.num_classes):
                    numerators[k] = np.exp(-self.Lambda*numerators[k])
                    denominator += numerators[k]
                    
                for k in range(self.num_classes):
                    probs[k] = numerators[k]/denominator
                    
                random_number = random.uniform(0,1)
                cum_sum_probs = np.cumsum(probs)
                
                # Generate samples and compute frequency for nodes
                for k in range(self.num_classes):
                    if random_number <= cum_sum_probs[k]:
                        self.z[i] = k
                        if iteration >= self.burn_in:
                            self.node_freq[i,k] += 1
                        break   
                        
                for k in range(self.num_classes):
                    numerators[k] = 0
                    probs[k] = 0
                    
            # Compute frequency for edges            
            if iteration >= self.burn_in:            
                for k in range(self.num_classes): 
                    idx = 0
                    for i in range(self.num_nodes):
                        for j in self.neighbors[i]:
                            if j > i:
                                if self.z[i] == k & self.z[j] == k:
                                    self.edge_freq[idx,k] += 1
                                idx = idx + 1
                                
        # Compute probabilities for edges
        for k in range(self.num_classes): 
            idx = 0
            for i in range(self.num_nodes):
                for j in self.neighbors[i]:
                    if j > i:
                        self.edge_probs[idx,k] = self.edge_freq[idx,k]/self.num_samples
                        idx = idx + 1  
                        
        # Compute probabilities for nodes                
        for i in range(self.num_nodes):
            for k in range(self.num_classes):
                self.node_probs[i,k] = self.node_freq[i,k]/self.num_samples
                    
