from __future__ import division
import random
# from snap import *
import numpy as np
import numpy.linalg as alg
import scipy as spy
import networkx as nx

import time
from itertools import *
import sys

from scipy.special import expit


class GibbsSampler:
    '''
    class for GibbsSampler, called during E step to calcualte the marginal latent 
    class distribution for nodes and edges of grahphical models.
    '''
    def __init__(self,temp,b,num_Samples,burn_In,Lambda,z_0):
        self.temp = temp
        self.b = b   
        self.num_nodes = nx.number_of_nodes(self.temp)
        self.num_edges = nx.number_of_edges(self.temp)
        self.num_Samples = num_Samples
        self.burn_In = burn_In
        self.num_Iterations = self.num_Samples + self.burn_In
        self.samples = np.zeros((self.num_Samples,self.num_nodes))
        self.Z = z_0
        self.Lambda = Lambda
        self.node_prob = np.zeros((self.num_nodes,2))
        self.edge_prob = np.zeros((self.num_edges,4))
    
    def update(self):
        for NI in self.temp.nodes_iter():
            alpha1 = 0
            alpha2 = 0
            for Id in self.temp.neighbors(NI):
        #         print NI.GetId(),Id
                if self.Z[Id] ==1 :
                    alpha1 += self.Lambda * (self.b[NI,0] - self.b[Id,0])**2
                else:
                    alpha2 += self.Lambda * (self.b[NI,1] - self.b[Id,1])**2
#                m = np.exp(alpha1)
#                n = np.exp(alpha2)
# np.exp(alpha1)/(np.exp(alpha1)+np.exp(alpha2))
                if random.random() < expit(alpha1-alpha2):
                    self.Z[NI] = 1
                else:
                    self.Z[NI] = 0
        return self.Z

    def sampling(self):
        display_step = 1000
        for iteration in range(self.num_Iterations):
            ZZ = self.update()
            # print ZZ
#            if iteration%display_step == 0:
#                print iteration
            if iteration >= self.burn_In:
                self.samples[iteration-self.burn_In,:] = ZZ
            # if iteration%display_step==0:
            # #     print iteration
         #    if iteration >= self.burn_In:
         #        self.samples[iteration-self.burn_In,:] = ZZ


    def cal_edge(self):
        for EI in self.temp.edges_iter():
        #     print EI
            A = self.samples[:,EI[0]]
            B = self.samples[:,EI[1]]
            count11 = np.sum([B[A==1]==1])
            count1_1 = np.sum([B[A==1]==0])
            count_1_1 = np.sum([B[A==0]==0])
            count_11 = np.sum([B[A==0]==1])
            p11 = count11/len(A)
            p_11 = count_11/len(A)
            p1_1 = count1_1/len(A)
            p_1_1 = count_1_1/len(A)
        #     print np.array([p11,p_11,p1_1,p_1_1])
            self.temp[EI[0]][EI[1]]['edge_prob']=np.array([p11,p_11,p1_1,p_1_1])

    def cal_node(self):
        for NI in self.temp.nodes_iter():
            A = self.samples[:,NI]
            count1 = [A[A==1]][0].shape[0]
            count_1 = [A[A==0]][0].shape[0]
            p1 = count1/len(A)
            p_1 = count_1/len(A)
            self.temp.node[NI]['node_prob'] = np.array([p1,p_1])
    
    def get_edge(self):
        count = 0
        for EI in self.temp.edges_iter():
            self.edge_prob[count,:] = self.temp[EI[0]][EI[1]]['edge_prob']
            count += 1





    def get_node(self):
        for NI in self.temp.nodes_iter():
            self.node_prob[NI,:] = self.temp.node[NI]['node_prob']

