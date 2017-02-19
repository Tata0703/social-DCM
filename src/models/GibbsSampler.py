from __future__ import division
import random
# from snapvx import *
from snap import *
import numpy as np
import numpy.linalg as alg
import scipy as spy

import time
from itertools import *
import sys



num_nodes = 100


class GibbsSampler:
	'''
	class for GibbsSampler, called during E step to calcualte the marginal latent 
	class distribution for nodes and edges of grahphical models.
	'''
	def __init__(self,G1,b,num_Iterations,burn_In):
		self.G1 = G1
		self.b = b    # instance variable unique to each instance
		# one sample of latent class label for each individual
		# self.Z = 0
		self.num_Iterations = num_Iterations
		self.burn_In = burn_In
		self.samples = np.zeros((num_Samples,num_nodes))

	def update(self):
		for NI in self.G.Nodes():
			alpha1 = 0
			alpha2 = 0
			for Id in NI.GetOutEdges():
				if Z[Id] ==1 :
					alpha1 += Lambda * (b[NI.GetId(),0] - b[Id,0])**2
				else:
					alpha2 += Lambda * (b[NI.GetId(),1] - b[Id,1])**2
				m = np.exp(alpha1)
				n = np.exp(alpha2)
				if random.random() < m/(m+n):
					self.Z[NI.GetId()] = 1
				else:
					self.Z[NI.GetId()] = -1

	def sampling(self):
		display_step = 1000
		for iteration in range(self.num_Iterations):
			if iteration%display_step ==0:
				print iteration
			if iteration >= self.burn_In:
				self.samples[iteration-self.burn_In,:]=self.update()


	def cal_edge(self):
		for NI in self.G1.Nodes():
		    A = samples[:,NI.GetId()]
		    count1 = [A[A==1]][0].shape[0]
		    count_1 = [A[A==-1]][0].shape[0]
		    p1 = count1/len(A)
		    p_1 = count_1/len(A)
		    self.G1.AddSAttrDatN(NI.GetId(), 'node_prob1', p1)
		    self.G1.AddSAttrDatN(NI.GetId(), 'node_prob_1', p_1)

	def cal_node(self):
		'''
		'''
