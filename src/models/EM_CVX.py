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
import numpy.linalg as LA

from src.models.GibbsSampler_networkx import GibbsSampler
from src.models.ADMM import ADMM
from src.models.ADMM_networkx_weighted import ADMM_networkx_weighted
from src.models.CVX_weighted import CVX_weighted


class EM_CVX:
	def __init__(self, X, y, temp, Lambda, Rho, num_classes,iterations,burn_In):
		self.X = X
		self.y = y
		self.num_classes = num_classes
		self.value = 0
		self.dim = X.shape[1]
		self.Lambda = Lambda
		self.Rho = Rho
		self.temp = temp
		self.num_nodes = nx.number_of_nodes(self.temp)
		self.num_edges = self.temp.number_of_edges()
		self.W = np.random.random((self.dim, self.num_classes))
		self.b = np.random.random((self.num_nodes,self.num_classes))
		self.pos_node = []
		self.iterations = iterations
		# self.prob_mat = np.zeros((self.num_nodes,self.num_classes))
		self.posterior_mat = np.zeros((self.num_nodes,self.num_classes))
		self.burn_In = burn_In
		self.z_0 = np.random.randint(low = 0,high = self.num_classes-1,size = self.num_nodes)
		self.G = GibbsSampler(self.temp,self.b,0,self.burn_In,self.Lambda,self.z_0)
		self.LL = []

	def E_step(self,i):
		num_Samples = (i+1)*1000
		self.G = GibbsSampler(self.temp,self.b,num_Samples,self.burn_In,self.Lambda,self.z_0)
		self.G.sampling()
		self.G.cal_node()
		self.G.cal_edge()
		self.G.get_node()
		self.G.get_edge()
		self.temp = self.G.temp
		self.z_0 = self.G.samples[-1,:]
		prior_node = self.G.node_prob
		self.prob_mat = np.zeros((self.num_nodes,self.num_classes))
		posterior_mat = np.zeros((self.num_nodes,self.num_classes))
		for k in range(self.num_classes):
			self.prob_mat[:,k] = 1 / (1+np.exp(np.multiply(self.y.flatten(),-np.dot(self.W[:,k],self.X.T)-self.b[:,k])))
		for k in range(self.num_classes):
			posterior_mat[:,k] = np.multiply(self.prob_mat,prior_node)[:,k]/np.sum(np.multiply(self.prob_mat,prior_node),axis=1)
		num_edges = self.temp.number_of_edges()
		prior_edge = self.G.edge_prob
		prob_mat_edge = np.zeros((self.num_edges,self.num_classes,self.num_classes))
		posterior_mat_edge = np.zeros((self.num_edges,self.num_classes))
		count = 0
		for EI in self.temp.edges_iter():
			for k1 in range(self.num_classes):
				for k2 in range(self.num_classes):
					prob_mat_edge[count,k1,k2] = self.prob_mat[EI[0],k1]*self.prob_mat[EI[1],k2]
			count += 1
		prob_mat_edge = prob_mat_edge.reshape((self.num_edges,self.num_classes**2))
		posterior_mat_edge[:,0] = np.multiply(prob_mat_edge,prior_edge)[:,0]/np.sum(np.multiply(prob_mat_edge,prior_edge),axis=1)
		posterior_mat_edge[:,1] = np.multiply(prob_mat_edge,prior_edge)[:,3]/np.sum(np.multiply(prob_mat_edge,prior_edge),axis=1)
		self.posterior_mat = posterior_mat
		self.posterior_mat_edge = posterior_mat_edge

	def M_step(self):
		self.expected_LL = 0
		for k in range(self.num_classes):
			pos_node = []
			for NI in self.temp.nodes_iter():
				self.temp.node[NI]['pos_node_prob'] = self.posterior_mat[NI,k]
				pos_node.append(self.posterior_mat[NI,k])
			count = 0
			for EI in self.temp.edges_iter():
				self.temp[EI[0]][EI[1]]['pos_edge_prob'] = self.posterior_mat_edge[count,k]
				count += 1
			A = CVX_weighted(self.X, self.y, self.b,pos_node,self.temp,self.Lambda, self.Rho)
			A.init_P()
			A.solve()
	#         A = ADMM_networkx_weighted(X, y, b[:,0],np.array(pos_node),temp,Lambda, Rho)
	#         A.runADMM_Grid(iterations)
			self.W[:,k] = A.W.flatten()
			self.b[:,k] = A.b.flatten()
			self.expected_LL -= A.value

	def run_EM(self):
		for i in range(self.iterations):
			W_old = np.copy(self.W)
			b_old = np.copy(self.b)
			self.E_step(i)
			self.M_step()
			print 'expected Log likelihood is',self.expected_LL
			self.LL.append(self.expected_LL)








	# prob_mat_edge = prob_mat_edge.reshape((num_edges,num_classes**2))
	# posterior_mat_edge[:,0] = np.multiply(prob_mat_edge,prior_edge)[:,0]/np.sum(np.multiply(prob_mat_edge,prior_edge),axis=1)
	# posterior_mat_edge[:,1] = np.multiply(prob_mat_edge,prior_edge)[:,3]/np.sum(np.multiply(prob_mat_edge,prior_edge),axis=1)
	# return posterior_mat,posterior_mat_edge
