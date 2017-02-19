import random
# from snapvx import *
import numpy as np
import numpy.linalg as LA
import scipy as spy
import time
from itertools import *
import sys
import cvxpy as cvx
from random import randint
import numpy as np
import random
from scipy.sparse import csc_matrix
from scipy import sparse as sp
import networkx as nx

class ADMM_fully_distributed:
	def __init__(self, X, y, b,pos_node ,temp, Lambda, Rho):
		'''
		temp: graphical model structure, SNAP graph class
		'''
		# known values
		self.X = X
		self.y = y
		self.dim = X.shape[1]
		self.Lambda = Lambda
		self.Rho = Rho
		self.temp = temp
		self.num_nodes = nx.number_of_nodes(self.temp)
		self.Z = csc_matrix((self.num_nodes, self.num_nodes), dtype=np.float).toarray()
		self.U = csc_matrix((self.num_nodes, self.num_nodes), dtype=np.float).toarray()
		for EI in self.temp.edges_iter():
			self.Z[EI[0],EI[1]] = np.random.rand()
			self.U[EI[0],EI[1]] = np.random.rand()
		self.W = np.zeros((self.dim))
		self.b = b
		self.pos_node = pos_node
		self.g = np.random.random((self.num_nodes,self.dim))
		self.h = np.random.random((self.num_nodes,self.dim))

	def update_W(self):
		'''
		dim : degree of feature vector
		N: number of nodes
		X: feature matrix, N*dim
		y: output choice, N*1

		return w: numpy matrix(dim,1)
		'''
		dim = self.X.shape[1]
		w = cvx.Variable(dim)
		#loss = cvx.sum_entries(cvx.mul_elemwise(np.array(self.pos_node),cvx.logistic(-cvx.mul_elemwise(self.y, self.X*w+self.b))))
		loss = 0
		for i in range(self.num_nodes):
			loss +=  cvx.norm(w - self.g[i,:] + self.h[i,:])**2*self.Rho/2
		problem = cvx.Problem(cvx.Minimize(loss))
		problem.solve(verbose=False)
		opt = problem.value
		#     print w.value
		# return w.value
		self.W = np.array(w.value).flatten()

	def update_b(self):
		'''
		temp: graph data structure
		Z: scipy sparse matrix N*N, same sparsity pattern as the adjacency matrix
		U: scipy sparse matrix N*N, same sparsity pattern as the adjacency matrix
		X: feature matrix, N*dim
		y: output choice, N*1

		return b:ndarray(N,)
		'''
		B = []
		# for i in self.temp.nodes_iter():
		for i in range(self.num_nodes):
			bi = cvx.Variable(1)
			# i = NI.GetId()
			loss = cvx.logistic(-cvx.mul_elemwise(self.y[i], self.X[i].dot(self.g[i,:])+bi))*self.temp.node[i]['pos_node_prob']
			# for Id in NI.GetOutEdges():
			# print self.X[i].shape
			# print self.W.shape
			# print self.X[i].dot(self.W)
			# print loss.size
			for Id in self.temp.neighbors(i):
				loss = loss+(bi-self.Z[i,Id]+ self.U[i,Id])**2*self.Rho/2
				#*self.temp[i][Id]['pos_edge_prob']
			problem = cvx.Problem(cvx.Minimize(loss))
			problem.solve(verbose=False)
			# opt = problem.value
			# B.append(bi.value)
			self.b[i] = bi.value
		# self.b = np.array(B).reshape((100,1))

	def update_g(self):
		for i in range(self.num_nodes):
			gt = cvx.Variable(self.dim)
			loss = cvx.logistic(-cvx.mul_elemwise(self.y[i], self.X[i]*gt+self.b[i]))*self.temp.node[i]['pos_node_prob']
			loss += cvx.norm(self.W - gt + self.h[i,:])**2*self.Rho/2
			problem = cvx.Problem(cvx.Minimize(loss))
			problem.solve(verbose=False)
			self.g[i,:] = gt.value.ravel()





	def update_Z(self):
		for k in self.temp.nodes_iter():
			for j in self.temp.neighbors(k):
				A = self.b[j] + self.U[j,k]
				B = self.b[k] + self.U[k,j]
				# print(A)
				# # print(B)
				# print (2*self.Lambda*self.temp[j][k]['pos_edge_prob']*A + (2*self.Lambda*self.temp[j][k]['pos_edge_prob']+self.Rho)*B)/(self.Lambda*4*self.temp[j][k]['pos_edge_prob']+self.Rho)
				# update z_kj
				# print (2*self.Lambda*A + (2*self.Lambda+self.Rho)*B)/(self.Lambda*4+self.Rho)
				self.Z[k,j] = (2*self.Lambda*self.temp[j][k]['pos_edge_prob']*A + (2*self.Lambda*self.temp[j][k]['pos_edge_prob']+self.Rho)*B)/(self.Lambda*4*self.temp[j][k]['pos_edge_prob']+self.Rho)
				# self.Z[j,k] = (2*self.Lambda*self.temp[k][j]['pos_edge_prob']*B + (2*self.Lambda*self.temp[k][j]['pos_edge_prob']+self.Rho)*A)/(self.Lambda*4*self.temp[k][j]['pos_edge_prob']+self.Rho)
		# print(self.Z)
				# A = self.b[i] + self.U[i,Id]
				# B = self.b[i] + self.U[Id,i]
				# theta = max(1 - self.Lambda/(self.Rho*LA.norm(A - B)), 0.5)
				# self.Z[i,Id] = theta*A + (1-theta)*B
				# self.Z[Id,i] = theta*B + (1-theta)*A

	def update_U(self):
		for i in self.temp.nodes_iter():
			for Id in self.temp.neighbors(i):
				self.U[i,Id] = self.U[i,Id] + self.b[i] - self.Z[i,Id]
				# self.U[Id,i] = self.U[Id,i] + self.b[Id] - self.Z[Id,i]

	def update_h(self):
		for i in range(self.num_nodes):
			self.h[i,:] = self.h[i,:] + (self.W -self.g[i,:])


	
	def runADMM_Grid(self,iterations):
		for i in range(iterations):
			W_old = self.W
			b_old = self.b
			self.update_W()
			self.update_b()
			'''
			to do: check ECOS solver, why combined backtracking failed?
			'''
			self.update_Z()
			self.update_g()
			self.update_h()
			self.update_U()
			if i%1 == 0:
				print 'iteration = ',i, 'objective = ', self.cal_LL()
				print LA.norm(self.W-W_old)+ LA.norm(self.b-b_old)
	def cal_LL(self):
		W = np.array(self.W).flatten()
		b = np.array(self.b).flatten()
		loss = np.sum(np.multiply(np.array(self.pos_node),np.log( (1+np.exp(-np.multiply(self.y,np.dot(self.X,W)+b))))))
		for EI in self.temp.edges_iter():
			loss +=  self.Lambda*(self.b[EI[0]]-self.b[EI[1]])**2*self.temp[EI[0]][EI[1]]['pos_edge_prob']
		return loss

