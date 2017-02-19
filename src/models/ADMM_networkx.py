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

class ADMM:
	def __init__(self, X, y, temp, Lambda, Rho):
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
		self.W = np.zeros((self.dim,1))
		self.b = np.zeros((self.num_nodes,1))

	def update_W(self):
		dim = self.X.shape[1]
		w = cvx.Variable(dim)
		loss = cvx.sum_entries(cvx.logistic(-cvx.mul_elemwise(self.y, self.X*w+self.b))) 
		problem = cvx.Problem(cvx.Minimize(loss))
		problem.solve(verbose=False)
		opt = problem.value
		#     print w.value
		# return w.value
		self.W = w.value

	def update_b(self):
		B = []
		for i in range(self.num_nodes):
		    bi = cvx.Variable(1)
		    # i = NI.GetId()
		    loss = cvx.logistic(-cvx.mul_elemwise(self.y[i], self.X[i]*self.W+bi))   
		    # for Id in NI.GetOutEdges():
		    for Id in self.temp.neighbors(i):
		        loss = loss+(bi-self.Z[i,Id]+self.U[i,Id])**2*self.Rho/2
		    problem = cvx.Problem(cvx.Minimize(loss))
		    problem.solve(verbose=False)
		    # opt = problem.value
		    # B.append(bi.value)
		    self.b[i,0] = bi.value
		# self.b = np.array(B).reshape((20,1))

	def update_U(self):
	    for i in self.temp.nodes_iter():
	        for Id in self.temp.neighbors(i):
	            self.U[i,Id] = self.U[i,Id] + self.b[i] - self.Z[i,Id]
	            # self.U[Id,i] = self.U[Id,i] + self.b[Id] - self.Z[Id,i]

	def update_Z(self):
		for k in self.temp.nodes_iter():
			for j in self.temp.neighbors(k):
				A = self.b[j] + self.U[j,k]
				B = self.b[k] + self.U[k,j]
				A = A[0]
				B = B[0]
				self.Z[k,j] = (2*self.Lambda*A + (2*self.Lambda+self.Rho)*B)/(self.Lambda*4+self.Rho)
				# self.Z[j,k] = (2*self.Lambda*B + (2*self.Lambda+self.Rho)*A)/(self.Lambda*4+self.Rho)

	def runADMM_Grid(self,iterations):
	    for i in range(iterations):
	    	W = np.array(self.W).flatten()
	    	b = np.array(self.b).flatten()
	    	loss = np.sum(np.log( (1+np.exp(-np.multiply(self.y,np.dot(self.X,W)+b)))))
        	print 'iteration = ',i
        	for EI in self.temp.edges_iter():
        		loss +=  self.Lambda*(self.b[EI[0]] - self.b[EI[1]])**2
	        self.update_W()
	        self.update_b()
	        self.update_Z()
	        self.update_U()

        	print 'loss is',loss
        	# print LA.norm(self.W-W_old)
        	# print LA.norm(self.b-b_old)
        	print '.........................'

