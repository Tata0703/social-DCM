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

class ADMM:
	def __init__(self, X, y, temp, Lambda, Rho):
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
		self.num_nodes = temp.GetNodes()
		self.Z = csc_matrix((self.num_nodes, self.num_nodes), dtype=np.float).toarray()
		self.U = csc_matrix((self.num_nodes, self.num_nodes), dtype=np.float).toarray()
		for EI in temp.Edges(): 
		    self.Z[EI.GetSrcNId(),EI.GetDstNId()] = np.random.rand()
		    self.U[EI.GetSrcNId(),EI.GetDstNId()] = np.random.rand()
		self.W = np.zeros((self.dim,1))
		self.b = np.zeros((self.num_nodes,1))

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
		loss = cvx.sum_entries(cvx.logistic(-cvx.mul_elemwise(self.y, self.X*w+self.b))) + self.Lambda/2 * cvx.sum_squares(w)
		problem = cvx.Problem(cvx.Minimize(loss))
		problem.solve(verbose=False)
		opt = problem.value
		#     print w.value
		# return w.value
		self.W = w.value

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
		for NI in self.temp.Nodes():
		    bi = cvx.Variable(1)
		    i = NI.GetId()
		    loss = cvx.logistic(-cvx.mul_elemwise(self.y[i], self.X[i]*self.W+bi))   
		    for Id in NI.GetOutEdges():
		        loss = loss+(bi-self.Z[NI.GetId(),Id]-self.U[NI.GetId(),Id])**2*self.Rho/2
		    problem = cvx.Problem(cvx.Minimize(loss))
		    problem.solve(verbose=False)
		    opt = problem.value
		    B.append(bi.value)
		self.b = np.array(B).reshape((100,1))

	def update_U(self):
	    for NI in self.temp.Nodes():
	        for Id in NI.GetOutEdges():
	            self.U[NI.GetId(),Id] = self.U[NI.GetId(),Id] + self.b[NI.GetId()] - self.Z[NI.GetId(),Id]

	def update_Z(self):
		for NI in self.temp.Nodes():
		    for Id in NI.GetOutEdges():
		        A = self.b[NI.GetId(),0] + self.U[NI.GetId(),Id]
		        B = self.b[Id,0] + self.U[Id,NI.GetId()]
		        theta = max(1 - self.Lambda/(self.Rho*LA.norm(A - B)), 0.5)
		        self.Z[NI.GetId(),Id] = theta*A + (1-theta)*B
		        self.Z[Id,NI.GetId()] = theta*B + (1-theta)*A

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
	        self.update_U()
	        if i%50 == 0:
	        	print 'iteration = ',i
	        	print LA.norm(self.W-W_old)+ LA.norm(self.b-b_old)

