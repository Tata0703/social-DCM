import random
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

class CVX_weighted:
	def __init__(self, X, y, b,pos_node ,temp, Lambda, Rho):
		self.X = X
		self.y = y
		self.value = 0
		self.dim = X.shape[1]
		self.Lambda = Lambda
		self.Rho = Rho
		self.temp = temp
		self.num_nodes = nx.number_of_nodes(self.temp)
		self.W = np.zeros((self.dim))
		self.b = b
		self.pos_node = pos_node
		self.P = np.zeros((self.num_nodes,self.num_nodes))
	
	def init_P(self):
		for i in self.temp.nodes_iter():
		    for j in self.temp.neighbors(i):
		        self.P[i,j] = self.temp[i][j]['pos_edge_prob'] 

		self.P = np.diag(np.sum(self.P,1)) - self.P 

	
	def solve(self):
		dim = self.X.shape[1]
		w = cvx.Variable(dim)
		num_nodes = nx.number_of_nodes(self.temp)
		b = cvx.Variable(num_nodes)
		loss = cvx.sum_entries(cvx.mul_elemwise(np.array(self.pos_node),cvx.logistic(-cvx.mul_elemwise(self.y, self.X*w+b)))) + self.Lambda*cvx.quad_form(b,self.P)
		problem = cvx.Problem(cvx.Minimize(loss))
		problem.solve(verbose=False)
		opt = problem.value
		self.W = w.value
		self.b = b.value
		self.value = opt