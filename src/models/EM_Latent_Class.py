from __future__ import division
import random
import numpy as np
import numpy.linalg as alg
import scipy as spy
import networkx as nx

import time
from itertools import *
import sys
import numpy.linalg as LA
import cvxpy as cvx
from random import randint
import numpy as np
import random
from scipy.sparse import csc_matrix
from scipy import sparse as sp
import networkx as nx
from scipy.special import expit

class EM_Latent_Class:
	def __init__(self, X, Y,num_classes,X_test,Y_test):
		self.X = X
		self.Y = Y
		self.num_classes = num_classes
		self.dim = X.shape[1]
		self.num_nodes = X.shape[0]
		self.W = np.random.random((self.dim, self.num_classes))
		self.B = np.random.random(self.num_classes)
		self.posterior_mat = np.random.random((self.num_nodes,self.num_classes))
		self.p = [0.5,0.5]
		self.expected_LL = 0
		self.converged = False
		self.X_test = X_test
		self.Y_test = Y_test
		self.predict_acc = 0
	
	def E_step(self):
		self.prob_mat = np.zeros((self.num_nodes,self.num_classes))
		for k in range(self.num_classes):
			self.prob_mat[:,k] = expit(np.multiply(self.Y.flatten(),np.dot(self.W[:,k],self.X.T)+self.B[k]))
		for k in range(self.num_classes):
			self.posterior_mat[:,k] = np.multiply(self.prob_mat,self.p)[:,k]/np.sum(np.multiply(self.prob_mat,self.p),axis=1)
		self.p[1] = np.sum(self.posterior_mat[:,1])/self.posterior_mat.shape[0]
		self.p[0] = np.sum(self.posterior_mat[:,0])/self.posterior_mat.shape[0]

	def M_step(self):
		expected_LL = 0
		for k in range(self.num_classes):
			w = cvx.Variable(self.dim)
			b = cvx.Variable(1)
			loss = cvx.sum_entries(cvx.mul_elemwise(np.array(self.posterior_mat[:,k]),cvx.logistic(-cvx.mul_elemwise(self.Y, self.X*w+np.ones(self.num_nodes)*b))))
			problem = cvx.Problem(cvx.Minimize(loss))
			problem.solve(verbose=False,solver = 'SCS')
			expected_LL -= problem.value
			self.W[:,k] = np.array(w.value).flatten()
			self.B[k] = b.value
		self.expected_LL = expected_LL
	def EM(self):
		iteration = 1
		while(self.converged==False):
			print 'iteration: ',iteration
			iteration += 1
			expected_LL_old = self.expected_LL
			self.E_step()
			self.M_step()
			if LA.norm(expected_LL_old-self.expected_LL)< 1:
				self.converged = True
	def predict(self):
		num_nodes = self.X_test.shape[0]
		self.predict_prob_mat = np.zeros((num_nodes,2))
		for k in range(self.num_classes):
			self.predict_prob_mat[:,0] += expit(np.dot(self.W[:,k],self.X_test.T)+self.B[k])*self.p[k]
		for k in range(self.num_classes):
			self.predict_prob_mat[:,1] += (1-expit(np.dot(self.W[:,k],self.X_test.T)+self.B[k]))*self.p[k]
		assignment = self.predict_prob_mat.argmax(axis=1).astype(int)
		self.predictions = []
		for i in range(len(self.Y_test)):
			if assignment[i] ==0 :
				self.predictions.append(1)
			else:
				self.predictions.append(-1)		
		count = 0
		count0 = 0
		for i in range(len(self.Y_test)):
			if self.Y_test[i]==0:
				count0 +=1
			else:
				if self.predictions[i] == self.Y_test[i]:
					count += 1
		self.predict_acc = count/(len(self.Y_test)-count0)

