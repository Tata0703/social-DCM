from __future__ import division
from sklearn.linear_model import LogisticRegression
import numpy as np

class Logistics_Prediction:
	def __init__(self, X, y,index_change):
		self.X = X
		self.y = y
		self.index_change = index_change

	def run_logistics(self):
		mask = np.ones(self.y.shape[0], dtype=bool)
		mask[self.index_change] = False
		self.result_y = self.y[mask,...]
		self.result_X = self.X[mask,...]
		XX = []
		YY = []
		for i in range(len(self.result_y)):
			if self.result_y[i]!=0:
				YY.append(self.result_y[i])
				XX.append(self.result_X[i])
		self.result_X = np.array(XX)
		self.result_y = np.array(YY)
		self.logistic = LogisticRegression()
		self.logistic.fit(self.result_X,self.result_y)

	def run_prediction(self):
		self.run_logistics()
		self.prediction = self.logistic.predict(self.X)
		count = 0
		NN = self.prediction.shape[0]
		for i in range(len(self.prediction)):
			if i in self.index_change:
			    if self.prediction[i] == self.y[i]:
			        count += 1
		print 'prediction accuracy',count/len(self.index_change)
		self.prediction_accuracy = count/len(self.index_change)