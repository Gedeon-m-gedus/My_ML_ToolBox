import numpy as np
class kmeans:
	def __init__(self, k=2):
		self.k = k
		self.assignedClusters = None

	def fit(self, X):
		self.assignedClusters = np.zeros(X.shape[0])
		for indx in range(X.shape[0]):
			mean_ = np.linearalg.norm()
