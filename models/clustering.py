import numpy as np
class kmeans:
	def __init__(self, k=2):
		self.k = k
		self.assignedClusters = None

	def fit(self, X):
		mean_ = {}
		for k_ in range(1,self.k+1):
			mean_['Cluster_'+str(k_)] = X[k_-1]
 
		self.assignedClusters = np.zeros(X.shape[0])

		for indx in range(X.shape[0]):
			temp_distances = []
			for key_ in mean_.keys():
				d = np.linearalg.norm(X[indx] - mean_[key_])
				temp_distances.append(d)
			cluster = np.argmin(np.array(temp_distances)) + 1
			self.assignedClusters[indx] = cluster

		## UPDATING THE MEAN
		for k_ in range(1,self.k+1):
			mean_['Cluster_'+str(k_)] = np.mean(X[np.where(self.assignedClusters == k_)[0]])
		
					
