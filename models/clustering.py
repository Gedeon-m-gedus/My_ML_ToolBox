import numpy as np

class kmeans:
	def __init__(self, k=2, epslon=0.001):
		self.k = k
		self.epslon = epslon
		self.assignedClusters = None

	def fit(self, X):
		mean_ = {}
		for k_ in range(1,self.k+1):
			mean_['Cluster_'+str(k_)] = X[k_-1]
 
		self.assignedClusters = np.zeros(X.shape[0])

		while True :
			for indx in range(X.shape[0]):
				temp_distances = []
				for key_ in mean_.keys():
					d = np.linalg.norm(X[indx] - mean_[key_])
					temp_distances.append(d)
				cluster = np.argmin(np.array(temp_distances)) + 1
				self.assignedClusters[indx] = cluster
			mean_check = mean_.copy()

			## UPDATING THE MEAN
			for k_ in range(1,self.k+1):
				mean_['Cluster_'+str(k_)] = np.mean(X[np.where(self.assignedClusters == k_)[0]],axis=0)
			diff = []
			for ki in mean_.keys():
				diff.append(np.linalg.norm(mean_check[ki] - mean_[ki]))
			if sum(diff) <= self.epslon:
			 	break
			
	def accuracy(self, true_labels):
		print('True clustes:\n',true_labels)
		if min(true_labels)==0:
			self.assignedClusters -= 1
		print('Predicted Clusters\n',self.assignedClusters)
		acc = 0
		for indx, value in enumerate(self.assignedClusters):
			if int(value) == int(true_labels[indx]):
				acc += 1
		return acc/len(true_labels)



