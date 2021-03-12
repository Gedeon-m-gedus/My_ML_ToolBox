import numpy as np
import matplotlib.pyplot as plt
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
			# print(mean_)
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
			# print(mean_check, mean_)
			diff = []
			for ki in mean_.keys():
				diff.append(np.linalg.norm(mean_check[ki] - mean_[ki]))
			if sum(diff) <= self.epslon:
			 	break
			print('mean updated',diff)

data = np.array([[1,1],
				[1,2],
				[0,0],
				[2,2],
				[1,3],
				[2,1],
				[4,3],
				[4,4],
				[4,5],
				[5,3],
				[5,4],
				[5,5],
				[6,4],
				[-1,-1],
				[6,5],
				[5,-2],
				[3,-2],
				[4,-2],
				[3,-1],
				[4,-0],
				[5,-1],
				[0,1],
				[5,1],
				[0,2],
				[3,3]]
				)
from sklearn.datasets import make_blobs
features, true_labels = make_blobs(n_samples=2000, centers=3, cluster_std=1.75, random_state=42)
data = features
model = kmeans(k=3)
model.fit(data)	
clusters = np.array(model.assignedClusters)

for i in range(1, model.k+1):
	plt.scatter(data[np.where(clusters==i)[0]][:,0],data[np.where(clusters==i)[0]][:,1])
plt.show()