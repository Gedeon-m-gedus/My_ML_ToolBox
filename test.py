import numpy as np
import matplotlib.pyplot as plt

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
features, true_labels = make_blobs(n_samples=200, centers=3, cluster_std=1.75, random_state=42)
data = features
# for i in true_labels:
# 	print(i)


from models.clustering import kmeans
model = kmeans(k=3)
model.fit(data)	
clusters = np.array(model.assignedClusters)
# for i in clusters:
# 	print(i)
# print(model.accuracy(true_labels))
# print(true_labels, model.assignedClusters)
for i in range(1, model.k+1):
	plt.scatter(data[np.where(clusters==i)[0]][:,0],data[np.where(clusters==i)[0]][:,1])
plt.show()
