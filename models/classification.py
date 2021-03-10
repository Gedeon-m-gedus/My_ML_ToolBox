import numpy as np

class perceptron:
    def __init__(self, learning_rate=0.1, n_iter=500):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.parameters = None
        
    def fit(self, X, y):
	b = np.ones(X.shape[0])
        b = b.reshape(b.shape[0],-1)
        X = X.reshape(X.shape[0],-1)
        X = np.hstack((b,X))
        y = np.where(y==0,-1,1)

        N = X.shape[0]
        d = X.shape[1]
        
        print('Number training of samples: ', N, '\nNumber of features: ',d)

        self.parameters = np.random.rand(d)
        for count in range(self.n_iter):
            for sample_index in range(N):
                X_i = X[sample_index]
                pred = np.sign(self.parameters.T.dot(X_i))
                if pred == y[sample_index]:
                    pass
                else:
                    self.parameters = self.parameters - self.lr * (y[sample_index] - pred) * X_i
        print(self.parameters)

    
    def predict(self, X):
        b = np.ones(X.shape[0])
        b = b.reshape(b.shape[0],-1)
        X = X.reshape(X.shape[0],-1)
        X = np.hstack((b,X))
        pred = np.sign(X.dot(self.parameters))
        return np.where(pred==1,1,0)

    
    def accuracy(self, lebel, pred):
        acc = 0
        for i, _ in enumerate(lebel):
            if pred[i] == lebel[i]:acc+=1
        return acc/len(pred)


class logisticRegression:
	pass
