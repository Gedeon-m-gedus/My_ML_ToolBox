class perceptron:
    def __init__(self, learning_rate=0.1, n_iter=500):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.parameters = None
        
    def fit(self, X, y):
        N = X.shape[0]
        d = X.shape[1]
        y = np.where(y==0,-1,1)
        print('Number training of samples: ', N, '\nNumber of features: ',d)
        self.parameters = np.zeros(d)
        print(self.parameters)
        for count in range(self.n_iter):
            for sample_index in range(N):
                X_i = X[sample_index]
                pred = np.sign(X_i.dot(self.parameters))
                if pred == y[sample_index]:
                    pass
                else:
                    self.parameters = self.parameters - self.lr * X_i
   
        print(self.parameters)
    
    def predict(self, data):
        preds = np.sign(data.dot(self.parameters))
        preds = np.where(preds==-1,0,1)
        return preds
        print('This function will predict on the given data') 
    
    def accuracy(self, lebel, pred):
        acc = 0
        for i, _ in enumerate(lebel):
            if pred[i] == lebel[i]:acc+=1
        return acc/len(pred)
