import numpy as np
from numpy import linalg

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


class Perceptron(object):

    def __init__(self, T=1):
        self.T = T

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for t in range(self.T):
            for i in range(n_samples):
                if self.predict(X[i])[0] != y[i]:
                    self.w += y[i] * X[i]
                    self.b += y[i]

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.sign(self.project(X))

class KernelPerceptron(object):

    def __init__(self, kernel=linear_kernel, T=1):
        self.kernel = kernel
        self.T = T

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        for t in range(self.T):
            for i in range(n_samples):
                if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0

        # Support vectors
        sv = self.alpha > 1e-5
        ind = np.arange(len(self.alpha))[sv]
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print "%d support vectors out of %d points" % (len(self.alpha),
                                                       n_samples)

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        return np.sign(self.project(X))










"""
A positive definite kernel is simetric
    K(x, x') = K(x', x)

Simple pd kernel for real number
    K(x, x') = xx'

pd kernel for vectors (linear kernel)
    K(x, x') = <xx'>  #inner product

"""


import numpy as np
from scipy.sparse import issparse

###### CONVERTING SPARSE ARRAY TO DENSE ARRAY
def sparse2dense(X):
    return np.array(X.todense()) if issparse(X) else X


###### LINEAR KERNEL
"""
Linear kernel:
    K(x, x') = <xx'>  #inner product
"""
def linearKernel(X, X_):
    return sparse2dense(X.dot(X_.T))



###### POLYNOMIAL KERNEL
"""
Degree d polynomial kernel is defined as:
    K(x, x') = (<xx'> + c)**d
    When c = 0, the kernel is called homogeneous
"""
def polynomialKernel(X, X_, degree=2, c=1):
    return (c + linearKernel(X, X_))**degree



##### RBF KERNEL
"""
Degree d polynomial kernel is defined as:
    K(x, x') = exp(-gamma||X - X'||**2
gamma =  1/(n_features * sigma**2)
"""
def rbfKernel(X, X_, sigma = 10):
n_features = X.shape[1]
gamma = 1 / (n_features * sigma**2)
return np.exp(-gamma * np.linalg.norm(X - X_))


###### THE KERNEL BASE
class kernel:
    our_kernels = {
        'linear':linearKernel,
        'poly': polynomialKernel
        'rbf': rbfKernel
    }
    def __init__(self, kName='poly', **kwargs):
        self.k_name = kName
        self.k_params = self.extractParams(**kwargs)
        self.kernel2use = self.our_kernels[kName]

    def extractParams(self, **kwargs):
        parm = {}
        if self.k_name == 'poly':
            parm['degree'] = kwargs.get('degree', 2)
            parm['c'] = kwargs.get('c', 1)

        if self.k_name == 'rbf':
            parm['sigma'] = kwargs.get('sigma')
        return parm
   
    def fit(self,X,y,**kwargs):
        K_X = self.kernel2use(X,X,self.k_params)
        return self.fit_K(K_X, y, **kwargs)

    def fit_K(self, K, y, **kwargs):

        pass

### Kernel Ridge Regression
"""
alpha = (K + lambda*I)**-1 * y
REF: https://people.eecs.berkeley.edu/~wainwrig/stat241b/lec6.pdf
"""
class kernelRidgeRegression(kernel):
    def __init__(self, lambda, **kwargs):
        self.lambda = lambda
        super().__init__(**kwargs)
   
    def fit_K(self, K,y):
        """
        alpha = (K + lambda*I)**-1 * y        
        """
        self.alpha = np.linalg.solve((K + self.lambda * (len(y)) * np.eye(len(y)) ) , y)
        print('ALPHA VALUE:-----> ', self.alpha)
        return self

    def predict(self, X):
        pass
