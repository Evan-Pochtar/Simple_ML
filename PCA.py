import numpy as np

class PCA():
    def __init__(self, percent=0.95, num_dim=None):
        self.num_dim = num_dim
        self.percent = percent
        self.mean = None
        self.W = None

    def fit(self,X):
        self.mean = X.mean(0).reshape(1,-1)
        X = centerData(X, self.mean)
        
        eig_val, eig_vec = computeE(X)

        if self.num_dim is None:
            self.num_dim = computeDim(eig_val, self.percent)

        self.W = eig_vec[:,:self.num_dim]
        X_pca = project(X, self.W)

        return X_pca, self.num_dim

    def predict(self, X):
        X = centerData(X, self.mean)
        X_pca = project(X, self.W)
        return X_pca

    def params(self):
        return self.W, self.mean, self.num_dim

def centerData(X, mean):
    centered_X = np.zeros_like(X)
    centered_X = X- mean
    return centered_X

def computeE(centered_X):
    eig_val = 0
    eig_vec = np.zeros([centered_X.shape[1]])
    cov = np.cov(centered_X, rowvar=False)
    eig_val, eig_vec = np.linalg.eigh(cov)
    sort = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sort]
    eig_vec = eig_vec[:, sort]

    return eig_val, eig_vec

def computeDim(eig_val, percent):
    num_dim = 0
    total_var = np.sum(eig_val)
    var = 0
    for val in eig_val:
        num_dim += 1
        var += val
        if var / total_var >= percent:
            break
    
    return num_dim

def project(X, w):
    X_pca = np.zeros([X.shape[0], w.shape[1]])
    X_pca = np.dot(X, w)
    return X_pca
