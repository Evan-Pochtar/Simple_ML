import numpy as np

class GaussianDiscriminant_C1:
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))
        self.S = np.zeros((k,d,d))
        self.p = np.zeros(2)

    def fit(self, Xtrain, ytrain):
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)
        
        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)

        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)

        self.p = computePrior(ytrain)

    def predict(self, Xtest):
        predictions = np.zeros(Xtest.shape[0])
        for i, x in enumerate(Xtest):
            g1 = -0.5 * np.dot(np.dot((x - self.m[0]).T, np.linalg.inv(self.S[0])), (x - self.m[0])) - 0.5 * np.log(np.linalg.det(self.S[0])) + np.log(self.p[0])
            g2 = -0.5 * np.dot(np.dot((x - self.m[1]).T, np.linalg.inv(self.S[1])), (x - self.m[1])) - 0.5 * np.log(np.linalg.det(self.S[1])) + np.log(self.p[1])
            if g1 > g2:
                predictions[i] = 1
            else:
                predictions[i] = 2

        return predictions


class GaussianDiscriminant_C2:
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))
        self.S = np.zeros((k,d,d))
        self.shared_S =np.zeros((d,d))
        self.p = np.zeros(2)

    def fit(self, Xtrain, ytrain):
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)
        
        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)
        
        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)
        
        self.p = computePrior(ytrain)

        p1, p2 = self.p
        self.shared_S = p1 * self.S[0,:,:] + p2 * self.S[1,:,:]

    def predict(self, Xtest):
        predictions = np.zeros(Xtest.shape[0])
        
        g1 = np.dot(Xtest, np.dot(np.linalg.inv(self.shared_S), self.m[0,:])) - 0.5 * np.dot(self.m[0,:].T, np.dot(np.linalg.inv(self.shared_S), self.m[0,:])) + np.log(self.p[0])
        g2 = np.dot(Xtest, np.dot(np.linalg.inv(self.shared_S), self.m[1,:])) - 0.5 * np.dot(self.m[1,:].T, np.dot(np.linalg.inv(self.shared_S), self.m[1,:])) + np.log(self.p[1])

        predictions[g1 > g2] = 1
        predictions[g1 <= g2] = 2
        
        return predictions


class GaussianDiscriminant_C3:
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))
        self.S = np.zeros((k,d,d))
        self.shared_S =np.zeros((d,d))
        self.p = np.zeros(2)

    def fit(self, Xtrain, ytrain):
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)

        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)

        self.p = computePrior(ytrain)

        self.S[0,:,:] = np.diag(np.diag(self.S[0,:,:]))
        self.S[1,:,:] = np.diag(np.diag(self.S[1,:,:]))
        
        self.shared_S = self.p[0] * self.S[0,:,:] + self.p[1] * self.S[1,:,:]

    def predict(self, Xtest):
        predictions = np.zeros(Xtest.shape[0])
        
        g1 = np.sum(Xtest @ np.diag(1 / np.diag(self.shared_S)) * self.m[0,:], axis=1) - 0.5 * self.m[0,:] @ np.diag(1 / np.diag(self.shared_S)) @ self.m[0,:] + np.log(self.p[0])
        g2 = np.sum(Xtest @ np.diag(1 / np.diag(self.shared_S)) * self.m[1,:], axis=1) - 0.5 * self.m[1,:] @ np.diag(1 / np.diag(self.shared_S)) @ self.m[1,:] + np.log(self.p[1])
        
        predictions[g1 > g2] = 1
        predictions[g1 <= g2] = 2
        
        return predictions

def splitData(features, labels):
    features1 = np.zeros([np.sum(labels == 1),features.shape[1]])  
    features2 = np.zeros([np.sum(labels == 2),features.shape[1]])
    
    features1 = features[np.where(labels == 1)[0]]
    features2 = features[np.where(labels == 2)[0]]

    return features1, features2

def computeMean(features):
    m = np.zeros(features.shape[1])
    m = np.mean(features, axis=0)
    return m

def computeCov(features):
    S = np.eye(features.shape[1])
    S = np.cov(features, rowvar=False)
    return S

def computePrior(labels):
    p = np.array([0.5,0.5])
    unique, counts = np.unique(labels, return_counts=True)

    p = counts / labels.shape[0]

    return p
