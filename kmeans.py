import numpy as np

class Kmeans:
    def __init__(self,k): 
        self.num_cluster = k
        self.center = None
        self.cluster_label = np.zeros([k])
        self.error_history = []

    def fit(self, X, y):
        dataIndex = [1, 200, 500, 1000, 1001, 1500, 2000, 2005][:self.num_cluster]
        self.center = initCenters(X, dataIndex)

        num_iter = 0

        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False

        while not is_converged:
            new_center = dict()
            new_center['center'] = np.zeros(self.center.shape)
            new_center['num_sample'] = np.zeros(self.num_cluster)

            for i in range(len(X)):
                distances = computeDis(X[i], self.center)
                cur_cluster = assignCen(distances)
                cluster_assignment[i] = cur_cluster
                new_center['center'][cur_cluster] += X[i]
                new_center['num_sample'][cur_cluster] += 1

            self.center = updateCen(new_center)

            cur_error = computeError(X, cluster_assignment, self.center)
            self.error_history.append(cur_error)

            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1

        contingency_matrix = np.zeros([self.num_cluster,3])
        label2idx = {0:0,8:1,9:2}
        idx2label = {0:0,1:8,2:9}
        for i in range(len(cluster_assignment)):
            contingency_matrix[cluster_assignment[i],label2idx[y[i]]] += 1
        cluster_label = np.argmax(contingency_matrix,-1)
        for i in range(self.num_cluster):
            self.cluster_label[i] = idx2label[cluster_label[i]]

        return num_iter, self.error_history

    def predict(self,X):
        prediction = np.ones([len(X),])

        for i in range(len(X)):
            distances = [np.linalg.norm(X[i] - center) for center in self.center]
            nearest = assignCen(distances)
            prediction[i] = self.cluster_label[nearest]

        return prediction

    def params(self):
        return self.center

def initCenters(X, dataIndex):
    return X[dataIndex]

def computeDis(x, centers):
    dis = np.zeros(len(centers))
    dis = np.linalg.norm(x-centers, axis=1)
    return dis

def assignCen(distances):
    assignment = -1
    assignment = np.argmin(distances)
    return assignment

def updateCen(new_center):
    centers = np.zeros_like(new_center['center'])
    centers = new_center['center'] / new_center['num_sample'][:, None]
    return centers

def computeError(X, assign, centers):
    error = 0
    error = np.sum([np.linalg.norm(X[i]-centers[assign[i]])**2 for i in range(X.shape[0])])
    return error
