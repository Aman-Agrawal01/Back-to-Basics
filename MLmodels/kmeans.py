import numpy as np

class KMeans():

    def __init__(self, k):
        self.n_clusters = k

    def fit(self, X, max_iter = 100):

        index = np.random.randint(X.shape[0], size = self.n_clusters)
        self.centroids = X[index,:]
        self.cluster = np.zeros(shape=(X.shape[0],1))
        
        for i in range(max_iter):
            for j in range(X.shape[0]):
                dist = list()
                for l in range(self.n_clusters):
                    dist.append(np.linalg.norm(self.centroids[l,:] - X[j,:]))   
                self.cluster[j] = np.argmin(np.array(dist))
            for j in range(self.n_clusters):
                index = np.where(self.cluster==j)[0]
                self.centroids[j,:] = np.mean(X[index,:],axis=0)