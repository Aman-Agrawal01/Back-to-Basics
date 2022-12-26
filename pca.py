import numpy as np
from scipy import stats

class PCA():
    
    def __init__(self, n_components = 1):
        self.n_components = n_components
    
    def fit(self,X):
        
        standardized_X = stats.zscore(X)
        cov = np.cov(standardized_X.T)
        
        eigenvalues , eigenvectors = np.linalg.eig(cov)
        index = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:,index]
        
        variance = eigenvalues[:self.n_components]
        self.pc = eigenvectors[:,:self.n_components]
        self.variance_proportion = np.sum(self.variance)/np.sum(self.eigenvalues)
    
    def transform(self, X):
        return np.matmul(X,self.pc)