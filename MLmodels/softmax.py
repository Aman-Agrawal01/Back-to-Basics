import numpy as np

class SoftmaxRegression():

    def __init__(self):
        """     Initializing the weights    """
        self.weights = None

    def add_ones(self,X):
        """     Adding the column vector consisting of ones at the
                last column, concating this to account the bias      """
        return np.hstack((X,np.ones((X.shape[0],1))))
    
    def softmax(self,y):
        """     Softmax function implementation        """
        return np.exp(y)/np.sum(np.exp(y),axis=1).reshape(-1,1)

    def fit(self, X, y, epochs=10000, lr=0.1):
        """     Training of the model 
                and learning the weights      
                Default #of epochs and learning rate is set for 
                the iris dataset      """
        n_class = len(np.unique(y))
        self.weights = np.zeros(shape=(X.shape[1]+1,n_class))
        y_trans = np.zeros((y.shape[0],3))

        for i,label in enumerate(y):
            y_trans[i,label] = 1

        for epoch in range(epochs):
            
            grad = np.zeros_like(self.weights)
            y_pred = self.predict_prob(X)  
            loss = -np.sum(y_trans*np.log(y_pred))
            print(f"Epoch - {epoch+1}/{epochs}    Loss - {loss}")
            
            for i, label in enumerate(y):
                grad[:-1,label] -= X[i,:].reshape(-1,1)
                grad[-1,label] -= 1
                grad[:,label] *= (1 - y_pred[i,label])
                
            self.weights -= lr*grad/y.shape[0]

    def predict_prob(self,X):
        """     Predict the conditional probability distribution    """
        return self.softmax(np.matmul(self.add_ones(X),self.weights))
    
    def loss_func(self, y_pred, y):
        """     Calculation of multi-class cross entropy function   """
        return -np.sum(np.log(np.array([y_pred[i,label] for i,label in enumerate(y)])))/y.shape[0]
    
    def predict(self, X):
        """     Classifying the data into class     """
        return np.argmax(self.predict_prob(X),axis=1)
    
    def score(self, X,y):
        """     Accuracy of the model       """
        return np.mean(self.predict(X).reshape(-1,1)==y)