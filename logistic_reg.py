import numpy as np

class logistic_reg():
    
    def predict(self,X):
        z = np.dot(X,self.weights) + self.bias
        return 1/(1+np.exp(-z))
    
    def loss(self, y_train, y_pred):
        return - np.mean(np.multiply(y_train,np.log(y_pred)) + np.multiply((1-y_train),np.log(1-y_pred)))
    
    def prediction(self, X):
        prob = self.predict(X)
        return np.where(prob > 0.5, 1, 0)
    
    def accuracy(self, y_pred, y_train):
        return np.mean(y_pred == y_train)

    def fit(self,X_train, y_train, max_iter = 100, learning_rate = 1e-10):
        
        if X_train.shape[0] != y_train.shape[0]:
            return print(f"Training examples are not same")
        
        self.examples = X_train.shape[0]
        self.features = X_train.shape[1]
        self.weights = np.zeros(shape= (self.features,1))
        self.bias = 0.0
        self.loss_history = list()
        
        for i in range(max_iter):  
            y_pred = self.predict(X_train)
            loss_reg = self.loss(y_train, y_pred)
            self.loss_history.append(loss_reg)
            grad_loss = - (y_train/y_pred)-((1-y_train)/(1-y_pred))
            grad_pred = np.multiply(y_pred,1-y_pred)
            grad_weights = np.matmul(X_train.T,np.multiply(grad_loss,grad_pred))
            grad_bias = np.sum(np.multiply(grad_loss,grad_pred))
            self.weights -= learning_rate*grad_weights
            self.bias -= learning_rate*grad_bias