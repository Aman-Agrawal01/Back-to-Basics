import numpy as np

class linear_reg():
    
    def predict(self,X_train):
        return np.dot(X_train,self.weights) + self.bias
    
    def loss(self, y_train, y_pred):
        return np.sum(np.square(y_pred - y_train)*(1/(2*self.examples)))
    
    def fit(self,X_train, y_train, max_iter = 100, learning_rate = 1e-7):
        
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
            grad_weights = np.dot(np.transpose(X_train), y_pred - y_train)/self.examples
            grad_bias = np.sum(y_pred - y_train)/self.examples
            self.weights -= learning_rate*grad_weights
            self.bias -= learning_rate*grad_bias