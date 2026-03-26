import numpy as np
import matplotlib.pyplot as plt
import inspect



data_digits = np.load(r"math4ai_capstone/starter_pack/data/digits_data.npz")
digits_split_indices = np.load(r"math4ai_capstone/starter_pack/data/digits_split_indices.npz")
data_linear = np.load(r"math4ai_capstone/starter_pack/data/linear_gaussian.npz")
data_moons = np.load(r"math4ai_capstone/starter_pack/data/moons.npz")


X_d_train = data_digits["X"][digits_split_indices["train_idx"]]
y_d_train = data_digits["y"][digits_split_indices["train_idx"]]
X_d_test = data_digits["X"][digits_split_indices["test_idx"]]
y_d_test = data_digits["y"][digits_split_indices["test_idx"]]
X_d_val = data_digits["X"][digits_split_indices["val_idx"]]
y_d_val = data_digits["y"][digits_split_indices["val_idx"]]


X_l_train=data_linear['X_train']
y_l_train= data_linear['y_train']
X_l_val=data_linear['X_val']
y_l_val= data_linear['y_val']
X_l_test=data_linear['X_test'] 
y_l_test=data_linear['y_test']


X_m_train=data_moons['X_train']
y_m_train= data_moons['y_train']
X_m_val=data_moons['X_val']
y_m_val= data_moons['y_val']
X_m_test=data_moons['X_test'] 
y_m_test=data_moons['y_test']


def accuracy(y_pred,y_test):
    return y_pred[y_pred==y_test].size/y_test.size


class Validation():
    def __init__(self,estimator,X,y,X_val,y_val,params = {},cv=5):
        self.estimator=estimator
        self.cv = cv
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.best_model = None
        self.results = []
        self.models = []
        self.params = params
        
    def fit(self):
        for i in range(self.cv):
            np.random.seed(i)
            model = self.estimator(**self.params);
            model.fit(self.X,self.y)
            y_pred = model.predict(self.X_val)
            self.results.append(self.accuracy(y_pred))
            self.models.append(model)
        return max(self.results),self.models[np.argmax(self.results)]

    def report(self):
        loss = [m.loss[-1] for m in self.models]
        l_m = np.mean(loss)
        a_m = np.mean(self.results)
        l_s = np.std(loss,ddof=1)
        a_s = np.std(self.results,ddof=1)
        
        confidence_interval_for_loss = (l_m - 2.776*l_s/np.sqrt(self.cv),l_m + 2.776*l_s/np.sqrt(self.cv))
        confidence_interval_for_accuracy = (a_m - 2.776*a_s/np.sqrt(self.cv),a_m + 2.776*a_s/np.sqrt(self.cv))
        
        low_m_a, high_m_a = confidence_interval_for_accuracy
        low_m_l, high_m_l = confidence_interval_for_loss
        
        print(f"--- Statistics for {self.cv} Seeds ---")
        print(f"Accuracy mean:{a_m:.5f}")
        print(f"Loss mean for {self.cv} models:{l_m:.5f}")
        print(f"95 % Confidence Interval for Accuracy mean:({low_m_a:.5f},{high_m_a:.5f})")
        print(f"95 % Confidence Interval for Loss mean:({low_m_l:.5f},{high_m_l:.5f})")
    
    def accuracy(self,y_pred):
        return y_pred[y_pred==self.y_val].size/self.y_val.size




class SoftMaxClassification:
    def __init__(self,penalty = "l2",lamda = 1e-4,learning_rate = 0.05,max_iter = 5000,batch_size=64):
        self.penalty = "l2"
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.loss = []
    
    def fit(self,X_train,y_train):
        self.n,self.m = X_train.shape
        number_of_class = np.unique(y_train).size
        Y = np.eye(number_of_class)[y_train]
        self.W = np.random.randn(number_of_class,self.m+1)*0.01
        X = np.hstack((X_train,np.ones((self.n,1))))
        idxs = np.arange(self.n)
        for _ in range(self.max_iter):
            np.random.shuffle(idxs)
            X_shuffled = X[idxs]
            Y_shuffled = Y[idxs]
            for j in range(0,self.n,self.batch_size):
                X_batch = X_shuffled[j:j+self.batch_size]
                Y_batch = Y_shuffled[j:j+self.batch_size]
                self.back_propagation(X_batch,Y_batch)
            softmax = self.softmax(X)
            L = - np.mean(np.sum(Y*np.log(softmax+1e-9),axis=1)) + self._l2()
            self.loss.append(L)
            
    def _l2(self):
        return self.lamda * np.sum(np.power(self.W[:,:-1],2))
        
    
    def softmax(self,X):
        Z = X @ self.W.T
        Z -= np.max(Z, axis=1, keepdims=True)
        S = np.exp(Z)
        return S/np.sum(S,axis=1,keepdims=True)
            
    def back_propagation(self,X,Y):
        softmax = self.softmax(X)
        dLdZ = softmax - Y
        dZdW = X
        reg_term = 2 * self.lamda * self.W
        reg_term[:,-1]= 0
        dLdW =  (dLdZ.T @ X)/X.shape[0] + reg_term
        self.W -= self.learning_rate * dLdW
        
    def predict(self,X_val):
        X_val = np.hstack((X_val,np.ones((X_val.shape[0],1))))
        result = self.softmax(X_val)
        return np.argmax(result,axis=1)
        
    def plot_loss(self):
        plt.figure(figsize=(8,6))
        plt.plot(range(self.max_iter),self.loss)
        plt.ylabel("Log Loss")
        plt.yticks(np.linspace(min(self.loss), max(self.loss), 10))
        plt.xlabel("Iteration")
        plt.grid()
        plt.show()
        
        
        
        