"""
    ALL OF THE MODULES
    models and utils created by anilk

models: Models for machine learning
"""


import numpy as np
import libs.utils as utils #Utils is a library for primary and most-used anilk metrics, functions

class KNN():
    def __init__(self,K=None,distance=None):
        self.K = K
        self.distance = distance
    
    def fit(self,X,y):
        """
        X (mx,d): Train of X for KNN algo
        y (mx,1): Train of y labels for KNN algo
        """
        self.X = X
        self.y = y
    
    def predict(self,X_pr):
        """
        X_pr(p,d): Predicted X for KNN
        """
        pred_labels = np.array([])
        for i in range(X_pr.shape[0]):

            if self.distance=="euclidean":                           #Calculate euclidean distance for prediction (sum((X-x)**2))**1/2
                distances = utils.euclidean_distance(self.X,X_pr[i])    
            elif self.distance=="manhattan":                         #Calculate manhattan distance for prediction (sum(|X-x|))
                distances = utils.manhattan_distance(self.X,X_pr[i])
                
            idx = np.argsort(distances)[:self.K]                    #Get lowest distances indexs
            lowest_distances = self.y[idx]                          #Get lowest distances
            label = np.bincount(lowest_distances).argmax()          #Bincount is function for counting 0 and 1s sequantly, argmax chooses most freq 0 or 1
            pred_labels = np.append(pred_labels,label)

        return pred_labels

class LogisticRegression():
    """
    Logistic Regression Class for Logistic Regression Operations
    """
    def __init__(self,n_iteartions = 1000,alpha = 0.1,verbose = 0):
        
        """init parameters for logistic regresison

            Args:
                n_iterations (int): number of iteretion
            Returns:
                nothing
            """
        self.verbose = verbose
        self.n_iterations = n_iteartions
        self.alpha = alpha
    
    def fit(self,X,y,eval_set=None,early_stopping_round = 100):
        """train data for logistic regresison

            Args:
                X (np.ndarray): X of training set (params,data_size)
                y (np.ndarray): y -labels- of traing set
            Returns:
                nothing
        """
            
        self.X = X
        self.y = y
        
        if eval_set != None:
            X_test=eval_set[0]
            y_test = eval_set[1]
            
            print(X_test.shape)
            print(y_test.shape)
        
        n = X.shape[0]
        m = X.shape[1]
        self.W = np.zeros((1,n))*0.0001
        self.b = 0
        
        W_s = [self.W]
        b_s = [self.b]
        
        epoch_scores_train = []
        epoch_scores_test = []
        for epoch in range(self.n_iterations):

            Z = np.dot(self.W,X) + self.b
            A = utils.sigmoid(Z)
            
            y_hat_train = np.where(A.reshape(-1,)>0.5,1,0)
            acc_train = utils.accuracy_score(y_hat_train,y)
            epoch_scores_train.append(acc_train)
            
            if eval_set != None:
                Z_test = np.dot(self.W,X_test) + self.b
                A_test = utils.sigmoid(Z_test)        
                y_hat_test = np.where(A_test.reshape(-1,)>0.5,1,0)
                acc_test = utils.accuracy_score(y_hat_test,y_test)
                epoch_scores_test.append(acc_test)
                
            dZ = A-y
            dw = (1/m)*np.dot(X,dZ.T).reshape(1,-1)
            db = (1/m)*np.sum(dZ)
            

            if self.verbose == 1:

                print(f"for {epoch} epoch train accuracy: {acc_train}")
                if eval_set!= None:
                    print(f"for {epoch} epoch test accuracy: {acc_test}")
                
            self.W = self.W - (self.alpha*dw)
            self.b = self.b - (self.alpha*db)
            
            W_s.append(self.W)
            b_s.append(self.b)
            
            if eval_set!= None:
                arg_max_test_results = np.argmax(epoch_scores_test)
                if len(epoch_scores_test) - arg_max_test_results>early_stopping_round:
                    print(f"EARLY STOPPING EPOCH {arg_max_test_results}, score: {epoch_scores_test[arg_max_test_results]}")
                    self.W = W_s[arg_max_test_results]
                    self.b = b_s[arg_max_test_results]
                    break
                if epoch+1 == self.n_iterations:
                    print(f"BEST EPOCH {arg_max_test_results}, score: {epoch_scores_test[arg_max_test_results]}")
                    self.W = W_s[arg_max_test_results]
                    self.b = b_s[arg_max_test_results]                   
                
        if eval_set != None:
            return epoch_scores_train,epoch_scores_test
        
    def predict(self,X,get_probs=False):
        
        Z = np.dot(self.W,X) + self.b
        A = utils.sigmoid(Z)
        
        if get_probs == True:
            return A
        else:
            return np.array([0 if i<0.5 else 1 for i in A.reshape(-1,)])
        
class THNN():
    def __init__(self,layer_sizes,n_iterations,verbose = 0,alpha=0.05):
        

        self.params = {}
        self.layer_sizes = layer_sizes
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.alpha = alpha
        
    def fit(self,X,y,eval_set):
        
        self.m = X.shape[1]
        self.n = X.shape[0]
        self.params["W1"] = np.random.randn(self.layer_sizes[0],self.n)*.1
        self.params["W2"] = np.random.randn(self.layer_sizes[1],self.layer_sizes[0])*.1
        self.params["W3"] = np.random.randn(1,self.layer_sizes[1])*.1
        self.params["b1"] = np.zeros((self.layer_sizes[0],1))
        self.params["b2"] = np.zeros((self.layer_sizes[1],1))
        self.params["b3"] = np.zeros((1,1))
        train_losses = []
        test_losses = []
        
        
        for i in range(self.n_iterations):
            Z1 = np.dot(self.params["W1"],X) + self.params["b1"]
            A1 = utils.sigmoid(Z1)
            Z2 = np.dot(self.params["W2"],A1) + self.params["b2"]
            A2 = utils.sigmoid(Z2)
        
            Z3 = np.dot(self.params["W3"],A2) + self.params["b3"]
            A3 = utils.sigmoid(Z3)
            
            dZ3 = A3-y
            dW3 = np.dot(dZ3,A2.T) / self.m
            db3 = (np.squeeze(np.sum(dZ3, axis=1, keepdims=True)) / self.m).reshape(-1,1)


            
            dA2 = np.dot(self.params["W3"].T,dZ3)
            dZ2 = dA2*utils.sigmoid(Z2,get_derivative=True)
            dW2 = np.dot(dZ2,A1.T) / self.m
            db2 = (np.squeeze(np.sum(dZ2, axis=1, keepdims=True)) / self.m).reshape(-1,1)
            
            dA1 = np.dot(self.params["W2"].T,dZ2)
            dZ1 = dA1*utils.sigmoid(Z1,get_derivative=True)
            dW1 = np.dot(dZ1,X.T) / self.m
            db1 = (np.squeeze(np.sum(dZ1, axis=1, keepdims=True)) / self.m).reshape(-1,1)

            if self.verbose ==1:
                loss_train = utils.classification_loss(np.squeeze(np.where(A3>0.5,1,0)),y)
                print(f'Loss score train {i}: {loss_train}')
                train_losses.append(loss_train)
                
                
            if eval_set != None and self.verbose ==1:
                y_pred = self.predict(eval_set[0])
                loss_test = utils.classification_loss(y_pred,eval_set[1])
                print(f'Loss score test {i}: {loss_test}')
                test_losses.append(loss_test)
            
            self.params["W1"] = self.params["W1"] - self.alpha*dW1
            self.params["W2"] = self.params["W2"] - self.alpha*dW2
            self.params["W3"] = self.params["W3"] - self.alpha*dW3

            
            self.params["b1"] = self.params["b1"] - self.alpha*db1
            self.params["b2"] = self.params["b2"] - self.alpha*db2
            self.params["b3"] = self.params["b3"] - self.alpha*db3

        return train_losses,test_losses

        
    def predict(self,X):
            Z1 = np.dot(self.params["W1"],X) + self.params["b1"]
            A1 = utils.sigmoid(Z1)
            Z2 = np.dot(self.params["W2"],A1) + self.params["b2"]
            A2 = utils.sigmoid(Z2)
        
            Z3 = np.dot(self.params["W3"],A2) + self.params["b3"]
            A3 = utils.sigmoid(Z3)
            
            y_hat = np.where(A3>0.5,1,0)
            
            return np.squeeze(y_hat)
        
class SVM():
    
    def __init__(self,alpha=0.1,n_iterations=1000,l=0.1):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.l=l
        
        
    def fit(self,X,y):
        self.W=np.zeros(X.shape[1])
        self.b=0
        
        self.X=X
        self.Y=y
        
        train_losses = []
        test_losess = []
        

        y_labels=np.where(self.Y<=0,-1,1)
        for i in range(self.n_iterations):
            y_pred = self.predict(X)
            for index,x_i in enumerate(self.X):
                condition=y_labels[index]*(np.dot(x_i,self.W)-self.b)>=1
                if(condition==True):
                    dW=2*self.l*self.W
                    db=0
                else:
                    dW=2*self.l*self.W-np.dot(x_i,y_labels[index])
                    db=y_labels[index]
                
                self.W=self.W-self.alpha*dW
                self.b=self.b-self.alpha*db
                
            
            
    def predict(self,X):
        o1=np.dot(X,self.W)-self.b
        preds=np.sign(o1)
        y_hat=np.where(preds<=-1,0,1)
        return y_hat
        
            
            
            