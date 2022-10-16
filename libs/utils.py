"""
    ALL OF THE MODULES
    models and utils created by anilk

utils: fundamental tools for machine learning
"""
import numpy as np
from models import KNN       #models is machine learning algorithms creating by anilk

# %% Data processing functions
def train_test_splitter(X,y,size):
    """
    data_fun.ipynb for check the function
    input:
        X: data of X
        y: data of y
        size: test size for split

    output:
        X_train
        X_test
        y_train
        y_test
    """
    num_data = X.shape[0]
    num_test = int(num_data*size)

    idxs = np.arange(num_data)
    np.random.shuffle(idxs)

    test_idxs = idxs[:num_test]
    test_idxs = np.sort(test_idxs)

    train_idxs = idxs[num_test:]
    train_idxs = np.sort(train_idxs)

    X_train = X[train_idxs]
    X_test = X[test_idxs]

    y_train = y[train_idxs]
    y_test = y[test_idxs]

    return X_train,X_test,y_train,y_test

# %% Model selections
def grid_search(model,params,metric,X_train,y_train,X_test,y_test,epoch):
    """
    Search test and train results for given metric (accuracy,AUC,f1 etc.)
    inputs
        model:           ML model
        params:          model params
        metric:          evaluation metric
        X_train,y_train: train data
        X_test,y_test:   test data
        epoch            0: don't show outputs, 1: show outputs
    outputs
        results(dict) : dict for specified parameters and results (param1,param2,...,train_result,test_result)
    """
    if model.__name__=="KNN":
        results = {'K':[],'distance':[],'train_result':[],'test_result':[]}
        
        k_list = params["K"]
        distance_list = params["distance"]

        
        for k in k_list:
            for distance in distance_list:

                model = KNN(K = k,distance = distance)
                model.fit(X_train,y_train)
                y_pred_test = model.predict(X_test)
                y_pred_train = model.predict(X_train)

                if metric=="accuracy":
                    train_result=accuracy_score(y_train,y_pred_train)
                    test_result = accuracy_score(y_test,y_pred_test)
                if epoch == 1:
                    print(f"For K is {k} and distance is {distance}, Train Accuracy = {train_result}, Test Accuracy = {test_result}")
                
                results["K"].append(k)
                results["distance"].append(distance)
                results["train_result"].append(train_result)
                results["test_result"].append(test_result)
        
        return results

# %%Metrics
def accuracy_score(y_true,y_pred):
    accuracy = sum(y_true == y_pred)/y_true.shape[0]
    return accuracy

euclidean_distance = lambda X,x_pr :np.sqrt(np.sum((X-x_pr)**2,axis=1))

manhattan_distance = lambda X,x_pr :np.sum(np.abs((X-x_pr)),axis=1)




 