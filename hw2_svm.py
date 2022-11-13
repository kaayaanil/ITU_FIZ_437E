# @anilk 2022
"""
ALL OF THE MODULES
models and utils which are in libs directory created by anilk 

PLEASE READ NOTES BELOW THE ALL CODES
"""
import numpy as np
from libs.utils import train_test_splitter,accuracy_score,grid_search,standardization
from libs.models import KNN,LogisticRegression,SVM
import matplotlib.pyplot as plt


np.random.seed(23)
# %% Data load and processing
data = np.load("data/data.npy",allow_pickle = True)
data = data.astype(np.int64)

y = data[:,-1]
X = data[:,:-1]

#For better result, standardization of data
X = standardization(X)*0.1


print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')

#train_test_splitter is function randomly select train and test data, specified ratio of test
X_train,X_test,y_train,y_test = train_test_splitter(X,y,0.1) 

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')


# %% Prediction
svm = SVM(alpha=0.1,n_iterations=1000,l=0.1)
svm.fit(X_train,y_train)
y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)

acc_train = accuracy_score(y_pred_train,y_train)
acc_test = accuracy_score(y_pred_test,y_test)

print(f"Acc train is: {acc_train}")
print(f"Acc test is: {acc_test}")

"""

Acc train is: 0.6357827476038339
Acc test is: 0.6826923076923077
 
"""