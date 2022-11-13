# @anilk 2022
"""
ALL OF THE MODULES
models and utils which are in libs directory created by anilk 

PLEASE READ NOTES BELOW THE ALL CODES
"""
import numpy as np
from libs.utils import train_test_splitter,accuracy_score,grid_search,standardization
from libs.models import KNN,LogisticRegression
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
logReg = LogisticRegression(n_iteartions=10000,verbose=1,alpha=0.01)

acc_train,acc_test = logReg.fit(X_train.T,y_train,eval_set=(X_test.T,y_test),early_stopping_round=3000)

y_pred = logReg.predict(X_test.T)
y_probs = logReg.predict(X_test.T,get_probs=True)

plt.plot(acc_train,label="train")
plt.plot(acc_test,label="test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.savefig("loss_logReg.png")



print(f"Prediction accuracy is {accuracy_score(y_pred,y_test)} ")
