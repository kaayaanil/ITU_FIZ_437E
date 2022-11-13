# @anilk 2022
"""
ALL OF THE MODULES
models and utils which are in libs directory created by anilk 

PLEASE READ NOTES BELOW THE ALL CODES
"""
import numpy as np
from libs.utils import train_test_splitter,accuracy_score,grid_search,standardization,normalization
from libs.models import KNN,LogisticRegression,THNN
import matplotlib.pyplot as plt


np.random.seed(23)
# %% Data load and processing
data = np.load("data/data.npy",allow_pickle = True)
data = data.astype(np.int64)

y = data[:,-1]
X = data[:,:-1]

#For better result, standardization of data
X = standardization(X)


print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')

#train_test_splitter is function randomly select train and test data, specified ratio of test
X_train,X_test,y_train,y_test = train_test_splitter(X,y,0.1) 

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')


# %% Prediction
model = THNN(n_iterations=10000,layer_sizes=(4,4),verbose=1,alpha=0.1)

loss_train, loss_test = model.fit(X_train.T,y_train,eval_set=(X_test.T,y_test))

plt.plot(loss_train,label="train")
plt.plot(loss_test,label="test")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss vs Iteration")
plt.legend()
plt.savefig("loss_two_hidden.png")
plt.show()

y_pred = model.predict(X_test.T)
print(f"Accuracy score last iteration test: {accuracy_score(y_pred,y_test)}")


