# @anilk 2022
"""
ALL OF THE MODULES
models and utils which are in libs directory created by anilk 

PLEASE READ NOTES BELOW THE ALL CODES
"""
import numpy as np
from libs.models import KNN
from libs.utils import train_test_splitter,accuracy_score,grid_search
import matplotlib.pyplot as plt

np.random.seed(23)
# %% Data load and processing
data = np.load("data/data.npy",allow_pickle = True)
data = data.astype(np.int64)

y = data[:,-1]
X = data[:,:-1]

print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')

#train_test_splitter is function randomly select train and test data, specified ratio of test
X_train,X_test,y_train,y_test = train_test_splitter(X,y,0.1) 

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')


# %% Prediction
knn = KNN(K=3,distance="euclidean")

knn.fit(X_train,y_train)

y_pred_test = knn.predict(X_test)
y_pred_train = knn.predict(X_train)

print(f"K = 3 test accuracy:",accuracy_score(y_test,y_pred_test))
print("K = 3 train accuracy:",accuracy_score(y_train,y_pred_train))

# %% Grid Search and Plot -d section of homework-
params = {'K':np.arange(1,29),'distance':["manhattan","euclidean"]}

results = grid_search(KNN,params,"accuracy",X_train,y_train,X_test,y_test,epoch =1)

if results["train_result"][0]==1 and results["train_result"][1]==1:
    print(f"Manhattan and euclidean distances for K=1 is 1 LOOKS GREATT")

plt.rcParams["figure.figsize"] = (7,8)
plt.figure(1)
plt.subplot(211)
plt.plot(params['K'],results["train_result"][::2],label="train")
plt.plot(params['K'],results['test_result'][::2],label="test")
plt.ylabel("Accuracy")
plt.title("Manhattan Distance K vs. Accuracy")
plt.grid()
plt.legend()

plt.subplot(212)
plt.plot(params['K'],results["train_result"][1::2],label="train")
plt.plot(params['K'],results['test_result'][1::2],label="test")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Euclidean Distance K vs. Accuracy")
plt.grid()
plt.legend()

plt.show()

"""

Notes:
All of the libraries created by me for using after assignments and projects
utils:  The library of fundemantal tools, functions metrics in machine learning
models: The library of include ML models

Please run codes show all of the figures and outputs
Output:

Shape of X: (1043, 2500)
Shape of y: (1043,)
Shape of X_train: (939, 2500)
Shape of X_test: (104, 2500)
K = 3 test accuracy: 0.6538461538461539
K = 3 train accuracy: 0.8093716719914803
For K is 1 and distance is manhattan, Train Accuracy = 1.0, Test Accuracy = 0.6538461538461539
For K is 1 and distance is euclidean, Train Accuracy = 1.0, Test Accuracy = 0.7211538461538461
For K is 2 and distance is manhattan, Train Accuracy = 0.8604898828541001, Test Accuracy = 0.6923076923076923
For K is 2 and distance is euclidean, Train Accuracy = 0.8743343982960596, Test Accuracy = 0.6634615384615384
For K is 3 and distance is manhattan, Train Accuracy = 0.8189563365282215, Test Accuracy = 0.7019230769230769
For K is 3 and distance is euclidean, Train Accuracy = 0.8093716719914803, Test Accuracy = 0.6538461538461539
For K is 4 and distance is manhattan, Train Accuracy = 0.8008519701810437, Test Accuracy = 0.7115384615384616
For K is 4 and distance is euclidean, Train Accuracy = 0.7816826411075612, Test Accuracy = 0.7019230769230769
For K is 5 and distance is manhattan, Train Accuracy = 0.7806176783812566, Test Accuracy = 0.7019230769230769
For K is 5 and distance is euclidean, Train Accuracy = 0.7678381256656017, Test Accuracy = 0.6538461538461539
For K is 6 and distance is manhattan, Train Accuracy = 0.7710330138445154, Test Accuracy = 0.6923076923076923
For K is 6 and distance is euclidean, Train Accuracy = 0.7497337593184239, Test Accuracy = 0.6826923076923077
For K is 7 and distance is manhattan, Train Accuracy = 0.7625133120340788, Test Accuracy = 0.6634615384615384
For K is 7 and distance is euclidean, Train Accuracy = 0.7561235356762513, Test Accuracy = 0.6538461538461539
For K is 8 and distance is manhattan, Train Accuracy = 0.7486687965921193, Test Accuracy = 0.6730769230769231
For K is 8 and distance is euclidean, Train Accuracy = 0.7380191693290735, Test Accuracy = 0.6346153846153846
For K is 9 and distance is manhattan, Train Accuracy = 0.7582534611288605, Test Accuracy = 0.6923076923076923
For K is 9 and distance is euclidean, Train Accuracy = 0.7358892438764644, Test Accuracy = 0.6634615384615384
For K is 10 and distance is manhattan, Train Accuracy = 0.7454739084132055, Test Accuracy = 0.7307692307692307
For K is 10 and distance is euclidean, Train Accuracy = 0.7294994675186368, Test Accuracy = 0.6730769230769231
For K is 11 and distance is manhattan, Train Accuracy = 0.739084132055378, Test Accuracy = 0.6634615384615384
For K is 11 and distance is euclidean, Train Accuracy = 0.7380191693290735, Test Accuracy = 0.6634615384615384
For K is 12 and distance is manhattan, Train Accuracy = 0.7412140575079872, Test Accuracy = 0.7307692307692307
For K is 12 and distance is euclidean, Train Accuracy = 0.7348242811501597, Test Accuracy = 0.6826923076923077
For K is 13 and distance is manhattan, Train Accuracy = 0.7412140575079872, Test Accuracy = 0.7115384615384616
For K is 13 and distance is euclidean, Train Accuracy = 0.7326943556975506, Test Accuracy = 0.6538461538461539
For K is 14 and distance is manhattan, Train Accuracy = 0.7348242811501597, Test Accuracy = 0.7307692307692307
For K is 14 and distance is euclidean, Train Accuracy = 0.7305644302449414, Test Accuracy = 0.6923076923076923
For K is 15 and distance is manhattan, Train Accuracy = 0.7294994675186368, Test Accuracy = 0.7019230769230769
For K is 15 and distance is euclidean, Train Accuracy = 0.731629392971246, Test Accuracy = 0.625
For K is 16 and distance is manhattan, Train Accuracy = 0.7231096911608094, Test Accuracy = 0.7211538461538461
For K is 16 and distance is euclidean, Train Accuracy = 0.7220447284345048, Test Accuracy = 0.6442307692307693
For K is 17 and distance is manhattan, Train Accuracy = 0.7252396166134185, Test Accuracy = 0.7019230769230769
For K is 17 and distance is euclidean, Train Accuracy = 0.7252396166134185, Test Accuracy = 0.6442307692307693
For K is 18 and distance is manhattan, Train Accuracy = 0.7188498402555911, Test Accuracy = 0.7211538461538461
For K is 18 and distance is euclidean, Train Accuracy = 0.7177848775292864, Test Accuracy = 0.6923076923076923
For K is 19 and distance is manhattan, Train Accuracy = 0.7199148029818956, Test Accuracy = 0.6634615384615384
For K is 19 and distance is euclidean, Train Accuracy = 0.7188498402555911, Test Accuracy = 0.6634615384615384
For K is 20 and distance is manhattan, Train Accuracy = 0.7103301384451545, Test Accuracy = 0.6923076923076923
For K is 20 and distance is euclidean, Train Accuracy = 0.7188498402555911, Test Accuracy = 0.6634615384615384
For K is 21 and distance is manhattan, Train Accuracy = 0.7231096911608094, Test Accuracy = 0.6730769230769231
For K is 21 and distance is euclidean, Train Accuracy = 0.7092651757188498, Test Accuracy = 0.6346153846153846
For K is 22 and distance is manhattan, Train Accuracy = 0.7177848775292864, Test Accuracy = 0.7115384615384616
For K is 22 and distance is euclidean, Train Accuracy = 0.7103301384451545, Test Accuracy = 0.6826923076923077
For K is 23 and distance is manhattan, Train Accuracy = 0.7252396166134185, Test Accuracy = 0.6538461538461539
For K is 23 and distance is euclidean, Train Accuracy = 0.7103301384451545, Test Accuracy = 0.6730769230769231
For K is 24 and distance is manhattan, Train Accuracy = 0.7145899893503728, Test Accuracy = 0.6730769230769231
For K is 24 and distance is euclidean, Train Accuracy = 0.7145899893503728, Test Accuracy = 0.6826923076923077
For K is 25 and distance is manhattan, Train Accuracy = 0.7167199148029819, Test Accuracy = 0.6442307692307693
For K is 25 and distance is euclidean, Train Accuracy = 0.7028753993610224, Test Accuracy = 0.6538461538461539
For K is 26 and distance is manhattan, Train Accuracy = 0.7103301384451545, Test Accuracy = 0.6634615384615384
For K is 26 and distance is euclidean, Train Accuracy = 0.7007454739084132, Test Accuracy = 0.6923076923076923
For K is 27 and distance is manhattan, Train Accuracy = 0.7135250266240681, Test Accuracy = 0.6826923076923077
For K is 27 and distance is euclidean, Train Accuracy = 0.6879659211927582, Test Accuracy = 0.6826923076923077
For K is 28 and distance is manhattan, Train Accuracy = 0.7092651757188498, Test Accuracy = 0.6634615384615384
For K is 28 and distance is euclidean, Train Accuracy = 0.6986155484558041, Test Accuracy = 0.6826923076923077

Manhattan and euclidean distances for K=1 is 1 LOOKS GREATT
"""