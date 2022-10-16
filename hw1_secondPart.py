"""
FIZ437E HOMEWORK 1 SECOND PART
ANIL FERDI KAYA 090180128
PLEASE READ NOTES WHIC BELOW THE ALL CODES
"""


import numpy as np
from scipy.special import j0
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge

np.random.seed(30)

def l2_loss(y_true,y_test):
    return np.sum((y_true-y_test)**2)

def mse(y_true,y_test):
    return np.sum((y_true-y_test)**2)/y_true.shape[0]
    
def getRandomIndex(pop_size,sample_size,test_ratio):
    idxs = np.random.choice(range(pop_size),sample_size,replace = False)
    train_idxs = idxs[:int(idxs.shape[0]*(1-test_ratio))]
    test_idxs = idxs[int(idxs.shape[0]*(1-test_ratio)):]

    train_idxs.sort()
    test_idxs.sort()
    return train_idxs,test_idxs

#%% START

"""SECTION 1"""
NUM_POINTS = 10000
x = np.linspace(0,8,NUM_POINTS) #X poıints of data
plt.plot(x,j0(x))
plt.title("Bassel function without noises")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

"""SECTION 2"""
noises = np.random.normal(0,1,NUM_POINTS) #Gauss noises
sns.histplot(noises,kde=True)
plt.title("Histogram of Noises -can be gauss dist.-")
plt.show()

y = j0(x)+(noises)*0.1 #New data with noises*0.1
plt.plot(x,y)
plt.title("Bassel function with noises")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

# %% Regression without ridge
print("REGRESSION WITHOUT RIDGE RESULTS:")
"""
SECTION 4 AND 5
"""
for sample_size in [10,20,100,1000,10**4]:
    train_idxs,test_idxs = getRandomIndex(NUM_POINTS,sample_size,0.2)

    X_train = x[train_idxs].reshape(-1,1)
    y_train = y[train_idxs].reshape(-1,1)
    X_test = x[test_idxs].reshape(-1,1)
    y_test = y[test_idxs].reshape(-1,1)

    """
    SECTION 3
    """
    poly = PolynomialFeatures(degree=8) 
    X_ = poly.fit_transform(X_train)
    polyReg = LinearRegression()
    polyReg.fit(X_, y_train)

    y_pred_train = polyReg.predict(X_)

    X_t = poly.fit_transform(X_test)
    y_pred_test = polyReg.predict(X_t)

    plt.plot(x,y,alpha = 0.4)
    plt.title(f"Bassel function training and test No:{sample_size} Samples Without Ridge")
    plt.scatter(X_train,y_train,c="r",label = "train")
    plt.scatter(X_test,y_test,c="g",label = "test")
    plt.scatter(X_test,y_pred_test,c='purple',label = "prediction test")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    
    print(f"Test set L2_loss for {sample_size} size: {l2_loss(y_test,y_pred_test)}, MSError: {mse(y_test,y_pred_test)}")

# %% Regression with ridge
"""SECTION 6"""
print("REGRESSION WITH RIDGE RESULTS:")
for sample_size in [10,20,100,1000,10**4]:
    train_idxs,test_idxs = getRandomIndex(NUM_POINTS,sample_size,0.2)

    X_train = x[train_idxs].reshape(-1,1)
    y_train = y[train_idxs].reshape(-1,1)
    X_test = x[test_idxs].reshape(-1,1)
    y_test = y[test_idxs].reshape(-1,1)


    poly = PolynomialFeatures(degree=8)
    X_ = poly.fit_transform(X_train)
    polyRidge = Ridge(alpha=0.001,normalize=True)
    polyRidge.fit(X_, y_train)

    y_pred_train = polyRidge.predict(X_)

    X_t = poly.fit_transform(X_test)
    y_pred_test = polyRidge.predict(X_t)

    plt.plot(x,y,alpha = 0.4)
    plt.title(f"Bassel function training and test No:{sample_size} Samples With Ridge")
    plt.scatter(X_train,y_train,c="r",label = "train")
    plt.scatter(X_test,y_test,c="g",label = "test")
    plt.scatter(X_test,y_pred_test,c='purple',label = "prediction test")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Test set L2_loss for {sample_size} size: {l2_loss(y_test,y_pred_test)}, MSError: {mse(y_test,y_pred_test)}")


"""
Please run codes show all of the figures and outputs
OUTPUT:

REGRESSION WITHOUT RIDGE RESULTS:
Test set L2_loss for 10 size: 5.269321579705227, MSError: 2.6346607898526133
Test set L2_loss for 20 size: 0.2438575570633934, MSError: 0.06096438926584835
Test set L2_loss for 100 size: 0.277246165408938, MSError: 0.0138623082704469
Test set L2_loss for 1000 size: 1.8179708084181136, MSError: 0.009089854042090569
Test set L2_loss for 10000 size: 20.019241079211376, MSError: 0.010009620539605689
REGRESSION WITH RIDGE RESULTS:
Test set L2_loss for 10 size: 0.0020169491834193653, MSError: 0.0010084745917096826
Test set L2_loss for 20 size: 0.02593041127204212, MSError: 0.00648260281801053
Test set L2_loss for 100 size: 0.3124057141891285, MSError: 0.015620285709456425
Test set L2_loss for 1000 size: 3.6085503405447366, MSError: 0.018042751702723684
Test set L2_loss for 10000 size: 31.744255266784066, MSError: 0.015872127633392033

Notes: Without Ridge -L2 (alpha)- regulaztion, the data be overfit in min sample sizes 

Also, L2Loss can be increase due to the sample size increase. 
But, mean squared error -diveded sample size to m l2 loss- decresase

So, if the data increase the performance of the model increase.
 

"""

