"""
    ALL OF THE MODULES
    models and utils created by anilk

models: Models for machine learning
"""


import numpy as np
import utils #Utils is a library for primary and most-used anilk metrics, functions

class KNN():
    def __init__(self,K,distance):
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
