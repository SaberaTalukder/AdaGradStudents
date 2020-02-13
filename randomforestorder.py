from sklearn import ensemble
import numpy as np
from scipy import stats



class RandomForestOrder():
    
    def __init__(self,max_depth=10):
        
        self.model = ensemble.RandomForestClassifier(n_estimators=100,max_depth=10) #100 and 10
        
       
    def normalize_vals(self, X):
        X_temp = np.asarray(X)
        std_dev = np.std(X_temp, axis=0)
        mean_val = np.mean(X_temp, axis=0)
        normalized_X = stats.zscore(X_temp, axis=0)

        return normalized_X, mean_val, std_dev

    def fit(self, X,y):

        X, mean_val, std_dev = self.normalize_vals(X)
        self.mean_val = mean_val
        self.std_dev = std_dev

        self.model.fit(X,y)

    def predict(self, X):
        #return self.model.predict(X)
        #X, mean_val, std_dev = self.normalize_vals(X)
        
        X = np.asarray(X)
        X = (X - self.mean_val) / self.std_dev
        return self.model.predict_proba(X)[:,1]


