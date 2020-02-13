from sklearn import ensemble
import numpy as np
from scipy import stats
import second_neural_network
import first_neural_network



class EnsembleOrder():
    
    def __init__(self):
        
        self.model1 = second_neural_network.second_neural_network()
        self.model2 = first_neural_network.first_neural_network()
        
       
    def fit(self, X,y):
        self.model1.fit(X,y)
        self.model2.fit(X,y)


    def predict(self, X):
        #return self.model.predict(X)
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        print(pred2.shape)
        avg_pred = np.mean( np.array([ pred1, pred2 ]), axis=0 )
        print(avg_pred.shape)

        return avg_pred

