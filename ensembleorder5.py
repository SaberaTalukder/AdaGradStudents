from sklearn import ensemble
import numpy as np
from scipy import stats
import second_neural_network
import first_neural_network
import third_neural_network
import randomforestorder


class EnsembleOrder():
    
    def __init__(self):
        
        self.model1 = second_neural_network.second_neural_network()
        self.model2 = first_neural_network.first_neural_network()
        self.model3 = randomforestorder.RandomForestOrder()
        self.model4 = third_neural_network.third_neural_network()
        
       
    def fit(self, X,y):
        self.model1.fit(X,y)
        self.model2.fit(X,y)
        self.model3.fit(X,y)
        self.model4.fit(X,y)


    def predict(self, X):
        #return self.model.predict(X)
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        pred3 = np.expand_dims(self.model3.predict(X), 1)
        pred4 = self.model4.predict(X)
        print(pred1.shape)
        print(pred2.shape)
        print(pred3.shape)
        print(pred4.shape)
        avg_pred = np.mean( np.array([ pred1, pred2, pred3, pred4 ]), axis=0 )
        print(avg_pred.shape)

        return avg_pred

