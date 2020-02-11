from orderbookmodel import OrderBookModel
from sklearn import ensemble



class RandomForestOrder(OrderBookModel):
    
    def __init__(self):
        model = ensemble.RandomForestClassifier()
        
        
    def fit(self, X,y):
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(X)

