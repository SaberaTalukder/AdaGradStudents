from sklearn import ensemble



class RandomForestOrder():
    
    def __init__(self,max_depth=10):
        
        self.model = ensemble.RandomForestClassifier(n_estimators=100,max_depth=max_depth)
        
        
    def fit(self, X,y):
        self.model.fit(X,y)

    def predict(self, X):
        #return self.model.predict(X)
        return self.model.predict_proba(X)[:,1]

