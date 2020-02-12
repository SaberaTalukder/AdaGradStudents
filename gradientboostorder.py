import numpy as np
from scipy.stats import logistic
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


class GradientBoosting():
    
    def __init__(self, n_clfs=100):
        '''
        Initialize the gradient boosting model.

        Inputs:
            n_clfs (default 100): Initializer for self.n_clfs.

        Attributes:
            self.n_clfs: The number of DT weak regressors.
            self.clfs: A list of the DT weak regressors, initialized as empty.
        '''
        self.n_clfs = n_clfs
        self.clfs = []
        

    def fit(self, X, Y, n_nodes=4):
        '''
        Fit the gradient boosting model. Note that since we are implementing this method in a class,
        rather than having a bunch of inputs and outputs, you will deal with the attributes of the class.
        (see the __init__() method).
        
        This method should thus train self.n_clfs DT weak regressors and store them in self.clfs.

        Inputs:
            X: A (N, D) shaped numpy array containing the data points.
            Y: A (N, ) shaped numpy array containing the (float) labels of the data points.
               (Even though the labels are ints, we treat them as floats.)
            n_nodes: The max number of nodes that the DT weak regressors are allowed to have.
        '''
        self.clfs = []
        for i in range(self.n_clfs):
            clf = DecisionTreeRegressor(max_leaf_nodes=n_nodes)
            print("clf i: %d" % i)
            Y_list = np.zeros(len(Y))
            if i > 0:
                Y_list = Y['y'].to_numpy()
                Y_list = Y_list - logistic.cdf(self.clfs[i-1].predict(X))
            clf.fit(X, Y_list)
            self.clfs.append(clf)


    def predict(self, X):
        '''
        Predict on the given dataset.

        Inputs:
            X: A (N, D) shaped numpy array containing the data points.

        Outputs:
            A (N, ) shaped numpy array containing the (float) labels of the data points.
            (Even though the labels are ints, we treat them as floats.)
        '''
        # Initialize predictions.
        Y_pred = np.zeros(len(X))

        # Add predictions from each DT weak regressor.
        for i, clf in enumerate(self.clfs):
            print(i)
            Y_curr = logistic.cdf(clf.predict(X))
            print(Y_curr)
            Y_pred += Y_curr

        Y_pred /= self.n_clfs

        return Y_pred

