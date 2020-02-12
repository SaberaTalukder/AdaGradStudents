import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

"""
This program performs calculations on the model.
"""

def get_val_err(num_folds, train, model):
    """Gets the average validation error of model across num_folds cross-validation folds."""

    kf = KFold(n_splits=num_folds)

    err_list = []

    # Iterate through cross-validation folds:
    i = 1
    print(len(train))
    for train_index_list, val_index_list in kf.split(train):

        # Print out test indices:
        print('Fold ', i, ' of ', num_folds, ' test indices:', val_index_list)
        print(len(val_index_list))
        # Training and testing data points for this fold:
        x_train, x_val = train.drop(['id','date','y'], axis=1).iloc[train_index_list], train.drop(['id','y','date'], axis=1).iloc[val_index_list]
        y_train, y_val = train[['y']].iloc[train_index_list], train[['y']].iloc[val_index_list]
        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        #val_err = log_loss(y_val, y_pred)
        val_err = roc_auc_score(y_val, y_pred)
        print(val_err)
        
        i += 1
        err_list.append(val_err)

    avg_err = np.mean(err_list)
    var_err = np.var(err_list)
    return avg_err, var_err, err_list
