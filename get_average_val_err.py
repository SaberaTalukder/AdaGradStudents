from sklearn.model_selection import KFold
import orderbookmodel

def get_avg_val_err(num_folds):
    """Gets the average validation error of model across num_folds cross-validation folds."""

    num_folds = 10
    kf = KFold(n_splits=num_folds)

    train_sets = []
    val_sets = []

    # Iterate through cross-validation folds:
    i = 1
    print(len(train))
    for train_index_list, val_index_list in kf.split(train):

        # Print out test indices:
        print('Fold ', i, ' of ', num_folds, ' test indices:', val_index_list)
        print(len(val_index_list))
        # Training and testing data points for this fold:
        x_train, x_val = train.drop('y', axis=1).iloc[train_index_list], train.drop('y', axis=1).iloc[val_index_list]
        y_train, y_val = train[['y']].iloc[train_index_list], train[['y']].iloc[val_index_list]

        
        i += 1
    return avg_err
