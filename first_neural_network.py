import numpy as np
import torch
import torch.nn as nn

from orderbookmodel import OrderBookModel
from scipy import stats

class first_neural_network(OrderBookModel):

    def __init__(self):
        # number of input parameters is 39
        self.loss_fn = nn.BCELoss()
        self.learning_rate = 1e-3
        self.batch_s = 32
        self.num_epochs = 10
        self.dropout = 0.1
        self.mean_val = 0
        self.std_dev = 0

        self.model = nn.Sequential(
            nn.Linear(39, 200),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def normalize_vals(self, X):

        X_temp = np.asarray(X)
        std_dev = np.std(X_temp, axis=0)
        mean_val = np.mean(X_temp, axis=0)
        normalized_X = stats.zscore(X_temp, axis=0)

        return normalized_X, mean_val, std_dev


    def fit(self, X, y):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = self.loss_fn

        X, mean_val, std_dev = self.normalize_vals(X)
        self.mean_val = mean_val
        self.std_dev = std_dev

        '''
        Sharon's code is not going to work with the normalize values function because it changes the
        X values in to a numpy array not into a pandas dataframe.
        '''
        # Sharon's code to make our data into the data loader setup
        # x_train_tensor = torch.tensor(X.values).type(torch.FloatTensor)
        x_train_tensor = torch.tensor(X).type(torch.FloatTensor)
        y_train_tensor = torch.tensor(y.values).type(torch.FloatTensor)

        print('x train tensor: ', x_train_tensor)
        print('y train tensor: ', y_train_tensor)

        train_dataset = []
        for i in range(len(x_train_tensor)):
            train_dataset.append([x_train_tensor[i], y_train_tensor[i]])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_s, shuffle=True)
        # End Sharon's code

        self.model.train()

        for epoch in (range(self.num_epochs)):
            # data is a tensor here
            for batch_idx, (data, target) in enumerate(train_loader):
                # Erase accumulated gradients
                optimizer.zero_grad()

                # Forward pass
                output = self.model(data)

                # Calculate loss
                loss = loss_fn(output, target)

                # Backward pass
                loss.backward()

                # Weight update
                optimizer.step()

            # Track loss each epoch
            print('Train Epoch: %d  Loss: %.4f' % (epoch + 1, loss.item()))


    def predict(self, X):
        # put model in evaluation mode
        self.model.eval()

        X = (X - self.mean_val) / self.std_dev
        # x_val_tensor = torch.tensor(X.values).type(torch.FloatTensor)
        x_val_tensor = torch.tensor(X).type(torch.FloatTensor)
        y_pred = self.model(x_val_tensor)
        return np.asarray(y_pred.detach().numpy())
