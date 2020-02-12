import numpy as np
import torch
import torch.nn as nn

from orderbookmodel import OrderBookModel

class first_neural_network(OrderBookModel):

    def __init__(self):
        # number of input parameters is 14
        self.model = nn.Sequential(
            nn.Linear(39, 100),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()
        self.learning_rate = 1e-3
        self.batch_s = 32
        self.num_epochs = 10


    def fit(self, X, y):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = self.loss_fn

        # Sharon's code to make our data into the data loader setup
        x_train_tensor = torch.tensor(X.values).type(torch.FloatTensor)
        y_train_tensor = torch.tensor(y.values).type(torch.FloatTensor)

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

                # print('target shape: ', target.shape)
                # print('output shape: ', output.shape)
                # print('target: ', target)
                # print('output: ', output)
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
        x_val_tensor = torch.tensor(X.values).type(torch.FloatTensor)
        y_pred = self.model(x_val_tensor)
        return np.asarray(y_pred.detach().numpy())
