import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as F

from orderbookmodel import OrderBookModel

class second_neural_network(OrderBookModel):

    def __init__(self):
        # number of input parameters is 14
        self.model = nn.Sequential(
            nn.Linear(39, 80),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()
        self.learning_rate = 1e-3
        self.batch_s =1 
        self.num_epochs = 20



    def fit(self, X, y):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = self.loss_fn

        x_train_tensor = torch.tensor(X.values).type(torch.FloatTensor)
        y_train_tensor = torch.tensor(y.values).type(torch.FloatTensor)

        train_dataset = []
        for i in range(len(x_train_tensor)):
           train_dataset.append([x_train_tensor[i], y_train_tensor[i]])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_s, shuffle=True)
        self.model.train()
        for epoch in range(self.num_epochs):
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
            print('Train Epoch: %d  Loss: %.4f' % (epoch + 1,  loss.item()))


    def predict(self, X):
        # put model in evaluation mode
        self.model.eval()
        x_val_tensor = torch.tensor(X.values).type(torch.FloatTensor)
        print('type of x_val_tensor:')
        print(type(x_val_tensor))
        
        y_pred = self.model(x_val_tensor)
        print('type of y_pred:')
        print(type(y_pred))
        y_pred = y_pred.detach().numpy()
        print('type of y_pred:')
        print(type(y_pred))
        
        #y_pred = pd.DataFrame(data=y_pred.flatten())
        print('type of y_pred:')
        print(type(y_pred))
        print(y_pred)
        return y_pred

