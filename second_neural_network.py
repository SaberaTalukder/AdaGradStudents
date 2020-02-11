import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from orderbookmodel import OrderbookModel

class first_neural_network(OrderbookModel):

    def __init__(self):
        # number of input parameters is 14
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14, 20),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(20, 1)
        )

    def fit(self, X, y):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=60, shuffle=True)
        self.model.train()



        pass

    def predict(self, X, y):
        pass
