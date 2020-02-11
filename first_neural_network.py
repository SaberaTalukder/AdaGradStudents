import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from orderbookmodel import OrderbookModel

class first_neural_network(OrderbookModel):

    def __init__(self):
        # number of input parameters is 15
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(15, 20),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(20, 10)
        )

    def fit(self, X, y):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        pass

    def predict(self, X, y):
        pass