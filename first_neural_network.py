import torch
import torch.nn as nn

from orderbookmodel import OrderbookModel

class first_neural_network(OrderbookModel):

    def __init__(self):
        # number of input parameters is 14
        self.model = nn.Sequential(
            nn.Flatten(),

            nn.Linear(14, 100),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(10, 1)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = 1e-3
        self.batch_s = 32
        self.num_epochs = 10


    def fit(self, X, y):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = self.loss_fn

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_s, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_s, shuffle=True)

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


    def predict(self, X, y):
        self.model.eval()
        loss_fn = self.loss_fn

        test_loss = 0
        correct = 0

        # Turning off automatic differentiation
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += loss_fn(output, target).item()  # Sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max class score
                correct += pred.eq(target.view_as(pred)).sum().item()  # calculates accuracy for classification

        test_loss /= len(test_loader.dataset)

        print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f)' %
              (test_loss, correct, len(test_loader.dataset),
               100. * correct / len(test_loader.dataset)))