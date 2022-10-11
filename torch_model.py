from torch.utils.data import Dataset
import torch
import copy


class SpaceData(Dataset):
    def __init__(self, X, y):

        X = copy.deepcopy(X).to_numpy(dtype=float)
        self.X = torch.tensor(X, dtype=torch.float)

        y = copy.deepcopy(y).to_numpy(dtype=float)
        self.y = torch.reshape(torch.tensor(y, dtype=torch.float), (-1, 1))

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(37, 75),
            torch.nn.ReLU(),
            torch.nn.Linear(75, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, input):
        return self.model(input=input)


custom_model = CustomModel()

loss_model = torch.nn.BCELoss()

optimiser = torch.optim.SGD(custom_model.parameters(), lr=10 ** (-4))


def loss_calulation(dataloader, custom_model, loss_model, no_grad=False):

    for (sample, label) in dataloader:

        if not no_grad:
            prediction = custom_model(sample)
            return loss_model(prediction, label)
        else:
            with torch.no_grad():
                prediction = custom_model(sample)
                return loss_model(prediction, label)


def accuracy_calulation(dataloader, custom_model):

    for (sample, label) in dataloader:
        with torch.no_grad():
            prediction = custom_model(sample)
            return (prediction.round() == label).sum() / len(sample)


def optimise(optimiser, loss):
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
