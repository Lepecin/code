from torch.utils.data import Dataset
import torch
import copy
from pandas import DataFrame, Series
from typing import Tuple
from torch import Tensor
from numpy import ndarray as Array


class SpaceData(Dataset):
    """
    Class for creating dataset from space data that is
    compatible with pytorch infrastructure.
    """

    features: "Tensor"
    labels: "Tensor"

    def __init__(self: "SpaceData", X: "DataFrame", y: "Series") -> "None":

        # Convert features and labels to numpy arrays
        X_num: "Array" = copy.deepcopy(X).to_numpy(dtype=float)
        y_num: "Array" = copy.deepcopy(y).to_numpy(dtype=float)

        # Conver numpy arrays to torch tensors
        self.features = torch.tensor(X_num, dtype=torch.float)
        self.labels = torch.reshape(torch.tensor(y_num, dtype=torch.float), (-1, 1))

    def __len__(self: "SpaceData") -> "int":

        # For getting dataset length
        return len(self.features)

    def __getitem__(self: "SpaceData", index: "int") -> "Tuple[Tensor, Tensor]":

        # For retreiving individual data rows
        return self.features[index], self.labels[index]


class CustomModel(torch.nn.Module):
    """
    Custom pytorch model used for
    binary classification of space data.
    """

    model: "torch.nn.Sequential"

    def __init__(self: "CustomModel") -> "None":
        super().__init__()

        # Create sequential model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(37, 75),
            torch.nn.ReLU(),
            torch.nn.Linear(75, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self: "CustomModel", input: "Tensor") -> "Tensor":
        return self.model(input=input)


def loss_calulation(sample, label, custom_model, loss_model):
    prediction = custom_model(sample)
    return loss_model(prediction, label)


def accuracy_calulation(sample, label, custom_model):
    prediction = custom_model(sample)
    return (prediction.round() == label).sum() / len(sample)


def optimise(optimiser, loss):
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


def trainer(
    train_dataloader,
    val_dataloader,
    custom_model,
    loss_model,
    optimiser,
    iterations,
    print_stats=False,
):
    for current in range(iterations):
        for (X, y) in train_dataloader:

            train_loss = loss_calulation(
                sample=X, label=y, custom_model=custom_model, loss_model=loss_model
            )

            optimise(optimiser=optimiser, loss=train_loss)

        for (X, y) in val_dataloader:
            with torch.no_grad():
                val_loss = loss_calulation(
                    sample=X, label=y, custom_model=custom_model, loss_model=loss_model
                )

                val_acc = accuracy_calulation(
                    sample=X, label=y, custom_model=custom_model
                )

        if (current + 1 % 10 == 0) and print_stats:
            print(
                f"loss: {val_loss.item():>7f}, accuracy: {val_acc.item():>7f}  [{current+1:>5d}/{iterations:>5d}]"
            )
