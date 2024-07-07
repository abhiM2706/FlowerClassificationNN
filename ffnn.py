import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split


# A typical Feed-Forward Neural Network class with one hidden layer
class Iris(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Iris, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
