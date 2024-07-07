import time

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split

from ffnn import Iris

# A neural network to do flower classification of the Iris species
# The data of the Iris flower consists of 4 inputs (sepal_length, sepal_width, petal_length, petal_width)
# This code utilizes pandas, numpy, and primarily PyTorch to build this neural network
# This is my first ever machine-learning project

# Loading the data and hardcoding all the strings in the data into numbers
data = pd.read_csv('iris.data')
data['class']=data['class'].replace('Iris-setosa',0)
data['class']=data['class'].replace('Iris-versicolor',1)
data['class']=data['class'].replace('Iris-virginica',2)

# Splitting the data into x and y
x = data.loc[:, data.columns != "class"]
y = data.loc[:, data.columns == "class"]

# Splitting data into training data and testing data because Iris data does not have testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
x_tensor_train = torch.tensor(x_train.to_numpy().tolist(), dtype=torch.float)
y_tensor_train = torch.tensor(y_train.to_numpy(), dtype=torch.float).long()
x_tensor_test = torch.tensor(x_test.to_numpy(), dtype=torch.float)
y_tensor_test = torch.tensor(y_test.to_numpy(), dtype=torch.float).long()
y_tensor_test=torch.flatten(y_tensor_test)
y_tensor_train=torch.flatten(y_tensor_train)

dataset_train = TensorDataset(x_tensor_train, y_tensor_train)
dataset_test = TensorDataset(x_tensor_test, y_tensor_test)
# With the training data create dataloaders
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)


# Call the FFNN class where the parameters are input size, output size, and size of hidden layer(s)
model = Iris(input_size=4, output_size=3, hidden_size=5)
# Loss Function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for i, train_data in enumerate(dataloader_train):
        optimizer.zero_grad()

        # dataloader_train data is in a tuple of inputs and labels of flowers
        inputs, labels = train_data

        # Make predictions for this batch
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        running_loss += loss.item()

        last_loss = running_loss / 1000  # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        tb_x = epoch_index * len(dataloader_train) + i + 1
        running_loss = 0.

    return last_loss


num_epochs = 1000
best_test_loss = 1_000_000.


for epoch in range(num_epochs):
    model.train(True)
    avg_loss = train(num_epochs)

    test_running_loss = 0.0
    if(num_epochs%25==0):
        model_path = 'model_{}_{}'.format(time.time(), num_epochs)
        torch.save(model.state_dict(), model_path)

model.eval()
test_running_loss = 0.0
correct = 0
total=0
with torch.no_grad():
    for i, test_data in enumerate(dataloader_test):
        inputs, labels = test_data
        outputs = model(inputs)
        test_loss = loss_fn(outputs, labels)
        test_running_loss += test_loss
        _, predicted_classes = torch.max(outputs, 1)
        accuracy = (predicted_classes == labels).float().sum().item() / labels.size(dim=0)
        correct += accuracy
        total+=1

avg_test_loss = test_running_loss / (i + 1)
print('LOSS train {} valid {}'.format(avg_loss, avg_test_loss))

num_epochs += 1
print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')