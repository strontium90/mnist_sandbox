#!/usr/bin/env python3


# %% IMPORTS

import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, \
                                   RandomRotation, InterpolationMode
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import network
from train_val import train_loop, validation_loop, update_graphs


# %% HYPER-PARAMETERS : should also be able to provide as arguments to the module

epochs = 10
learning_rate = 1e-3
weight_decay = 0.001
mbatch_size = 32
mbatch_group = -1
num_workers = 8


# %% TARGET DEVICE

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device.")


# %% DATA LOADING

test_transform = Compose([ToTensor(),
                          Normalize((0.1307,), (0.3081,))])
                          
train_transform = Compose([RandomRotation([-20, 20],
                           InterpolationMode.BILINEAR),
                           test_transform])

train_set = datasets.MNIST(root='./data', train=True,
                           download=True, transform=train_transform)
test_set = datasets.MNIST(root='./data', train=False,
                          download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=mbatch_size,
                                          shuffle=True, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=mbatch_size,
                                         shuffle=False, num_workers=num_workers)
                                         
num_classes = len(train_set.classes)


# %% MODEL CREATION AND SUMMARY

net = network.Net()
net = net.to(device)
summary(net, input_size=(1, 1, 28, 28), col_names=["input_size", "output_size", "num_params"])


# %% LOSS FUNCTION AND OPTIMIZER

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)


# %% TENSORBOARD WRITER INSTANTIATION

timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
summary_writer = SummaryWriter("./runs/mnist_sandbox_{}".format(timestamp))


# %% TRAIN/VAL LOOP

training_time = 0

for t in range(epochs):
    print(f"EPOCH {t+1:4d}", 70*"-", flush=True)

    tic = time.time()
    train_loop(train_loader, net, criterion, optimizer, device)
    toc = time.time()
    training_time += (toc - tic)
    train_res = validation_loop(train_loader, net, criterion, num_classes, device)
    test_res = validation_loop(test_loader, net, criterion, num_classes, device)    
    update_graphs(summary_writer, t, train_res, test_res)

summary_writer.close()
print(f"Finished training for {epochs} epochs in {training_time} seconds.")

