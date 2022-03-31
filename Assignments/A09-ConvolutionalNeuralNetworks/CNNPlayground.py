# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Tasks" data-toc-modified-id="Tasks-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Tasks</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Dataset-Utility" data-toc-modified-id="Dataset-Utility-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Dataset Utility</a></span></li><li><span><a href="#Neural-Network-Utility" data-toc-modified-id="Neural-Network-Utility-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Neural Network Utility</a></span></li><li><span><a href="#Training-Utility" data-toc-modified-id="Training-Utility-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Training Utility</a></span></li><li><span><a href="#Validation-Utility" data-toc-modified-id="Validation-Utility-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Validation Utility</a></span></li><li><span><a href="#Loss-Plotting-Utility" data-toc-modified-id="Loss-Plotting-Utility-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Loss Plotting Utility</a></span></li><li><span><a href="#Data-Loading" data-toc-modified-id="Data-Loading-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Data Loading</a></span></li><li><span><a href="#Model-Creation" data-toc-modified-id="Model-Creation-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Model Creation</a></span></li><li><span><a href="#Training-and-Analysis" data-toc-modified-id="Training-and-Analysis-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Training and Analysis</a></span></li></ul></div>

# %% [markdown]
# ## Tasks
#
# 1. **Before you start**: open [gradescope](https://www.gradescope.com/) and respond with your initial predictions.
#
# 1. In this notebook we'll be using a new (to us) dataset, and you'll compare the performance of several models. Your end goal is to acheive the highest acuracy you have the patience to attain.
#
# 1. Train a ten-neuron network without any hidden layers or activations. The notebook is already setup to do so, you need only run all cells.
#
# 1. Train a fully connected neural network with multiple hidden layers. You'll only need to change the `neurons_per_hidden_layer` variable.
#
# 1. Create and train a convolutional neural network. A good place to start is [this tutorial form PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network). The most difficult part is getting the output of the convolutional layers to match the input of the fully connected (linear) layers. Your best tools include: (1) error messages talking about shapes, the call to `summary(model)`, and (3) hand computing the shape of each layer's output.
#
# 1. Try including [dropout](https://pytorch.org/docs/stable/nn.html#dropout-layers) and/or [BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d) layers.
#
# 1. Tune the Adam optimizer (pass in some parameters).
#
# 1. Try one of the models [provided by torchvision](https://pytorch.org/vision/stable/models.html). For example:
#
# ~~~python
# from torchvision.models import resnet18
# model = resnet18(num_classes=num_classes).to(device)
# ~~~
#
# This notebook contains several "TODO" comments. Search these out when trying to tune hyperparameters.

# %% [markdown]
# ## Imports

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from torchsummary import summary

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.utils import make_grid

from fastprogress.fastprogress import master_bar, progress_bar

import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(context="talk")


# %% [markdown]
# ## Dataset Utility

# %%
def get_cifar10_data_loaders(path, batch_size, valid_batch_size=0):

    # Data specific transforms
    data_std = (0.2470, 0.2435, 0.2616)
    data_mean = (0.4914, 0.4822, 0.4465)
    xforms = Compose([ToTensor(), Normalize(data_mean, data_std)])

    # Training dataset and loader
    train_dataset = CIFAR10(root=path, train=True, download=True, transform=xforms)

    # Set the batch size to N if batch_size is 0
    tbs = len(train_dataset) if batch_size == 0 else batch_size
    train_loader = DataLoader(train_dataset, batch_size=tbs, shuffle=True)

    valid_dataset = CIFAR10(root=path, train=False, download=True, transform=xforms)

    # Set the batch size to N if batch_size is 0
    vbs = len(valid_dataset) if valid_batch_size == 0 else valid_batch_size
    valid_loader = DataLoader(valid_dataset, batch_size=vbs, shuffle=True)

    return train_loader, valid_loader


# %% [markdown]
# ## Neural Network Utility

# %%
class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(NeuralNetwork, self).__init__()

        # The first "layer" just rearranges an image into a column vector
        first_layer = nn.Flatten()

        # The hidden layers include:
        # 1. a linear component (computing Z) and
        # 2. a non-linear comonent (computing A)
        hidden_layers = [
            nn.Sequential(nn.Linear(nlminus1, nl), nn.ReLU())
            for nl, nlminus1 in zip(layer_sizes[1:-1], layer_sizes)
        ]

        # The output layer must be Linear without an activation. See:
        #   https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        # Group all layers into the sequential container
        all_layers = [first_layer, *hidden_layers, output_layer]
        self.layers = nn.Sequential(*all_layers)

    def forward(self, X):
        return self.layers(X)


# %% [markdown]
# ## Training Utility

# %%
def train_one_epoch(mb, loader, device, model, criterion, optimizer):

    model.train()

    losses = []

    num_batches = len(loader)
    dataiterator = iter(loader)

    for batch in progress_bar(range(num_batches), parent=mb):

        mb.child.comment = f"Training"

        # Grab the batch of data and send it to the correct device
        X, Y = next(dataiterator)
        X, Y = X.to(device), Y.to(device)

        # Compute the output
        output = model(X)

        # Compute loss
        loss = criterion(output, Y)
        losses.append(loss.item())

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses


# %% [markdown]
# ## Validation Utility

# %%
def validate(mb, loader, device, model, criterion):

    model.eval()

    losses = []
    num_correct = 0

    num_classes = len(valid_loader.dataset.classes)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    N = len(loader.dataset)
    num_batches = len(loader)
    dataiterator = iter(loader)

    with torch.no_grad():

        batches = range(num_batches)
        batches = progress_bar(batches, parent=mb) if mb else batches
        for batch in batches:

            if mb:
                mb.child.comment = f"Validation"

            # Grab the batch of data and send it to the correct device
            X, Y = next(dataiterator)
            X, Y = X.to(device), Y.to(device)

            output = model(X)

            loss = criterion(output, Y)
            losses.append(loss.item())

            # Convert network output into predictions (one-hot -> number)
            predictions = output.argmax(dim=1)

            # Sum up total number that were correct
            comparisons = predictions == Y
            num_correct += comparisons.type(torch.float).sum().item()

            # Sum up number of correct per class
            for result, clss in zip(comparisons, Y):
                class_correct[clss] += result.item()
                class_total[clss] += 1

    accuracy = 100 * (num_correct / N)
    accuracies = {
        clss: 100 * class_correct[clss] / class_total[clss]
        for clss in range(num_classes)
    }

    return losses, accuracy, accuracies


# %% [markdown]
# ## Loss Plotting Utility

# %%
def update_plots(mb, train_losses, valid_losses, epoch, num_epochs):

    # Update plot data
    max_loss = max(max(train_losses), max(valid_losses))
    min_loss = min(min(train_losses), min(valid_losses))

    x_margin = 0.2
    x_bounds = [0 - x_margin, num_epochs + x_margin]

    y_margin = 0.1
    y_bounds = [min_loss - y_margin, max_loss + y_margin]

    train_xaxis = torch.linspace(0, epoch + 1, len(train_losses))
    valid_xaxis = torch.linspace(0, epoch + 1, len(valid_losses))
    graph_data = [[train_xaxis, train_losses], [valid_xaxis, valid_losses]]

    mb.update_graph(graph_data, x_bounds, y_bounds)


# %% [markdown]
# ## Data Loading

# %%
# TODO: tune the training batch size
train_batch_size = 128

# Let's use some shared space for the data (so that we don't have copies
# sitting around everywhere)
data_path = "/raid/cs152/cache/pytorch/data"

# Use the GPUs if they are available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device.")

valid_batch_size = 5000
train_loader, valid_loader = get_cifar10_data_loaders(
    data_path, train_batch_size, valid_batch_size
)

# Input and output sizes depend on data
num_features = torch.Size(train_loader.dataset.data.shape[1:]).numel()
class_names = train_loader.dataset.classes
num_classes = len(class_names)

print(class_names)

# %%
# Grab a bunch of images and change the range to [0, 1]
nprint = 64
images = torch.tensor(train_loader.dataset.data[:nprint] / 255)
targets = train_loader.dataset.targets[:nprint]
labels = [f"{class_names[target]:>10}" for target in targets]

# Create a grid of the images (make_grid expects (BxCxHxW))
image_grid = make_grid(images.permute(0, 3, 1, 2))

_, ax = plt.subplots(figsize=(16, 16))
ax.imshow(image_grid.permute(1, 2, 0))
ax.grid(None)

images_per_row = int(nprint ** 0.5)
for row in range(images_per_row):
    start_index = row * images_per_row
    print(" ".join(labels[start_index : start_index + images_per_row]))


# %% [markdown]
# ## Model Creation

# %%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # TODO: create layers here
        
        # Early CNNs had the following structure:
        #    X -> [[Conv2d -> ReLU] x N -> MaxPool2d] x M 
        #      -> [Linear -> ReLU] x K -> Linear
        #   Where
        #     0 ≤ N ≤ 3
        #     0 ≤ M ≤ 3
        #     0 ≤ K < 3
        #
        # The "[[Conv2d -> ReLU] x N -> MaxPool2d] x M" part extracts
        # useful features, and the "[Linear -> ReLU] x K -> Linear" part
        # performs the classification.

    def forward(self, X):
        
        # TODO: implement forward pass here
        pass


# %%
# TODO: try out different network widths and depths
neurons_per_hidden_layer = []
layer_sizes = [num_features, *neurons_per_hidden_layer, num_classes]
model = NeuralNetwork(layer_sizes).to(device)

# TODO: complete the CNN class in the cell above this one

# TODO: use an off-the-shell model from PyTorch
from torchvision.models import resnet18
model = resnet18(num_classes=num_classes).to(device)

summary(model)

# TODO: try out different Adam hyperparameters
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# ## Training and Analysis

# %%
# TODO: tune the number of epochs
num_epochs = 3

train_losses = []
valid_losses = []
accuracies = []

# A master bar for fancy output progress
mb = master_bar(range(num_epochs))
mb.names = ["Train Loss", "Valid Loss"]
mb.main_bar.comment = f"Epochs"

# Loss and accuracy prior to training
vl, accuracy, _ = validate(None, valid_loader, device, model, criterion)
valid_losses.extend(vl)
accuracies.append(accuracy)

for epoch in mb:

    tl = train_one_epoch(mb, train_loader, device, model, criterion, optimizer)
    train_losses.extend(tl)

    vl, accuracy, acc_by_class = validate(mb, valid_loader, device, model, criterion)
    valid_losses.extend(vl)
    accuracies.append(accuracy)

    update_plots(mb, train_losses, valid_losses, epoch, num_epochs)

# %%
plt.plot(accuracies, '--o')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.xticks(range(num_epochs+1))
plt.ylim([0, 100])

max_name_len = max(len(name) for name in class_names)

print("Accuracy per class")
for clss in acc_by_class:
    class_name = class_names[clss]
    class_accuracy = acc_by_class[clss]
    print(f"  {class_name:>{max_name_len+2}}: {class_accuracy:.1f}%")

# %%
