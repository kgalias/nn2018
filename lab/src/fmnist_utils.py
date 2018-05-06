"""
Utility functions reducing boilerplate code in the laboratory notebooks.
"""

import numpy as np
from numpy import linalg as LA
import tqdm
import json

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch import optim

from keras.datasets import fashion_mnist, mnist
from keras.utils import np_utils

import matplotlib.pylab as plt

def get_data(N=1000, which="fmnist"):
    # Get FashionMNIST (see 1b_FMNIST.ipynb for data exploration)
    if which == "fmnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif which == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        raise NotImplementedError()

    # Logistic regression needs 2D data
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # 0-1 normalization
    x_train = x_train / 255.
    x_test = x_test / 255.

    # Convert to Torch Tensor. Just to avoid boilerplate code later
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    # Use only first 1k examples. Just for notebook to run faster
    x_train, y_train = x_train[0:1000], y_train[0:1000]
    x_test, y_test = x_test[0:1000], y_test[0:1000]
    
    return (x_train, y_train), (x_test, y_test)

def build_conv(input_dim, output_dim, n_filters=16, hidden_dims=[128]):
    model = torch.nn.Sequential()
    previous_dim = input_dim
    
    # Convolution part
    model.add("conv2d", torch.nn.Conv2d(1, n_filters, kernel_size=5, padding=2))
    
    # Classifier part
    for id, D in enumerate(hidden_dims):
        model.add_module("linear_{}".format(id), torch.nn.Linear(previous_dim, D, bias=True))
        model.add_module("nonlinearity_{}".format(id), torch.nn.ReLU())
        previous_dim = D
    model.add_module("final_layer", torch.nn.Linear(D, output_dim, bias=True))
    return model

def build_mlp(input_dim, output_dim, hidden_dims=[512]):
    model = torch.nn.Sequential()
    previous_dim = input_dim
    for id, D in enumerate(hidden_dims):
        model.add_module("linear_{}".format(id), torch.nn.Linear(previous_dim, D, bias=True))
        model.add_module("nonlinearity_{}".format(id), torch.nn.ReLU())
        previous_dim = D
    model.add_module("final_layer", torch.nn.Linear(D, output_dim, bias=True))
    return model

def build_linreg(input_dim, output_dim, hidden_dim=512):
    model = torch.nn.Sequential()
    model.add_module("linear_2", torch.nn.Linear(input_dim, output_dim, bias=True))
    return model

def step(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        x = x.cuda()
        y = y.cuda()
    
    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward(retain_graph=True)

    # Update parameters
    optimizer.step()

    return output.data[0]

def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        x = x.cuda()
        
    model.training = False
    output = model.forward(x).cpu().data.numpy()
    model.training = True

    if output.shape[1] >= 2:
        return output.argmax(axis=1)
    else:
        return (output > 0.5).reshape(-1,)

def train(model, loss, optim, 
          x_train, y_train, x_test, y_test, batch_size=100, n_epochs=10):
    """
    Trains given model on the FashionMNIST dataset.
    
    Returns
    -------
    history: dict
        History containing 'acc' and 'test_acc' keys.
    """
    torch.manual_seed(42) 
    n_examples, n_features = x_train.size()
    history = {"acc": [], "test_acc": []}
    for i in tqdm.tqdm(range(n_epochs), total=n_epochs):
        
        # Ugly way to shuffle dataset
        ids = np.random.choice(len(x_train), len(x_train), replace=False)
        x_train = torch.from_numpy(x_train.numpy()[ids])
        y_train = torch.from_numpy(y_train.numpy()[ids])
        
        cost = 0.
        num_batches = n_examples // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += step(model, loss, optim, x_train[start:end], y_train[start:end])
        
        predY = predict(model, x_test)
        test_acc = np.mean(predY == y_test.numpy())
        history['test_acc'].append(test_acc)
 
        # Usually it is computed from per batch averages, but I compute
        # here using the whole train set to reduce level of noise in the learning curves
        predY = predict(model, x_train)
        train_acc = np.mean(predY == y_train.numpy())
        history['acc'].append(train_acc)
        
    return history