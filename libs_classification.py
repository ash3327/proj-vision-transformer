import os
import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

def set_optimizers(model:nn.Module, loss_fn=nn.CrossEntropyLoss, optimizer=torch.optim.SGD, lr=1e-1, decay=None, **kwargs):
    # Loss function
    loss_fn = loss_fn()

    # SGD Optimizer
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = decay(optimizer, **kwargs) if decay != None else None
    return loss_fn, optimizer, scheduler

# Train function
def train(dataloader:DataLoader, model:nn.Module, loss_fn, optimizer, scheduler=None, device='cpu'):
    size = len(dataloader.dataset)
    
    # Turn on training mode
    model.to(device)
    model.train()
    train_loss, correct = 0, 0
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        
        # print(X.shape, y.shape)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss
        train_loss += loss.item()
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    
    train_loss /= len(dataloader)
    correct /= size
    lr = optimizer.param_groups[0]['lr']
    if scheduler != None:
        scheduler.step()
    
    print(f" Train accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f}, lr: {lr}")
    return train_loss, correct, lr

# Test function
def test(dataloader:DataLoader, model:nn.Module, loss_fn, device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    # Turn on evalution mode
    model.eval()
    test_loss, correct = 0, 0
    
    # Turn off gradient descent
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            # record loss
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    
    print(f" Test accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss, correct