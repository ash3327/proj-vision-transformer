import os
import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from models import ATMLoss, SegVit

# *Reusing my previous code

## Saving and Loading Models
def text_to_file(JOINED_PATH:str, contents, mode:str='w'):
    with open(JOINED_PATH, mode) as f:
        print(contents, file=f)

def save_model(model:nn.Module, PATH:str='models', FILENAME:str=None, extra_info:str=""):
    if FILENAME == None:
        import time
        FILENAME = f'model_{int(time.time())}.h5'

    import os
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    PATH = os.path.join(PATH, FILENAME)
    torch.save(model.state_dict(), PATH)

    text_to_file(PATH+'_details.txt', str(model)+'\n'+extra_info)
    return PATH

def save(TRAIN_ID, model, training_accuracy, training_losses, validation_accuracy, validation_losses, lrs):
    extra_info = f" Train accuracy: {(100*training_accuracy[-1]):>0.1f}%, Avg loss: {training_losses[-1]:>8f}, lr: {lrs[-1]}"
    if validation_accuracy is not None:
        extra_info += f"\n Test accuracy: {(100*validation_accuracy[-1]):>0.1f}%"
        if validation_losses is not None: 
            extra_info += f", Avg loss: {validation_losses[-1]:>8f}"
    return save_model(model=model, PATH=os.path.join('models',str(TRAIN_ID)), extra_info=extra_info)

def load_model(model:nn.Module, FILE_PATH:str, device='cpu'):
    if FILE_PATH is None:
        return
    model.load_state_dict(torch.load(FILE_PATH, map_location=torch.device(device)))

## Related to Training
def set_optimizers(model:nn.Module, loss_fn=nn.CrossEntropyLoss, optimizer=torch.optim.SGD, lr=1e-1, decay=None, **kwargs):
    # Loss function
    loss_fn = loss_fn()

    # SGD Optimizer
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = decay(optimizer, **kwargs) if decay != None else None
    return loss_fn, optimizer, scheduler

# Train function
def train(dataloader:DataLoader, model:nn.Module, loss_fn, optimizer, 
          scheduler=None, device='cpu', log:bool=True, train_lim=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    # Turn on training mode
    model.to(device)
    model.train()
    train_loss, pixels, correct, total_iou = 0, 0, 0, 0
    
    kwargs = dict() if train_lim is None else dict(total=train_lim)
    dt = tqdm(dataloader, **kwargs) if log else dataloader
    i = 0
    for X, y in dt:
        if train_lim is not None and i > train_lim:
            break
        else: 
            i+=1
        X, y = X.to(device), y.to(device).float()

        # Compute prediction error
        
        # print(X.shape, y.shape)
        pred = model(X)
        kwargs = dict(model=model) if isinstance(loss_fn, ATMLoss) else dict()
        loss = loss_fn(pred, y, **kwargs)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss
        train_loss += loss.item()
        if isinstance(model, SegVit):
            pred = pred['pred']

        # evaluate accuracy
        Y = torch.argmax(y, dim=1)
        preds = torch.argmax(pred, dim=1)
        
        correct += (preds == Y).sum().float()
        pixels += torch.numel(preds)

        preds = F.one_hot(preds, pred.size(1))
        ys = F.one_hot(Y, y.size(1))
        
        pred_flat = preds.view(preds.size(0), -1, preds.size(3))
        y_flat = ys.view(ys.size(0), -1, ys.size(3))

        intersect = torch.sum(pred_flat * y_flat, dim=1)
        union = torch.sum(pred_flat, dim=1) + torch.sum(y_flat, dim=1) - intersect
        cls_iou = intersect / union
        mean_iou = torch.mean(torch.nanmean(cls_iou, dim=1))
        total_iou += mean_iou
    
    train_loss /= num_batches
    correct /= pixels
    correct = correct.detach().cpu().numpy()
    total_iou /= num_batches
    total_iou = total_iou.detach().cpu().numpy()
    lr = optimizer.param_groups[0]['lr']
    if scheduler != None:
        scheduler.step() if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step(loss)
    
    if log:
        print(f" Train accuracy: {100*correct}%, Avg loss: {train_loss}, IOU: {100*total_iou} lr: {lr}")
    return train_loss, correct, total_iou, lr

# Test function
def test(dataloader:DataLoader, model:nn.Module, loss_fn, device='cpu', log:bool=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    # Turn on evalution mode
    model.to(device)
    model.eval()
    test_loss, pixels, correct, total_iou = 0, 0, 0, 0
    
    # Turn off gradient descent
    with torch.no_grad():
        dt = tqdm(dataloader) if log else dataloader
        for X, y in dt:
            X, y = X.to(device), y.to(device).float()
            pred = model(X)
            
            # record loss
            kwargs = dict(model=model) if isinstance(loss_fn, ATMLoss) else dict()
            test_loss += loss_fn(pred, y, **kwargs).item()
            if isinstance(model, SegVit):
                pred = pred['pred']
            
            # evaluate accuracy
            Y = torch.argmax(y, dim=1)
            preds = torch.argmax(pred, dim=1)
            
            correct += (preds == Y).sum().float()
            pixels += torch.numel(preds)

            preds = F.one_hot(preds, pred.size(1))
            ys = F.one_hot(Y, y.size(1))
            
            pred_flat = preds.view(preds.size(0), -1, preds.size(3))
            y_flat = ys.view(ys.size(0), -1, ys.size(3))
            intersect = torch.sum(pred_flat * y_flat, dim=1)
            union = torch.sum(pred_flat, dim=1) + torch.sum(y_flat, dim=1) - intersect
            cls_iou = intersect / union
            mean_iou = torch.mean(torch.nanmean(cls_iou, dim=1))
            total_iou += mean_iou
    
    test_loss /= num_batches
    correct /= pixels
    correct = correct.detach().cpu().numpy()
    total_iou /= num_batches
    total_iou = total_iou.detach().cpu().numpy()

    if log:
        print(f" Test accuracy: {100*correct}%, Avg loss: {test_loss}, IOU: {100*total_iou}")
    return test_loss, correct, total_iou

# Generating Output
def gen_output(dataloader:DataLoader, model:nn.Module, device='cpu'):    
    # Turn on evalution mode
    model.to(device)
    model.eval()

    names = list()
    predictions = list()
    
    # Turn off gradient descent
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y
            pred = model(X)
            p = pred.argmax(1).cpu().numpy()
            p = p.tolist()
            
            names.extend(y)
            predictions.extend(p)
            
    return names, predictions