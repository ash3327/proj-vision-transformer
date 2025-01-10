import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from models import SegVit

def imshow(X, range_=range(9), labels:list[str]=None, colorbar=True, 
           full_color=False):
    if labels == None:
        labels = [None for i in range_]

    if len(range_) > len(X):
        range_ = range(len(X))

    #print(np.min(X),np.max(X))
    for i, j in enumerate(range_):
        img = X[j]
        #img = img / 2 + 0.5     # unnormalize
        
        plt.subplot(331 + i)
        
        kwargs = dict()
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)

        if full_color:
            img = (img-np.min(img))/(np.max(img)-np.min(img))
        else:
            kwargs = {'vmin':0, 'vmax':34}

        A = plt.imshow(np.transpose(img, (1, 2, 0)), **kwargs)
        if colorbar:
            plt.colorbar(A)
        else:
            plt.axis('off')
        plt.title(labels[i])
    plt.tight_layout()
    plt.show()

def imshow_test(model, test_loader, device, show_reference=True, show_original=False, softmax=False, range=range(3), **kwargs):
    model.eval()
    test_loss, correct = 0, 0
    model.to(device=device)

    #classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    # Turn off gradient descent
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if isinstance(model, SegVit):
                pred = pred['pred']
            if not softmax:
                pred = torch.sigmoid(pred)
                pred = (pred>.5).float()
            else:
                pred = nn.functional.softmax(pred, dim=1)
                pred = torch.argmax(pred, dim=1)
                y = nn.functional.softmax(y, dim=1)
                y = torch.argmax(y, dim=1)

            if show_original:
                imshow(X.cpu().numpy(), range_=range, colorbar=False, full_color=True)
            print(torch.max(pred))
            imshow(pred.cpu().numpy(), range_=range, **kwargs)
            if show_reference:
                imshow(y.cpu().numpy(), range_=range, **kwargs)
            
            break