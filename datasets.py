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

def get_transforms(train_transform_stack, test_transform_stack, to_tensor=True):
    
    if to_tensor:
        train_transform_stack += [transforms.ToTensor()]
        test_transform_stack += [transforms.ToTensor()]
    # Train transformation
    train_transform = transforms.Compose(
        train_transform_stack
    )
    # Test transformation
    test_transform = transforms.Compose(
        test_transform_stack
    )
    return train_transform, test_transform

def one_hot(x, n_classes):
    x = np.array(x)
    #print(x.shape,  torch.tensor(x, dtype=torch.int64))
    return torch.zeros((n_classes,*x.shape), dtype=torch.float).scatter_(0, torch.tensor(x, dtype=torch.int64).unsqueeze(0), value=1)
    #return F.one_hot(torch.tensor(np.array(x), dtype=torch.int64).long(), num_classes=n_classes)

def get_transforms2(tr_stack, te_stack, 
                    train_transform_stack, test_transform_stack,
                    n_classes, to_tensor=True):
    
    if to_tensor:
        tr_stack = [lambda x:one_hot(x,n_classes)]+tr_stack
        train_transform_stack = [transforms.ToTensor()]+train_transform_stack
        te_stack = [lambda x:one_hot(x,n_classes)]+te_stack
        test_transform_stack = [transforms.ToTensor()]+test_transform_stack
    
    
    train_transform = transforms.Compose(train_transform_stack)#+[transforms.Normalize(mean=[0.],std=[1.]),]
    test_transform = transforms.Compose(test_transform_stack)


    
    trtarg = transforms.Compose(tr_stack)
    tetarg = transforms.Compose(te_stack)
    return train_transform, trtarg, test_transform, tetarg

def fetch_dataset(dataset=datasets.CIFAR10, train_transform=None, test_transform=None):
    """
    EXAMPLE: datasets.MNIST
    """
    # download dataset
    train_data = dataset(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_data = dataset(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_data, test_data

def fetch_dataset2(dataset=datasets.CIFAR10, 
                   train_transform=None, train_transform_targ=None,
                   test_transform=None, test_transform_targ=None):
    """
    EXAMPLE: datasets.MNIST
    """
    # download dataset
    train_data = dataset(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
        target_transform=train_transform_targ
    )

    test_data = dataset(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
        target_transform=test_transform_targ
    )
    return train_data, test_data


from torch.utils.data import Dataset
from torch import random

class CityscapeDataset(Dataset):
    
    def __init__(self, dataset:datasets.Cityscapes, img_transform=None, targ_transform=None):
        self.dataset = dataset
        self.img_transform = img_transform
        self.targ_transform = targ_transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, mask = self.dataset.__getitem__(index)
        
        #state = random.get_rng_state()
        seed = torch.randint(0, 2**32, ())
        random.manual_seed(seed)
        img = self.img_transform(img)
        random.manual_seed(seed)
        mask = self.targ_transform(mask)
        #random.set_rng_state(state)
        return img, mask

def get_dataloader(train_data, test_data, batch_size=256, **kwargs):
    # create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, **kwargs)
    return train_loader, test_loader