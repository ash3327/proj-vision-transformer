"""
Formulate All Your Network Models Here
"""

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

'''
    ResNet
'''
class _ResBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out
    
class _BigResBlock(nn.Module):

    def __init__(self, inchannel, midchannel, outchannel, stride=1):
        super(_BigResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel,midchannel,kernel_size=1,stride=1,padding=0,bias=False),
            
            nn.BatchNorm2d(midchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(midchannel,midchannel,kernel_size=3,stride=stride,padding=1,bias=False),
            
            nn.BatchNorm2d(midchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(midchannel,outchannel,kernel_size=1,stride=1,padding=0,bias=False),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(inchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)

        return out
    
class ResNet18(nn.Module):

    def __init__(self, ResBlock=_ResBlock, num_classes=10, depth_multiple:int=2):
        super(ResNet18, self).__init__()

        assert depth_multiple >= 1

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU()
        )
        self.lay1 = self.make_lay(ResBlock, 64 *depth_multiple, 2, stride=2)
        self.lay2 = self.make_lay(ResBlock, 128*depth_multiple, 2, stride=2)
        self.lay3 = self.make_lay(ResBlock, 256*depth_multiple, 2, stride=2)
        self.lay4 = self.make_lay(ResBlock, 512*depth_multiple, 2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512*depth_multiple, num_classes)
        
        
    def make_lay(self, block=_ResBlock, channels=64, num_blocks=2, stride=1):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lay1(out)
        out = self.lay2(out)
        out = self.lay3(out)
        out = self.lay4(out)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)

        return out
    
class ResNet34(nn.Module):

    def __init__(self, ResBlock=_ResBlock, in_channel=3, num_classes=10, first_lay_kernel_size=5, first_lay_stride=1, first_lay_padding=2):
        super(ResNet34, self).__init__()

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, self.inchannel, kernel_size=first_lay_kernel_size, 
                      stride=first_lay_stride, padding=first_lay_padding, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU()
        )
        self.lay1 = self.make_lay(ResBlock, 64, 3, stride=2)
        self.lay2 = self.make_lay(ResBlock, 128, 4, stride=2)
        self.lay3 = self.make_lay(ResBlock, 256, 6, stride=2)
        self.lay4 = self.make_lay(ResBlock, 512, 3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
        
        
    def make_lay(self, block=_ResBlock, channels=64, num_blocks=2, stride=1):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lay1(out)
        out = self.lay2(out)
        out = self.lay3(out)
        out = self.lay4(out)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)

        return out
    
    
class ResNet50(nn.Module):

    def __init__(self, ResBlock=_BigResBlock, num_classes=1000):
        super(ResNet50, self).__init__()

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU()
        )
        self.lay1 = self.make_lay(ResBlock, 64, 3, stride=2)
        self.lay2 = self.make_lay(ResBlock, 128, 4, stride=2)
        self.lay3 = self.make_lay(ResBlock, 256, 23, stride=2)
        self.lay4 = self.make_lay(ResBlock, 512, 3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(self.inchannel, num_classes)
        
        
    def make_lay(self, block=_BigResBlock, channels=64, num_blocks=2, stride=1):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, channels*4, stride))
            self.inchannel = channels*4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lay1(out)
        out = self.lay2(out)
        out = self.lay3(out)
        out = self.lay4(out)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)

        return out
    
'''
    ViT
'''
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
    
    