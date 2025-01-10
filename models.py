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
    U-Net
'''
class DoubleConv(nn.Module):

    def __init__(self, in_dim, out_dim, filter_size=3, stride=1, padding=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=filter_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=filter_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)
    
class ResBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()

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
    
class BigResBlock(nn.Module):

    def __init__(self, inchannel, midchannel, outchannel, stride=1):
        super(BigResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel,midchannel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(midchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(midchannel,midchannel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(midchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(midchannel,outchannel,kernel_size=1,stride=1,padding=0,bias=False),
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
    
class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, 
                 features=[64, 128, 256, 512],
                 block:nn.Module=DoubleConv):
        super(UNet, self).__init__()
        self.upconvs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(block(in_channels, feature))
            in_channels = feature
        
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(block(feature*2, feature))

        self.bottleneck = block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = list()
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.ups)):
            x = self.upconvs[i](x)
            skip_connection = skip_connections[i]
            if x.shape != skip_connection.shape[2:]:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i](concat_skip)
        
        return self.final_conv(x)
    
'''
    SegVit
'''
import os
class CheckPoint:
    def __init__(self, path, require_fetch:bool=True, temp_loc=None):
        self.path = path#'https://huggingface.co/Akide/SegViTv1/resolve/main/COCOstuff10k_shrunk_49.1.pth'
        self.require_fetch = require_fetch
        if temp_loc is None:
            temp_loc = os.path.join('temp_ckpt', os.path.basename(path))
            os.makedirs(os.path.dirname(temp_loc), exist_ok=True)
        self.temp_loc = temp_loc

    def load(self):
        if self.require_fetch:
            if os.path.exists(self.path):
                self.temp_loc = self.path
            else:
                import requests
                response = requests.get(self.path)
                print(f'Fetching from {self.path}...')
                with open(self.temp_loc, 'wb') as f:
                    f.write(response.content)
            state_dict = torch.load(self.temp_loc)
            print('Finished fetching.')
            return state_dict
        return torch.hub.load_state_dict_from_url(self.path)

    @classmethod
    def get_default(cls):
        return cls('https://huggingface.co/Akide/SegViTv1/resolve/main/COCOstuff10k_shrunk_49.1.pth')
    
    @classmethod
    def load_default(cls, file_name='COCOstuff10k_shrunk_49.1.pth'):
        path = os.path.join('temp_ckpt', file_name)
        try:
            return cls(path)
        except Exception:
            print(f'The checkpoint file {path} is not found. \nDownloading for you.')
            return cls.get_default()

# reference: segvit\tools\train.py  
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger, setup_multi_processes

from segvit.decode_heads import atm_head, tpn_atm_head
from segvit.losses import atm_loss
from segvit.backbone import vit_shrink

# github source: https://github.com/zbwxp/SegVit
# reference: https://blog.csdn.net/qq_43606119/article/details/130764322
class SegVit(nn.Module):

    def __init__(self, num_classes, checkpoint:CheckPoint=None, freeze_vit:bool=False):
        super(SegVit, self).__init__()
        cfg = self.__load_configs()

        # reference: https://blog.csdn.net/qq_43606119/article/details/130764322
        if checkpoint is not None and checkpoint.path is None:
            assert False, 'the bug is not fixed yet.'
            checkpoint = CheckPoint(cfg.get('checkpoint'))
        self.backbone = self.get_vit_shrink(cfg.get('model').get('backbone'), checkpoint=checkpoint)
        self.decoder = self.get_decode_head(cfg.get('model').get('decode_head'), num_classes=num_classes)
        if freeze_vit:
            for lay in self.backbone.layers[-1:]:
                for param in lay.parameters():
                    param.requires_grad = False
            return
            self.backbone.requires_grad_(False)
        return

        # reference: segvit\tools\train.py   
        model = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        model.init_weights()
        model = revert_sync_batchnorm(model)
        self.model = model

    def __load_configs(self):
        from mmcv.utils import Config

        path_config = 'segvit\configs\segvit\segvit_vit-l_jax_512x512_80k_cocostuff10k.py'
        path_config_shrink = 'segvit\configs\segvit\shrink\segvit_vit-l_jax_shrink8_tpnatm_coco.py'

        # reference: segvit\tools\train.py    
        cfg = Config.fromfile(path_config)
        cfg_shrink = Config.fromfile(path_config_shrink)
        cfg.merge_from_dict(cfg_shrink)
        return cfg
    
    def get_vit_shrink(self, cfg, checkpoint=None):
        backbone_cfg = {k: v for k, v in cfg.items() if k != 'type'}
        
        model = vit_shrink.vit_shrink(
            **backbone_cfg
        )
        if checkpoint:
            state_dict = checkpoint.load()
            model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def get_decode_head(self, cfg, num_classes):
        decode_head_cfg = {k: v for k, v in cfg.items() if k != 'type'}
        decode_head_cfg.update(dict(num_classes=num_classes))
        
        decode_head = tpn_atm_head.TPNATMHead(
            **decode_head_cfg
        )
        decode_head.init_weights()
        decode_head.atm.init_weights()
        return decode_head

    def forward(self, _x):
        # reference: https://blog.csdn.net/qq_43606119/article/details/130764322
        #print('x shape:',_x.shape)
        x = self.backbone(_x)
        #print(x)
        #print('hid shape:',[t.shape for t in x])
        out = self.decoder(x)
        
        # if self.training:
        #     #print(out['pred'].shape)
        #     return out['pred']
        if not self.training:
            return dict(pred=out)
        return out

class ATMLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input, target: torch.Tensor, model: SegVit=None) -> torch.Tensor:
        return torch.sum(list(model.decoder.losses(input, target).values()))
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

if __name__ == '__main__':
    from torchvision import models
    from torchsummary import summary

    torch.cuda.empty_cache()
    
    input_shape = (3,512,512)
    x = torch.randn(2,*input_shape)
    ckpt = CheckPoint.load_default()#CheckPoint.get_default()
    model = SegVit(34, checkpoint=ckpt, freeze_vit=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('device:', device)
    model.to(device)
    model.train()
    #model.eval()
    #with torch.no_grad(): # torch.no_grad() have no problem -> ie problems in storing gradients
    import time
    start = time.time()
    x = x.to(device)
    res = model(x)
    end = time.time()
    #print(res, res.shape)
    print(end-start, 's')
    
    #print(model)
    
    
    #model.eval()
    #summary(model, input_shape, batch_size=3)