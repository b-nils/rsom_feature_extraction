import numpy as np
from math import floor, ceil

import torch
import torch.nn.functional as F
import torch.nn as nn

class DeepVesselNet(nn.Module):

    def __init__(self, 
            in_channels=2,
            channels = [2, 5, 10, 20, 50, 1],
            kernels = [3, 5, 5, 3, 1],
            depth = 5, 
            dropout=False, 
            groupnorm=False,
            use_vblock=False,
            vblock_layer=None,
            save_memory=False):
        
        super(DeepVesselNet, self).__init__()
        
        # SETTINGS
        self.in_channels = in_channels
        self.depth = depth
        self.dropout = dropout
        self.use_vblock = use_vblock
        self.save_memory = save_memory

        if use_vblock:
            assert vblock_layer is not None
            # don't try to use it in the last layer
            assert vblock_layer <= depth - 2
        
        self.vblock_layer = vblock_layer
       
        # generate dropout list for every layer
        if dropout:
            self.dropouts = [0, 0, 0.3, 0.3, 0]
        else:
            self.dropouts = [0] * depth

        # generate channels list for every layer
        self.channels = channels
        # override in_channels
        self.channels[0] = in_channels

        # generate kernel size list
        self.kernels = kernels
        
        self.groupnorm = groupnorm
        if groupnorm:
            self.groupnorms = [0] + [1]*(depth-2) + [0]
        else:
            self.groupnorms = [0]*(depth)

        assert len(self.dropouts) == depth
        assert len(self.channels) == depth + 1
        assert len(self.kernels) == depth
        assert len(self.groupnorms) == depth

        layers = []

        # TODO fix notation depth layers?
        
        # deep layers
        for i in range(depth-1):
            if use_vblock and i == vblock_layer:
                layers.append(V_Block(
                    self.channels[i],
                    self.channels[i+1],
                    self.kernels[i],
                    4))
            else:
                layers.append(DVN_Block(
                    self.channels[i],
                    self.channels[i+1],
                    self.kernels[i],
                    self.groupnorms[i],
                    self.dropouts[i]))
        # last layer
        layers.append(nn.Conv3d(self.channels[-2], self.channels[-1], self.kernels[-1])) 

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.training or not self.save_memory:
            return self.layers(x)
        else:
            # self.layers is nested nn.Sequential
            for i in range(len(self.layers)):
                # nn.Sequential is named block
                if hasattr(self.layers[i], 'block'):
                    for j in range(len(self.layers[i].block)):
                        x = self.layers[i].block[j](x)
                        torch.cuda.empty_cache()
                else:
                    x = self.layers[i](x)
                    torch.cuda.empty_cache()
            return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DVN_Block(nn.Module):

    def __init__(self, in_size, out_size, kernel_size, groupnorm, dropout):
        super(DVN_Block, self).__init__()
        
        block = []

        block.append(nn.Conv3d(in_size, out_size, kernel_size))
        block.append(nn.ReLU())
        if groupnorm:
            block.append(nn.GroupNorm(5, out_size))
        if dropout:
            block.append(nn.Dropout3d(dropout))
        
        self.block = nn.Sequential(*block)

    def forward(self, x):
        
        return self.block(x)

class V_Block(nn.Module):

    def __init__(self, in_size, out_size, kernel_size_conv, kernel_size_pool):
        super(V_Block, self).__init__()

        assert kernel_size_conv % 2
        
        self.ksp = kernel_size_pool
        
        straight = []
        straight.append(nn.Conv3d(in_size, floor(out_size/2), kernel_size_conv))
        straight.append(nn.ReLU())
        
        self.straight = nn.Sequential(*straight)
        padding = 0

        down = []
        down.append(nn.MaxPool3d(kernel_size_pool, padding=0))
        down.append(nn.Conv3d(in_size, ceil(out_size/2), kernel_size_conv))
        down.append(nn.ReLU())
        
        self.down = nn.Sequential(*down)

        up = []
        up.append(nn.Upsample(scale_factor=kernel_size_pool, mode='trilinear'))
        pad = (kernel_size_pool-1) * floor(kernel_size_conv/2)
        up.append(nn.ReplicationPad3d(pad))

        self.up = nn.Sequential(*up)

        self.cat = Cat()
    
    def forward(self, x):  
        bridge = self.straight(x)
        
        # pad to the next multiplier of kernel_size_pool
        pad = (floor(-x.shape[-1] % self.ksp / 2), 
               ceil(-x.shape[-1] % self.ksp / 2), 
               floor(-x.shape[-2] % self.ksp / 2), 
               ceil(-x.shape[-2] % self.ksp / 2), 
               floor(-x.shape[-3] % self.ksp / 2), 
               ceil(-x.shape[-3] % self.ksp / 2))

        x = nn.functional.pad(x, pad)

        x = self.down(x)
        x = self.up(x)
        x = nn.functional.pad(x, tuple(-el for el in pad))
        return self.cat(x, bridge)  

class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, x, bridge):
        return torch.cat((x, bridge), 1)

class Res_Block(nn.Module):

    def __init__(self, in_size, out_size, kernel_size, relu, groupnorm):
        super(Res_Block, self).__init__()
        # order is:
        # relu - bn - conv
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        
        block = []

        if relu:
            block.append(nn.ReLU())
        if groupnorm:
            block.append(nn.GroupNorm(5, in_size))
        
        block.append(nn.Conv3d(in_size, out_size, kernel_size))
        
        self.block = nn.Sequential( *block)

    def forward(self, x):
        z = nn.functional.pad(x, (-floor(self.kernel_size/2),) * 6) 
        
        reps = ceil(self.out_size/self.in_size)
        z = z.repeat((1, reps, 1, 1, 1))
        z = z[:, :self.out_size, ...]
        return z + self.block(x)

class ResVesselNet(nn.Module):
    def __init__(self, 
            in_channels=2,
            channels = [2, 5, 10, 20, 50, 1],
            kernels = [3, 5, 5, 3, 1],
            depth = 5, 
            groupnorm = False):
        
        super(ResVesselNet, self).__init__()
       
        # SETTINGS
        self.in_channels = in_channels
        self.depth = depth

        # generate channels list for every layer
        self.channels = channels
        # override in_channels
        self.channels[0] = in_channels

        # generate kernel size list
        self.kernels = kernels
        
        self.groupnorm = groupnorm
        if groupnorm:
            self.groupnorms = [0] + [1]*(depth-2) + [0]
        else:
            self.groupnorms = [0]*(depth)

        assert len(self.channels) == depth + 1
        assert len(self.kernels) == depth
        assert len(self.groupnorms) == depth

        layers = []

        # deep layers
        for i in range(depth-1):
            if not i:
                layers.append(nn.Conv3d(
                    self.channels[i],
                    self.channels[i+1],
                    self.kernels[i]))
            else:
                layers.append(Res_Block(
                    self.channels[i],
                    self.channels[i+1],
                    self.kernels[i],
                    1,
                    self.groupnorms[i-1]))
        # last layer
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(self.channels[-2], self.channels[-1], self.kernels[-1])) 
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

