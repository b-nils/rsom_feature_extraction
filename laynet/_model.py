# adapted from

# https://github.com/jvanvugt/pytorch-unet
# Adapted from https://discuss.pytorch.org/t/unet-implementation/426
import torch
from torch import nn
import torch.nn.functional as F
import math

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
        dropout=False
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
            dropout (bool) if True, add dropout layer in up block
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.dropout = dropout
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            # print(i)
            # if self.dropout and i<=1:
            if self.dropout and i<depth-2:
                # dropout = True
                self.up_path.append(
                    UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, True)
                    )
                # print('UNetUpBlock w dropout')
            else:
                self.up_path.append(
                    UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, False)
                )
                # print('UNetUpBlock')
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)

    def count_parameters(self):

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        # print(out.shape)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, dropout):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners = True),
                # Upsample is deprecated. it uses nn.functional.interpolate.
                # but this cannot used in nn.sequential, so I commented out the 
                # warning in nn.Upsample.
                # Changed from align_corners=None to True
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.is_dropout = dropout
        self.dropout = nn.Dropout(p=0.5)

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        if self.is_dropout:
            out = self.dropout(out)
            # print('self.training', self.dropout.training)
        else:
            # print('self.training', self.dropout.training)
            pass
        out = self.conv_block(out)
        # debug
        # print(out.shape)
        return out



class Fcn(nn.Module):
    def __init__(self, 
            in_channels=2,
            channels=[2, 8, 16, 24, 32, 64, 128, 1],
            kernels=[5, 5, 5, 5, 5, 5, 1],
            depth=7,
            padding=True,
            dropout=False, 
            batchnorm=True):
        
        super(Fcn, self).__init__()
       
        # SETTINGS
        self.in_channels = in_channels
        self.depth = depth
        self.dropout = dropout
        self.padding = padding
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
        
        self.batchnorm = batchnorm
        if batchnorm:
            self.batchnorms = [0] + [1]*(depth-2) + [0]
        else:
            self.batchnorms = [0]*(depth)

        assert len(self.dropouts) == depth
        assert len(self.channels) == depth + 1
        assert len(self.kernels) == depth
        assert len(self.batchnorms) == depth

        layers = []
        for i in range(depth-1):
            layers.append(FcnBlock(
                self.channels[i],
                self.channels[i+1],
                self.kernels[i],
                self.batchnorms[i],
                self.dropouts[i],
                self.padding))
        # last layer
        layers.append(nn.Conv2d(self.channels[-2], self.channels[-1], self.kernels[-1])) 

        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)


class FcnBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, batchnorm, dropout, padding):
        super(FcnBlock, self).__init__()

        block = []

        block.append(nn.Conv2d(in_size, 
            out_size, 
            kernel_size, 
            padding=(math.floor(kernel_size/2) if padding else 0)))
        block.append(nn.ReLU())
        if batchnorm:
            block.append(nn.BatchNorm2d(out_size))
        if dropout:
            block.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

if __name__ == '__main__':

    A = torch.ones([1,2,100,100])

    net = Fcn(padding=True)

    y = net(A)


