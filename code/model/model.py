""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.antialias import Downsample1D as downsamp

# double conv
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    """with no padding"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# out conv
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=5,stride=1,padding=0)

    def forward(self, x):
        return self.conv(x)

##########################################################################
##---------- ResUnet ---------- 

# MultiScaleResidualBlock
class MultiScaleResidualBlock(nn.Module):
    def __init__(self,in_channels,bias=False):
        super(MultiScaleResidualBlock, self).__init__()
        
        self.conv_3 = nn.Conv1d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1, bias = bias)
        self.conv_5 = nn.Conv1d(in_channels, in_channels, kernel_size = 5, stride = 1, padding = 2, bias = True)
        
        self.confusion = nn.Conv1d(in_channels * 2, in_channels, kernel_size = 1, stride = 1, padding = 0, bias = True)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels*2, kernel_size=5, padding=0, bias=True),
            nn.BatchNorm1d(in_channels*2),
            nn.ReLU(inplace=True),     
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity_data = x
        output_3 = self.relu(self.conv_3(x))
        output_5 = self.relu(self.conv_5(x))

        output = torch.cat([output_3, output_5], 1)
        output = self.confusion(output)
        output = torch.add(output, identity_data)
        output = self.conv(output)
        return output

# down sampling
class Down(nn.Module):
    """Downscaling with DownSample then MultiScaleResidualBlock"""

    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Sequential(downsamp(channels=in_channels,filt_size=3,stride=2),
                                MultiScaleResidualBlock(in_channels),
                                )

    def forward(self, x):
        return self.body(x)

# up sampling
class Up(nn.Module):
    """Upscaling with UpSample then double conv"""

    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, in_channels // 2)

    def forward(self, x1, x2):
        # x2 big feature map
        # print('==========up==========')
        # print(x1.shape)
        x1 = self.up(x1) 
        # print(x1.shape) 
        # input is CHW
        
        diff = x2.size()[2] - x1.size()[2]
        
        x2 = x2[:,:,diff // 2:(diff // 2+x1.size()[2])]
        # print(x2.shape) 
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

# general
class ResUnet(nn.Module):
    def __init__(self,in_shape,n_channels=1, n_classes=1,bilinear=False):
        super(ResUnet, self).__init__()
        self.n_channels = in_shape[1] # n_channels
        self.n_classes = 1 # n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32)
        self.down2 = Down(64)
        self.down3 = Down(128)
        self.up1 = Up(256)
        self.up2 = Up(128)
        self.up3 = Up(64)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x = self.up1(x4, x3)
        # print(x.shape)
        x = self.up2(x, x2)
        # print(x.shape)
        x = self.up3(x, x1)
        # print(x.shape)
        out = self.outc(x)
        # print(out.shape)
        out = torch.sigmoid(out)
        # print(out.shape)
        return out

##########################################################################
##---------- Unet ---------- 
# without MRB
# down sampling
class Down1(nn.Module):
    """Downscaling with DownSample then conv"""

    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Sequential(downsamp(channels=in_channels,filt_size=3,stride=2),
                                  nn.Conv1d(in_channels, in_channels*2 , kernel_size=5, padding=0, bias=True),
                                  nn.BatchNorm1d(in_channels*2),
                                  nn.ReLU(inplace=True),     
                                )

    def forward(self, x):
        return self.body(x)

# general
class Unet(nn.Module):
    def __init__(self,in_shape,n_channels=1, n_classes=1,bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = in_shape[1] # n_channels
        self.n_classes = 1 # n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down1(32)
        self.down2 = Down1(64)
        self.down3 = Down1(128)
        self.up1 = Up(256)
        self.up2 = Up(128)
        self.up3 = Up(64)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x = self.up1(x4, x3)
        # print(x.shape)
        x = self.up2(x, x2)
        # print(x.shape)
        x = self.up3(x, x1)
        # print(x.shape)
        out = self.outc(x)
        # print(out.shape)
        out = torch.sigmoid(out)
        # print(out.shape)
        return out