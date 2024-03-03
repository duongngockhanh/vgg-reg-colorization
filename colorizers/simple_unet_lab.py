import torch.nn as nn
import torch
from .base_color import *

class ConvBlock(nn.Module):
  def __init__(self,in_channels, out_channels):
    super().__init__()
    self.model = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(num_features=out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(num_features=out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    x = self.model(x)
    return x

class Encoder(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.maxpool = nn.MaxPool2d(kernel_size=2)
    self.conv = ConvBlock(in_channels, out_channels)

  def forward(self, x):
    x = self.maxpool(x)
    x = self.conv(x)
    return x

class Decoder(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    self.conv = ConvBlock(in_channels, out_channels)

  def forward(self, previous_output, skip_output):
    x = self.conv_trans(previous_output)
    output = torch.cat([x, skip_output], dim=1)
    output = self.conv(output)
    return output

class Simple_UNet_Lab(BaseColor):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_conv = ConvBlock(in_channels, 64)

    self.enc1 = Encoder(64, 128)
    self.enc2 = Encoder(128, 256)
    self.enc3 = Encoder(256, 512)
    self.enc4 = Encoder(512, 1024)

    self.dec1 = Decoder(1024, 512)
    self.dec2 = Decoder(512, 256)
    self.dec3 = Decoder(256, 128)
    self.dec4 = Decoder(128, 64)

    self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    self.tanh = nn.Tanh()

  def forward(self, x):
    x = self.normalize_l(x)
    x1 = self.in_conv(x)

    x2 = self.enc1(x1)
    x3 = self.enc2(x2)
    x4 = self.enc3(x3)
    x5 = self.enc4(x4)

    x = self.dec1(x5, x4)
    x = self.dec2(x, x3)
    x = self.dec3(x, x2)
    x = self.dec4(x, x1)

    x = self.out_conv(x)
    x = self.tanh(x)
    x = self.unnormalize_ab(x)

    return x