import torch
import torch.nn as nn
import torch.nn.functional as F

import model.nn as N


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, features=32, is_time_embed=False, is_cls_embed=False):
        super(UNet, self).__init__()
        self.is_time_embed = is_time_embed
        self.is_cls_embed = is_cls_embed
        if self.is_time_embed:
            self.time_embed = nn.Linear(1, features)
        if self.is_cls_embed:
            self.cls_embed = nn.Linear(1, features)

        self.features = features
        self.encoder1 = UNet._block(in_channels, features, is_time_embed, is_cls_embed)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, is_time_embed, is_cls_embed)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, is_time_embed, is_cls_embed)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, is_time_embed, is_cls_embed)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, is_time_embed, is_cls_embed)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, is_time_embed, is_cls_embed)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, is_time_embed, is_cls_embed)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, is_time_embed, is_cls_embed)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, is_time_embed, is_cls_embed)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, t=None, c=None):
        enc1 = self.encoder1(x, t, c)
        enc2 = self.encoder2(self.pool1(enc1), t, c)
        enc3 = self.encoder3(self.pool2(enc2), t, c)
        enc4 = self.encoder4(self.pool3(enc3), t, c)

        bottleneck = self.bottleneck(self.pool4(enc4), t, c)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4, t, c)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3, t, c)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2, t, c)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1, t, c)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, is_time_embed, is_cls_embed):
        return _Block(in_channels, features, is_time_embed, is_cls_embed)


class _Block(nn.Module):
    def __init__(self, in_channels, features, is_time_embed=False, is_cls_embed=False):
        super(_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, features, 3, padding=1, bias=False)
        self.norm1 = N.AdaGN(1, features)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        self.norm2 = N.AdaGN(1, features)
        self.is_time_embed = is_time_embed
        if self.is_time_embed:
            self.time_embed = nn.Linear(1, features)
        self.is_cls_embed = is_cls_embed
        if self.is_cls_embed:
            self.cls_embed = nn.Linear(1, features)
        self.zero=nn.Parameter(torch.zeros(1,1),requires_grad=False)

    def forward(self, x, t, c):
        ys = self.time_embed(t) if self.is_time_embed else self.zero
        yb = self.cls_embed(c) if self.is_cls_embed else self.zero
        ys=ys.unsqueeze(-1).unsqueeze(-1)
        yb=yb.unsqueeze(-1).unsqueeze(-1)
        x = self.conv1(x)
        x = self.norm1(x, ys, yb)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x, ys, yb)
        x = F.relu(x)
        return x


if __name__ == '__main__':
    m = UNet(3, is_time_embed=False, is_cls_embed=False)
    data = torch.randn(8, 3, 128, 128)
    output = m(data)
    print(output.shape)
