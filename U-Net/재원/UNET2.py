# 필요한 library import
import os
import numpy as np

import torch
import torch.nn as nn


import torch
import torch.nn as nn

# Conv + BatchNorm + ReLU 블록 정의
class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# 다운샘플링 블록 (MaxPooling)
class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)


# 업샘플링 블록 (Transpose Convolution)
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


# U-Net 모델 정의
class UNET2(nn.Module):
    def __init__(self, in_channels=3, out_channels=59): #초기값 RGB 3채널, Output 클래스 59개로 변향
        super(UNET2, self).__init__()

        # 다운샘플링 경로 (Contracting path)
        self.enc1 = DoubleConvolution(in_channels, 64)
        self.enc2 = DoubleConvolution(64, 128)
        self.enc3 = DoubleConvolution(128, 256)
        self.enc4 = DoubleConvolution(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (가장 깊은 층)
        self.bottleneck = DoubleConvolution(512, 1024)

        # 업샘플링 경로 (Expanding path)
        self.up4 = UpSample(1024, 512)
        self.dec4 = DoubleConvolution(1024, 512)

        self.up3 = UpSample(512, 256)
        self.dec3 = DoubleConvolution(512, 256)

        self.up2 = UpSample(256, 128)
        self.dec2 = DoubleConvolution(256, 128)

        self.up1 = UpSample(128, 64)
        self.dec1 = DoubleConvolution(128, 64)

        # 최종 출력 레이어 (클래스 개수: 59)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 수축경로 (3*3 Conv + 다운샘플링)
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # 확장경로 (업샘플링 + 스킵커넥션 + 3*3 Conv)
        up4 = self.up4(bottleneck)
        cat4 = torch.cat((up4, enc4), dim=1)
        dec4 = self.dec4(cat4)

        up3 = self.up3(dec4)
        cat3 = torch.cat((up3, enc3), dim=1)
        dec3 = self.dec3(cat3)

        up2 = self.up2(dec3)
        cat2 = torch.cat((up2, enc2), dim=1)
        dec2 = self.dec2(cat2)

        up1 = self.up1(dec2)
        cat1 = torch.cat((up1, enc1), dim=1)
        dec1 = self.dec1(cat1)

        # 최종 출력
        return self.final_conv(dec1)
