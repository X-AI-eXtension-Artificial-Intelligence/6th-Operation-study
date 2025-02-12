import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )
            return layers

        self.enc_1_1 = CBR2d(in_channel, 64)
        self.enc_1_2 = CBR2d(64, 64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_2_1 = CBR2d(64, 128)
        self.enc_2_2 = CBR2d(128, 128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_3_1 = CBR2d(128, 256)
        self.enc_3_2 = CBR2d(256, 256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_4_1 = CBR2d(256, 512)
        self.enc_4_2 = CBR2d(512, 512)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_5_1 = CBR2d(512, 1024)
        self.enc_5_2 = CBR2d(1024, 1024)

        self.upConv_4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_4_1 = CBR2d(1024, 512)
        self.dec_4_2 = CBR2d(512, 512)

        self.upConv_3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.dec_3_1 = CBR2d(512, 256)
        self.dec_3_2 = CBR2d(256, 256)

        self.upConv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec_2_1 = CBR2d(256, 128)
        self.dec_2_2 = CBR2d(128, 128)

        self.upConv_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec_1_1 = CBR2d(128, 64)
        self.dec_1_2 = CBR2d(64, 64)

        self.out_conv = nn.Conv2d(in_channels=64, out_channels=out_channel, kernel_size=1)# nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=out_channel, kernel_size=1),
        # )

    def forward(self, x, x_canny): # x_canney
        enc_1_1 = self.enc_1_1(x)
        enc_1_2 = self.enc_1_2(enc_1_1)
        pool_1 = self.pool_1(enc_1_2)

        enc_2_1 = self.enc_2_1(pool_1)
        enc_2_2 = self.enc_2_2(enc_2_1)
        pool_2 = self.pool_2(enc_2_2)

        enc_3_1 = self.enc_3_1(pool_2)
        enc_3_2 = self.enc_3_2(enc_3_1)
        pool_3 = self.pool_3(enc_3_2)

        enc_4_1 = self.enc_4_1(pool_3)
        enc_4_2 = self.enc_4_2(enc_4_1)
        pool_4 = self.pool_4(enc_4_2)

        enc_5_1 = self.enc_5_1(pool_4)
        enc_5_2 = self.enc_5_2(enc_5_1)

        ### 인코더에 대해 canny도 통과시켜줘야함
        enc_c_1_1 = self.enc_1_1(x_canny)
        enc_c_1_2 = self.enc_1_2(enc_c_1_1)
        pool_c_1 = self.pool_1(enc_c_1_2)

        enc_c_2_1 = self.enc_2_1(pool_c_1)
        enc_c_2_2 = self.enc_2_2(enc_c_2_1)
        pool_c_2 = self.pool_2(enc_c_2_2)

        enc_c_3_1 = self.enc_3_1(pool_c_2)
        enc_c_3_2 = self.enc_3_2(enc_c_3_1)
        pool_c_3 = self.pool_3(enc_c_3_2)

        enc_c_4_1 = self.enc_4_1(pool_c_3)
        enc_c_4_2 = self.enc_4_2(enc_c_4_1)


        upConv_4 = self.upConv_4(enc_5_2)
        cat_4 = torch.cat([upConv_4, enc_c_4_2], dim=1) # enc_4_2를 집어넣는 것이 아닌 enc_4_2에 캐니 이미지 넣은 것을

        dec_4_1 = self.dec_4_1(cat_4)
        dec_4_2 = self.dec_4_2(dec_4_1)

        upConv_3 = self.upConv_3(dec_4_2)
        cat_3 = torch.cat([upConv_3, enc_c_3_2], dim=1)
        dec_3_1 = self.dec_3_1(cat_3)
        dec_3_2 = self.dec_3_2(dec_3_1)

        upConv_2 = self.upConv_2(dec_3_2)
        cat_2 = torch.cat([upConv_2, enc_c_2_2], dim=1)
        dec_2_1 = self.dec_2_1(cat_2)
        dec_2_2 = self.dec_2_2(dec_2_1)

        upConv_1 = self.upConv_1(dec_2_2)
        cat_1 = torch.cat([upConv_1, enc_c_1_2], dim=1)
        dec_1_1 = self.dec_1_1(cat_1)
        dec_1_2 = self.dec_1_2(dec_1_1)

        out = self.out_conv(dec_1_2)
        print('### final dimension :  ', out.size())

        return out