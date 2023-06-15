import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super(Discriminator, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 128, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            nn.Dropout2d(p=0.4),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.Dropout2d(p=0.4),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=2),
            nn.Dropout2d(p=0.4),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=2),
            nn.Dropout2d(p=0.4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        return out


class Block(nn.Module):
    def __int__(self, in_channel, out_channel, stride=2):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=stride, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Generator(nn.Module):
    def __init__(self, in_channels=3, feature=64):
        super(Generator, self).__init__()
        self.input_down = nn.Sequential(
            nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(feature, feature * 2, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(2 * feature),
            nn.LeakyReLU(0.2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(feature * 2, feature * 4, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(4 * feature),
            nn.LeakyReLU(0.2),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(feature * 4, feature * 8, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(8 * feature),
            nn.LeakyReLU(0.2),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(feature * 8, feature * 8, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(8 * feature),
            nn.LeakyReLU(0.2),
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(feature * 8, feature * 8, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(8 * feature),
            nn.LeakyReLU(0.2),
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(feature * 8, feature * 8, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(8 * feature),
            nn.LeakyReLU(0.2),
        )

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(feature * 8, feature * 8, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(feature * 8, feature * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout2d(0.4),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(feature * 8 * 2, feature * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout2d(0.4),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(feature * 8 * 2, feature * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout2d(0.4),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(feature * 8 * 2, feature * 8, kernel_size=4, stride=2, padding=1, bias=False, ),
            nn.ReLU()
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(feature * 8 * 2, feature * 4, kernel_size=4, stride=2, padding=1, bias=False, ),
            nn.Dropout2d(0.4),
            nn.ReLU()
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(feature * 4 * 2, feature * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU()
        )
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(feature * 2 * 2, feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU()
        )

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(feature * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.input_down(x)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out5 = self.down4(out4)
        out6 = self.down5(out5)
        out7 = self.down6(out6)

        bottleneck = self.bottle_neck(out7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, out7], 1))
        up3 = self.up3(torch.cat([up2, out6], 1))
        up4 = self.up4(torch.cat([up3, out5], 1))
        up5 = self.up5(torch.cat([up4, out4], 1))
        up6 = self.up6(torch.cat([up5, out3], 1))
        up7 = self.up7(torch.cat([up6, out2], 1))
        out = self.final_up(torch.cat([up7, out1], 1))
        return out
