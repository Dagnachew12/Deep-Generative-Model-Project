# import torch
# from torch import nn
#
#
# class Block(nn.Module):
#     def __init__(self, in_channel, out_channel, stride):
#         super(Block, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=stride, padding=1, bias=True,
#                       padding_mode='reflect'),
#             nn.InstanceNorm2d(out_channel),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class Discriminator(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Discriminator, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channel, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.conv2 = Block(64, 128, 2)
#         self.conv3 = Block(128, 256, 2)
#         self.conv4 = Block(256, 512, 2)
#         self.out = nn.Conv2d(512, out_channel, kernel_size=4, stride=1, padding=1, bias=True, padding_mode='reflect')
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.out(out)
#         out = torch.sigmoid(out)
#         return out
#
#
# class GenBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, down=True, act=True, **kwargs):
#         super(GenBlock, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.down = down
#         self.act = act
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, padding_mode='reflect', **kwargs) if down else nn.ConvTranspose2d(
#                 in_channel, out_channel, **kwargs),
#             nn.InstanceNorm2d(out_channel),
#             nn.ReLU(inplace=True) if act else nn.Identity(),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, channel):
#         super(ResidualBlock, self).__init__()
#         self.block = nn.Sequential(
#             GenBlock(channel, channel, kernel_size=3, padding=1),
#             GenBlock(channel, channel, act=False, kernel_size=3, padding=1),
#         )
#
#     def forward(self, x):
#         return x + self.block(x)
#
#
# class Generator(nn.Module):
#     def __init__(self, in_channel, feature=64, num_residual=20):
#         super(Generator, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = feature
#         self.input = nn.Sequential(
#             nn.Conv2d(in_channel, feature, kernel_size=7, stride=1, padding=3, bias=True,
#                       padding_mode='reflect'),
#             nn.InstanceNorm2d(feature),
#             nn.LeakyReLU(inplace=True),
#         )
#         self.down_block = nn.ModuleList(
#             [
#                 GenBlock(feature, feature * 2, kernel_size=3, stride=2, padding=1),
#                 GenBlock(feature * 2, feature * 4, kernel_size=3, stride=2, padding=1)
#             ]
#         )
#
#         self.res_block = nn.Sequential(
#             *[ResidualBlock(feature * 4) for _ in range(num_residual)]
#         )
#
#         self.up_block = nn.ModuleList(
#             [
#                 GenBlock(feature * 4, feature * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 GenBlock(feature * 2, feature, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
#             ]
#         )
#
#         self.out = nn.Conv2d(feature, in_channel, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
#
#     def forward(self, x):
#         x = self.input(x)
#         for layer in self.down_block:
#             x = layer(x)
#
#         x = self.res_block(x)
#         for layer in self.up_block:
#             x = layer(x)
#         x = self.out(x)
#         out = torch.tanh(x)
#
#         return out