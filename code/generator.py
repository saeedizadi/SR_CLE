import torch.nn as nn
import torch.nn.functional as F

from utils import initialize_weights


class Residual_Block(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3, out_channels=64, stride=1):
        super(Residual_Block, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=1, bias=True),
            nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.layers(x) + x


class UpSample_Block(nn.Module):
    # changed 256 --> 64 since pixelShuffle has been removed
    def __init__(self, in_channels=64, out_channels=64):
        super(UpSample_Block, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),

            # Removes Upamspling
            # nn.PixelShuffle(2),
            nn.PReLU())

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, nResBlks, nUpBlks):
        super(Generator, self).__init__()

        self.nResBlks = nResBlks
        self.nUpBlks = nUpBlks

        # --- change 3 --> 1 as the images are converted to grayscale
        self.layers0 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4, bias=True),
                                     nn.PReLU())

        self.layers1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                     nn.BatchNorm2d(64)
                                     )

        # --- change 3 --> 1 as the images are converted to grayscale
        self.conv = nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4, bias=True)

        self.resblocks = nn.ModuleList([Residual_Block() for i in range(nResBlks)])
        self.upsamplingblocks = nn.ModuleList([UpSample_Block() for i in range(nUpBlks)])

        initialize_weights(self, method='kaiming')

    def forward(self, x):

        x = self.layers0(x)

        y = x.clone()

        for i in range(self.nResBlks):
            y = self.resblocks[i](y)

        y = self.layers1(y) + x

        for i in range(self.nUpBlks):
            y = self.upsamplingblocks[i](y)

        # --- added sigmoid to scale output in [0,1]
        y = F.sigmoid(self.conv(y))

        return y
