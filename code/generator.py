import torch.nn as nn

from utils import initialize_weights
import torch.nn.functional as F
class Residual_Block(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3, out_channels= 64, stride=1):
        super(Residual_Block, self).__init__()

        self.layers = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU(),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
                                    nn.BatchNorm2d(out_channels)
                                    )
    def forward(self,x):

        #added RELU after the addition.
        return  F.prelu(self.layers(x) + x)


class UpSample_Block(nn.Module):

    # changed 256 --> 64 since pixelShuffle has been removed
    def __init__(self, in_channels=64, out_channels=64):
        super(UpSample_Block, self).__init__()

        self.layers = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),

                                    #Removes Upamspling
                                    #nn.PixelShuffle(2),
                                    nn.PReLU())
    def forward(self,x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, nResBlks, nUpBlks):
        super(Generator, self).__init__()

        self.nResBlks = nResBlks
        self.nUpBlks = nUpBlks

        self.layers0 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, bias=True),
                                     nn.PReLU())

        self.layers1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                     nn.BatchNorm2d(64)
                                     )

        self.conv = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=True)

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

        y = self.conv(y)

        return y
