import torch
import torch.nn as nn


class DenseNetConv(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, padding=1,stride=1):
        super(DenseNetConv, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=ks, padding=padding, stride=stride),
                                   nn.ReLU(inplace=True))

    def forward(self,x):
        return self.layer(x)


class DenseNetBlock(nn.Module):
    def __init__(self, ks=3, padding=1, stride=1):
        super(DenseNetBlock, self).__init__()

        self.features = []
        self.growth_rate = 16
        self.num_convs = 8



        self.layers = nn.ModuleList([DenseNetConv(in_channels=(i+1)*self.growth_rate, out_channels=self.growth_rate) for i in range(self.num_convs)])
        print self.layers

    def forward(self,x):

        for i in range(8):
            self.features[i] = self.layers[i](x)
            x = torch.cat(seq=self.features[:i], dim=0)

            print x.size()



