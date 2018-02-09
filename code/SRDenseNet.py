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

        self.conv0 = nn.Sequential(nn.Conv2d(1, 128, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(128,16,3,1,1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(48, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.Conv2d(64, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.Conv2d(80, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(nn.Conv2d(96, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(nn.Conv2d(112, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True))


    def forward(self,x):
        x = self.conv0(x) # Nx128x64x64
        conv1_out = self.conv1(x) # Nx16x64x64

        conv2_out = self.conv2(conv1_out) # Nx16x64x64

        conv3_in = torch.cat(([conv1_out, conv2_out]), 1) #Nx32x64x64
        conv3_out = self.conv3(conv3_in)

        conv4_in = torch.cat(([conv1_out, conv2_out, conv3_out]), 1) #Nx48x64x64
        conv4_out = self.conv4(conv4_in)

        conv5_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out]), 1)#Nx64x64x64
        conv5_out = self.conv2(conv5_in)

        conv6_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]), 1) #Nx80x64x64
        conv6_out = self.conv6(conv6_in)

        conv7_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out]), 1) #Nx96x64x64
        conv7_out = self.conv7(conv7_in)



        conv8_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out, conv7_out]), 1) #Nx96x64x64
        conv8_out = self.conv8(conv8_in)

        output = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out,
                            conv5_out, conv6_out, conv7_out, conv8_out]), 1)

        print output.size()

        return output






