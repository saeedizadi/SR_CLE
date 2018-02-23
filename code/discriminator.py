import torch.nn as nn
import torch

import torch.nn.functional as F




class DenseNetBlock(nn.Module):
    def __init__(self, in_channels=128, ks=3, padding=1, stride=1):
        super(DenseNetBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
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

    def forward(self, x):
        # x = self.conv0(x) # Nx128x64x64
        conv1_out = self.conv1(x)  # Nx16x64x64

        conv2_out = self.conv2(conv1_out)  # Nx16x64x64

        conv3_in = torch.cat(([conv1_out, conv2_out]), 1)  # Nx32x64x64
        conv3_out = self.conv3(conv3_in)

        conv4_in = torch.cat(([conv1_out, conv2_out, conv3_out]), 1)  # Nx48x64x64
        conv4_out = self.conv4(conv4_in)

        conv5_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out]), 1)  # Nx64x64x64
        conv5_out = self.conv5(conv5_in)

        conv6_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]), 1)  # Nx80x64x64
        conv6_out = self.conv6(conv6_in)

        conv7_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out]), 1)  # Nx96x64x64
        conv7_out = self.conv7(conv7_in)

        conv8_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out, conv7_out]),
                             1)  # Nx96x64x64
        conv8_out = self.conv8(conv8_in)

        output = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out,
                             conv5_out, conv6_out, conv7_out, conv8_out]), 1)

class Discrimantor(nn.Module):
    def __init__(self):
        super(Discrimantor, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(1, 128, kernel_size=3, padding=1, stride=1),
                                   nn.ReLU(inplace=True))

        self.layer1 = nn.Sequential(DenseNetBlock(),
                                    DenseNetBlock())


        self.layer2 = nn.Sequential(DenseNetBlock(),
                                    DenseNetBlock())

        self.layer3 = nn.Sequential(DenseNetBlock(),
                                    DenseNetBlock())

        self.pool = nn.MaxPool2d(2)

        self.classifier = nn.Sequential(nn.Linear(256, 100),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(100,1))

    def forward(self,x):

        x = self.conv0(x)

        conv1_out = x.clone()

        x = self.layer1(x)
        x = self.pool(x)

        x = self.layer2(x)
        x = self.pool(x)

        x = self.layer3(x)
        x = F.avg_pool2d(x,kernel_size=(16,16))

        x = self.classifier(x)

        return F.sigmoid(x)


