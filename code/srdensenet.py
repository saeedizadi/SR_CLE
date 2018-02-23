import torch
import torch.nn as nn
from utils import initialize_weights
import torch.nn.functional as F


class UpSample_Block(nn.Module):
    # changed 256 --> 64 since pixelShuffle has been removed
    def __init__(self, in_channels=256, out_channels=1024):
        super(UpSample_Block, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class DenseNetConv(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, padding=1,stride=1):
        super(DenseNetConv, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=ks, padding=padding, stride=stride),
                                   nn.ReLU(inplace=True))

    def forward(self,x):
        return self.layer(x)


class DenseNetBlock(nn.Module):
    def __init__(self, in_channels=128 ,ks=3, padding=1, stride=1):
        super(DenseNetBlock, self).__init__()

        self.features = []

        #self.conv0 = nn.Sequential(nn.Conv2d(1, 128, 3, 1, 1),
        #                           nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=16,kernel_size=3,stride=1,padding=1),
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

        #x = self.conv0(x) # Nx128x64x64
        conv1_out = self.conv1(x) # Nx16x64x64

        conv2_out = self.conv2(conv1_out) # Nx16x64x64

        conv3_in = torch.cat(([conv1_out, conv2_out]), 1) #Nx32x64x64
        conv3_out = self.conv3(conv3_in)

        conv4_in = torch.cat(([conv1_out, conv2_out, conv3_out]), 1) #Nx48x64x64
        conv4_out = self.conv4(conv4_in)

        conv5_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out]), 1)#Nx64x64x64
        conv5_out = self.conv5(conv5_in)

        conv6_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]), 1) #Nx80x64x64
        conv6_out = self.conv6(conv6_in)

        conv7_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out]), 1) #Nx96x64x64
        conv7_out = self.conv7(conv7_in)

        conv8_in = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out, conv7_out]), 1) #Nx96x64x64
        conv8_out = self.conv8(conv8_in)

        output = torch.cat(([conv1_out, conv2_out, conv3_out, conv4_out,
                            conv5_out, conv6_out, conv7_out, conv8_out]), 1)

        return output


class SRDenseNet_ALL(nn.Module):
    def __init__(self, num_denseblks):
        super(SRDenseNet_ALL, self).__init__()


        self.num_denseblks = num_denseblks

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True))



        self.growth_rate=128
        self.denseblks = nn.ModuleList([DenseNetBlock(self.growth_rate*(i+1)) for i in range(num_denseblks)])


        self.bottleneck = nn.Conv2d(1152, 256, kernel_size=1, stride=1)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                     nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                     nn.ReLU(inplace=True))

        self.reconst_conv = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        initialize_weights(self, 'kaiming')


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        low_feats = x.clone()

        features = [low_feats]
        for i in range(self.num_denseblks):
            x = torch.cat((features[:]), dim=1)
            features.append(self.denseblks[i](x))

        x = torch.cat((features[:]), dim=1)

        x = self.bottleneck(x)


        x = self.deconv1(x)
        x = self.deconv2(x)

        out = self.reconst_conv(x)
        return F.sigmoid(out)

class SRDenseNet(nn.Module):
    def __init__(self, num_denseblks):
        super(SRDenseNet, self).__init__()


        self.num_denseblks = num_denseblks

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True))


        self.denseblks = nn.ModuleList([DenseNetBlock() for i in range(num_denseblks)])


        #self.bottleneck = nn.Conv2d(256, 128, kernel_size=1, stride=1)

        #self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        #                             nn.ReLU(inplace=True))
        #self.deconv2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        #                             nn.ReLU(inplace=True))

        self.reconst_conv = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

	self.up1 = UpSample_Block()
	self.up2 = UpSample_Block()
	self.up3 = UpSample_Block()

        initialize_weights(self, 'kaiming')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        low_feats = x.clone()



        for i in range(self.num_denseblks):
            x = self.denseblks[i](x)

        deconv1_in = torch.cat(([low_feats, x]), dim=1)

        #x = self.deconv1(deconv1_in)
        #x = self.deconv2(x)
        x = deconv1_in
	x = self.up1(x)
	x = self.up2(x)
	x = self.up3(x)
        out = self.reconst_conv(x)

        return F.sigmoid(out)


