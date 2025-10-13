import torch
import torch.nn as nn
import torch.nn.functional as F
def modulate(x, shift, scale):
    return x * (1 + scale) + shift

'''
class ada_bn3D(nn.Module):
    t_dim = 320
    affine = True
    def __init__(self, dim, momentum= 0.1):
        super(ada_bn3D, self).__init__()
        t_dim = ada_bn3D.t_dim
        self.main = nn.BatchNorm3d(dim,affine  = ada_bn3D.affine, momentum=momentum)
        self.adaBN_modulation = nn.Sequential(nn.Linear(t_dim,t_dim, bias=True),nn.SiLU(),nn.Linear(t_dim,2*dim, bias=True))
    def forward(self, x,t =None):
        if t is not None:
            scale,shift = self.adaBN_modulation(t)[...,None,None,None].chunk(2, dim=1)
            return modulate(self.main(x), shift, scale)
        else:
            return self.main(x)'''
class ada_bn3D(nn.Module):
    t_dim = 320
    affine = True
    def __init__(self, dim, momentum= 0.1):
        super(ada_bn3D, self).__init__()
        t_dim = ada_bn3D.t_dim
        self.main = nn.BatchNorm3d(dim,affine  = ada_bn3D.affine, momentum=momentum)
        self.adaBN_modulation = nn.Sequential(nn.Conv3d(t_dim,t_dim,1, bias=False),nn.SiLU(),nn.Conv3d(t_dim,2*dim,1, bias=False))
    def forward(self, x,t =None):
        if t is not None:
            scale,shift = self.adaBN_modulation(t).chunk(2, dim=1)
            return modulate(self.main(x), shift, scale)
        else:
            return self.main(x)

class Bottleneck3D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        stride=1,
        dilation=[1, 1, 1],
        expansion=4,
        downsample=None,
        fist_dilation=1,
        multi_grid=1,
        bn_momentum=0.0003,
    ):
        super(Bottleneck3D, self).__init__()
        # often，planes = inplanes // 4
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 1, 3),
            stride=(1, 1, stride),
            dilation=(1, 1, dilation[0]),
            padding=(0, 0, dilation[0]),
            bias=False,
        )
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 3, 1),
            stride=(1, stride, 1),
            dilation=(1, dilation[1], 1),
            padding=(0, dilation[1], 0),
            bias=False,
        )
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            dilation=(dilation[2], 1, 1),
            padding=(dilation[2], 0, 0),
            bias=False,
        )
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False
        )
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(planes, momentum=bn_momentum),
        )

    def forward(self, x,t = None):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x),t))
        out2 = self.bn2(self.conv2(out1),t)
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu),t)
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu),t)
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu),t)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu


class Process(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, dilations=[1, 2, 3]):
        super(Process, self).__init__()
        self.main = nn.ModuleList(
            [
                Bottleneck3D(
                    feature,
                    feature // 4,
                    bn_momentum=bn_momentum,
                    norm_layer=norm_layer,
                    dilation=[i, i, i],
                )
                for i in dilations
            ]
        )

    def forward(self, x,t = None):

        for fn in self.main:
            x = fn(x,t)
        return x

class Downsample(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, expansion=8):
        super(Downsample, self).__init__()
        self.main = nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(
                    feature,
                    int(feature * expansion / 4),
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
            )
        self.norm = norm_layer(int(feature * expansion / 4), momentum=bn_momentum)
    def forward(self, x,t = None):
        x = self.main(x)
        x = self.norm(x,t)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            
        )
        self.norm = norm_layer(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU()

    def forward(self, x,t = None):
        x = self.main(x)
        x = self.relu(self.norm(x,t))
        return x


class UNet3D(nn.Module):
    def __init__(
        self,
        feature,
        bn_momentum=0.1,
        affine = True
    ):
        super(UNet3D, self).__init__()
        self.feature = feature
        ada_bn3D.t_dim = self.feature
        ada_bn3D.affine = affine
        print("affine:",affine)
        norm_layer = ada_bn3D
        self.bn_l1 = norm_layer(self.feature)
        self.bn_l2 = norm_layer(self.feature* 2)
        self.bn_l3 = norm_layer(self.feature* 4)
        dilations = [1, 2, 3]
        self.process_l1 = Process(self.feature, norm_layer, bn_momentum, dilations=[1, 2, 3])
        self.down_l1_l2 = Downsample(self.feature,norm_layer, bn_momentum)
        

        self.process_l2 = Process(self.feature * 2, norm_layer, bn_momentum, dilations=[1, 1, 1])
        self.down_l2_l3 = Downsample(self.feature * 2, norm_layer, bn_momentum)

        self.process_l3_u = Process(self.feature * 4, norm_layer, bn_momentum, dilations=[1, 1, 1])
        self.up_13_l2 = Upsample(
            self.feature * 4, self.feature * 2, norm_layer, bn_momentum
        )

        self.process_l2_u = Process(self.feature * 2, norm_layer, bn_momentum, dilations=[1, 2, 3])
        self.up_12_l1 = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )


    def forward(self, x3d_l1,t = None):
        if t is not None:
            t_l1,t_l2,t_l3 = t
        else:
            t_l1,t_l2,t_l3 = None,None,None
        #x3d_l1 = self.bn_l1( x3d_l1,t = t_l1)
        x3d_l2 = self.down_l1_l2(self.process_l1(x3d_l1,t= None),t= None)
        #x3d_l2 = self.bn_l2( x3d_l2,t = t_l2)
        x3d_l3 = self.down_l2_l3(self.process_l2(x3d_l2,t= None),t= None)
        #x3d_l3 = self.bn_l3( x3d_l3,t = t_l3)
        #print("this time ")
        x3d_up_l2 = self.up_13_l2(self.process_l3_u(x3d_l3,t= None),t= None) + x3d_l2
        x3d_up_l1 = self.up_12_l1(self.process_l2_u(x3d_up_l2,t= None),t= None) + x3d_l1
        return x3d_up_l1




