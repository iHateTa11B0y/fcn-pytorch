import torch
from torch import nn
import torch.nn.functional as F

class DownPathModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownPathModule, self).__init__()
        net = nn.Sequential()
        net.add_module('conv1', nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
        net.add_module('bn1', nn.BatchNorm2d(out_channel))
        net.add_module('relu1', nn.ReLU(inplace=True))
        net.add_module('conv2', nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        net.add_module('bn2', nn.BatchNorm2d(out_channel))
        net.add_module('relu2', nn.ReLU(inplace=True))
        net.add_module('maxpool', nn.MaxPool2d(2))
        
        self.net = net
    
    def forward(self, x):
        return self.net(x)

class UpPathModule(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(UpPathModule, self).__init__()
        if bilinear:
            Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            Up = nn.ConvTranspose2d(in_channel // 2, in_channel // 2, kernel_size=2, stride=2)
  
        self.upconv = nn.Sequential(
                                    nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                                    Up,
                                   )
        net = nn.Sequential()
        net.add_module('conv1', nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
        net.add_module('bn1', nn.BatchNorm2d(out_channel))
        net.add_module('relu1', nn.ReLU(inplace=True))
        net.add_module('conv2', nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        net.add_module('bn2', nn.BatchNorm2d(out_channel))
        net.add_module('relu2', nn.ReLU(inplace=True))

        self.net = net

    def forward(self, x, lateral):
        x = self.upconv(x)
        #print(x.shape)
        diffY = - x.size()[2] + lateral.size()[2]
        diffX = - x.size()[3] + lateral.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        #print(x.shape)
        x = torch.cat([x, lateral], dim=1)
        return self.net(x)

class DownPath(nn.Module):
    def __init__(self, in_channel, out_channel_stage_0=64, stage_num=4):
        super(DownPath, self).__init__()
        self.stage_num = stage_num
        out_channel = out_channel_stage_0
        for sn in range(stage_num):
            setattr(self, 'dp_stage_{}'.format(sn), DownPathModule(in_channel, out_channel))
            in_channel = out_channel
            out_channel *= 2
    def forward(self, x):
        res = []
        for sn in range(self.stage_num):
            stagei = getattr(self, 'dp_stage_{}'.format(sn))
            x = stagei(x)
            res.append(x.clone())
        return res
        
class UpPath(nn.Module):
    def __init__(self, in_channel, stage_num=4, bilinear=True):
        super(UpPath, self).__init__()
        self.stage_num = stage_num
        out_channel = in_channel
        for i in range(stage_num):
            sn = stage_num - 1 - i
            out_channel = int(out_channel / 2)
            setattr(self, 'up_stage_{}'.format(sn), UpPathModule(in_channel, out_channel, bilinear=bilinear))
            in_channel = out_channel

    def forward(self, x, lateral_res):
        for i in range(self.stage_num):
            sn = self.stage_num - 1 - i
            stagei = getattr(self, 'up_stage_{}'.format(sn))
            x = stagei(x, lateral_res[sn])
        return x


class Unet(nn.Module):
    def __init__(self, in_channel, 
                       out_channel_down_stage_0=64, 
                       out_channel_out_conv=2, 
                       stage_num=4,
                       bilinear=True):
        super(Unet, self).__init__()
        net_dp = DownPath(in_channel, out_channel_down_stage_0, stage_num)

        # bottom stage
        out_channel = 2 ** stage_num * out_channel_down_stage_0
        self.bottom = DownPathModule(out_channel//2, out_channel)
        in_channel = out_channel

        net_up = UpPath(in_channel, stage_num, bilinear)

        if bilinear:
            Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            Up = nn.ConvTranspose2d(in_channel // 2, in_channel // 2, kernel_size=2, stride=2)
        self.outconv = nn.Sequential(
                                     Up,
                                     nn.Conv2d(out_channel_down_stage_0, out_channel_out_conv, 1)
        )
        self.net_dp = net_dp
        self.net_up = net_up
    
    def forward(self, x):
        resx = self.net_dp(x)
        x = self.bottom(resx[-1])
        x = self.net_up(x, resx)
        return self.outconv(x)


