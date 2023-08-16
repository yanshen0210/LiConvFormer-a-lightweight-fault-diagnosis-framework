import torch.nn as nn
import torch.nn.functional as F


class basic_block(nn.Module):
    def __init__(self,in_channels):
        '''定义了带实线部分的残差块'''
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x+y)


class basic_block2(nn.Module):
    '''定义了带虚线部分的残差块'''
    def __init__(self,in_channels,out_channels):
        super(basic_block2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels,out_channels,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        z = self.bn1(self.conv1(x))
        y = F.relu(self.bn2(self.conv2(x)))
        y = self.bn3(self.conv3(y))
        return F.relu(y+z)


class ResNet18(nn.Module):
    '''按照网络结构图直接连接，确定好通道数量就可以'''
    def __init__(self, _, in_channel, out_channel):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv1d(in_channel,64,kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxp1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.rest1 = basic_block(64)
        self.rest2 = basic_block(64)
        self.rest3 = basic_block2(64,128)
        self.rest4 = basic_block(128)
        self.rest5 = basic_block2(128,256)
        self.rest6 = basic_block(256)
        self.rest7 = basic_block2(256,512)
        self.rest8 = basic_block(512)
        self.avgp1 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512,out_channel)

    def forward(self, x):
        in_size = x.size(0)
        x = self.maxp1(F.relu(self.bn1(self.conv1(x))))
        x = self.rest1(x)
        x = self.rest2(x)
        x = self.rest3(x)
        x = self.rest4(x)
        x = self.rest5(x)
        x = self.rest6(x)
        x = self.rest7(x)
        x = self.rest8(x)
        x = self.avgp1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
