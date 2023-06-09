import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

# 搭建神经网络
class TICNN(nn.Module):
    def __init__(self):
        super(TICNN, self).__init__()
        # 1X1024-16X128  通道X信号尺寸
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=64,stride=8,padding=28)
        self.dropout1=nn.Dropout(p=0.5)
        self.bn1=nn.BatchNorm1d(num_features=16)
        # 16X128-16X64  通道X信号尺寸
        self.pool1=nn.MaxPool1d(kernel_size=2,stride=2,padding=0)

        # 16X64-64X64  通道X信号尺寸
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        # 64X64-64X32  通道X信号尺寸
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # 64X32-64X32  通道X信号尺寸
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        # 64X32-64X16  通道X信号尺寸
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # 64X16-64X16  通道X信号尺寸
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=64)
        # 64X16-64X8  通道X信号尺寸
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # 64X8-64X8  通道X信号尺寸
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(num_features=64)
        # 64X8-64X4  通道X信号尺寸
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # 64X4-64X4  通道X信号尺寸
        self.conv6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(num_features=64)
        # 64X4-64X2  通道X信号尺寸
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # 将数据展平
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*2, out_features=100, bias=True)

        # 100-----13
        self.fc2 = nn.Linear(in_features=100, out_features=13, bias=True)

    def forward(self, x):
        x=self.dropout1(self.conv1(x))
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv3(x)))
        x = self.pool4(x)
        x = F.relu(self.bn5(self.conv3(x)))
        x = self.pool5(x)
        x = F.relu(self.bn6(self.conv3(x)))
        x = self.pool6(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model=TICNN() #实例化网络模型
    wang_DS_RGB=Model.to(device) #将模型转移到cuda上
    input=torch.ones((64,1,1024)) #生成一个batchsize为64的，通道数为1，宽度为2048的信号
    input=input.to(device) #将数据转移到cuda上
    output=Model(input) #将输入喂入网络中进行处理
    print(output.shape)
    summary(Model,input_size=(1,1024)) #输入一个通道为1的宽度为2048，并展示出网络模型结构和参数
