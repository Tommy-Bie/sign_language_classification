# 自己设计的CNN模型
import torch.nn as nn
import torch.nn.functional as F
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)       # conv1 （RGB图像，输入通道数为3）
        self.conv2 = nn.Conv2d(6, 16, 3, 1)      # conv2
        self.fc1 = nn.Linear(54 * 54 * 16, 120)  # fc1
        self.fc2 = nn.Linear(120, 84)            # fc2
        self.fc3 = nn.Linear(84, 6)              # fc3 6个类别

    def forward(self, X):
        X = F.relu(self.conv1(X))                # 激活函数：ReLU
        X = F.max_pool2d(X, 2, 2)                # 最大池化
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54 * 54 * 16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)           # 输出层为softmax