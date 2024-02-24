import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


# 搭建神经网络
class Tudui(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(Tudui, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, 5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
            x = self.model(x)
            return x

#if __name__=='__main__':
#    tudui = Tudui()
 #   input = torch.ones((64,3,32,32))
 #   output = tudui(input)
 #   print(output.shape)#得到64行数据，每行10个数据，每个数据经过处理（softmax ）后得到图片概率
