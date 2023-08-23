import torch
from torch import nn

channels, kernel, stride = [32, 5, 2]
linear = channels * ((100 - kernel) // stride + 1) * ((160 - kernel) // stride + 1)  # 计算线性层的输入节点数
tensor = torch.zeros(1, 3, 100, 160)
image_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 25 * 40, 128),
            nn.ReLU(),
        )
image = image_model(tensor)
print(image.shape)
