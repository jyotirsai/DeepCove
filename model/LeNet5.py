import torch
from torch import nn

class LeNet5(nn.Module):
  def __init__(self):
      super().__init__()
      self.conv_1 = nn.Sequential(
          nn.Conv2d(in_channels=1,out_channels=6, kernel_size=5, stride=1),
          nn.ReLU(),
          nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
      )
      self.conv_2 = nn.Sequential(
          nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
          nn.ReLU(),
          nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
      )
      self.fc_1 = nn.Linear(in_features=256, out_features=120, bias=True)
      self.fc_2 = nn.Linear(120, 84, bias=True)
      self.relu_3 = nn.ReLU()

      self.fc_3 = nn.Linear(84, 10, bias=True)

  def forward(self, x):
      x = self.conv_1(x)
      x = self.conv_2(x)
      x = torch.flatten(x, 1)
      x = self.fc_1(x)
      x = self.fc_2(x)
      x = self.relu_3(x)
      x = self.fc_3(x)
      return x