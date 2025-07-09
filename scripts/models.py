import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
  def __init__(self):
    super(SimpleModel, self).__init__()
    self.step1 = nn.Flatten()
    self.step2 = nn.BatchNorm1d(28*28)
    self.step3 = nn.Linear(28*28, 128)
    self.step4 = nn.ReLU()
    self.step5 = nn.Linear(128, 64)
    self.step6 = nn.ReLU()
    self.step7 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.step1(x)
    x = self.step2(x)
    x = self.step3(x)
    x = self.step4(x)
    x = self.step5(x)
    x = self.step6(x)
    x = self.step7(x)
    return x

class AdvancedModel(nn.Module):
    def __init__(self):
        super(AdvancedModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)                  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)          
        x = F.relu(self.fc1(x))       
        x = self.fc2(x)                
        return x

