import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()

def getDigitMNIST():
  mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
  mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)

  train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
  test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

  return train_loader, test_loader