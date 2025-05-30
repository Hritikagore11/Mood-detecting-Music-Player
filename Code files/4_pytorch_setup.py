
import os
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = '/content/music/train'
test_path = '/content/music/test'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
