import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

batch_size = 256
epoch = 30

root_dir = "/data/datasets/cifar-100-python"

train_dataset = datasets.CIFAR100(
    root=root_dir,
    train=True,
    transform=transform,
    download=False  # 이미 로컬에 있으므로 False
)

# 테스트셋 불러오기
test_dataset = datasets.CIFAR100(
    root=root_dir,
    train=False,
    transform=transform,
    download=False
)

# 데이터로더 예시
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 확인
images, labels = next(iter(train_loader))
print(f"이미지 배치 shape: {images.shape}, 레이블 shape: {labels.shape}")