import torch
from torchvision import datasets, transforms

# 데이터 전처리 설정
transform = transforms.Compose([
    transforms.ToTensor(),  # 0~255 → 0.0~1.0
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))  # CIFAR-100 평균/표준편차
])

# 데이터셋 경로
root_dir = "/data/datasets/cifar-100-python"

# 훈련셋 불러오기
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
