import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torchvision import models
import os


def train_loop(dataloader, model, loss_func, optimizer):
	size = len(dataloader)
	images, _ = next(iter(train_loader))
	batch_size = images.shape[0]

	model.train()

	for batch, (inputs, labels) in enumerate(dataloader):
		inputs = inputs.to(device)
		labels = labels.to(device)

		output = model(inputs)
		prob, preds = torch.max(outputs, dim=1)
		loss = loss_func(outputs, labels)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		print(f"loss: {loss}, {(batch+1)*batch_size}/{size}")


learning_rate = 0.001
epoch_num = 30
batch_size = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리 설정
transform = transforms.Compose([
    transforms.ToTensor(),  # 0~255 → 0.0~1.0
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))  # CIFAR-100 평균/표준편차
])

# 데이터셋 경로
root_dir = "/data/datasets"

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# 확인
images, labels = next(iter(train_loader))
print(f"이미지 배치 shape: {images.shape}, 레이블 shape: {labels.shape}")

resnet = models.resnet50(pretrained = True)
resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=100)

resnet = resnet.to(device)

CE_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)

for epoch in range(epoch_num):
	loss_avg = 0.0
	acc_avg = 0.0
	print(f"--------------------Epoch:{epoch}--------------------")

	train_loop(train_loader, resnet, CE_loss, optimizer)
		
		


