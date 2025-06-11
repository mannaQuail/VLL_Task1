import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torchvision import models
import os


def train_loop(dataloader, model, loss_func, optimizer):
	batch_num = len(dataloader)

	images, _ = next(iter(train_loader))
	batch_size = images.shape[0]
	
	total_size = batch_num*batch_size

	model.train()

	for batch, (inputs, labels) in enumerate(dataloader):
		inputs = inputs.to(device)
		labels = labels.to(device)

		output = model(inputs)
		prob, preds = torch.max(output, dim=1)
		loss = loss_func(output, labels)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		current_batch_num = (batch+1)*batch_size

		if (batch+1)%30 == 0:
			print(f"loss: {loss:>7f}, [{current_batch_num:>5d}/{total_size:>5d}]")

def test_loop(dataloader, model, loss_func):
	batch_num = len(dataloader)

	images, _ = next(iter(train_loader))
	batch_size = images.shape[0]
	
	total_size = batch_num*batch_size

	test_loss = 0.0
	test_acc = 0.0

	model.eval()

	with torch.no_grad():
		for inputs, labels in dataloader:
			inputs = inputs.to(device)
			labels = labels.to(device)

			output = model(inputs)

			test_loss += loss_func(output, labels)*batch_size
			test_acc += (torch.argmax(output, dim=1)==labels).type(torch.float).sum().item()

	test_loss /= total_size
	test_acc /= total_size

	print(f"test loss: {test_loss:>7f}, test acc: {test_acc:>3f}\n")


learning_rate = 0.001
epoch_num = 30
batch_size = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리 설정
transform = transforms.Compose([
	transforms.Resize(224),
    transforms.ToTensor(),  # 0~255 → 0.0~1.0
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # CIFAR-100 평균/표준편차
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
print(f"training shape: {images.shape}, label shape: {labels.shape}")

resnet = models.resnet50(pretrained = True)
resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=100)

resnet = resnet.to(device)

CE_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)

print("check")

for epoch in range(epoch_num):
	loss_avg = 0.0
	acc_avg = 0.0
	print(f"------------Epoch:{epoch}------------")

	train_loop(train_loader, resnet, CE_loss, optimizer)
	test_loop(test_loader, resnet, CE_loss)