import pytorch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

batch_size = 256
epoch = 30

data_transforms = {
    "train" : transforms.Compose([
        transforms.Resize([64, 64]),  
        transforms.RandomHorizontalFlip(),  # base model과 다르게 augmentation을 적용하는데, 
        transforms.RandomVerticalFlip(),    # pre_trained model이라 데이터 양을 더 늘린 것 같음(자료에 이유가 안 나와있다..) 
        transforms.RandomCrop(52),  # 이미지 일부를 52*52로 잘라내어씀
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # 정규화에 쓰일 각 채널의 평균값
                            [0.229, 0.224, 0.225])   # 정규화에 쓰일 각 채널의 표준편차값
    ]),
    "val" : transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(52),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
    
}

data_dir = path + "/splitted"

dataset = {x : ImageFolder(root = os.path.join(data_dir, x), transform = data_transforms[x]) for x in ["train", "val"]}
loader = {x : torch.utils.data.DataLoader(dataset[x], batch_size = batch_size, shuffle = True, num_workers = 4) for x in ["train", "val"]}
dataset_sizes = {x : len(dataset[x]) for x in ["train", "val"]}
class_names = dataset["train"].classes

from torchvision import models
resnet = models.resnet50(pretrained = True) # False가 되면 모델의 구조만 가져오고 초깃값은 랜덤 설정
num_ftrs = resnet.fc.in_features # fc는 모델의 마지막 layer를, in_features는 해당 층의 입력 채널 수 반환
resnet.fc = nn.Linear(num_ftrs, 33) # 마지막 fc층의 출력 채널을 클래스 수에 맞게 변환
resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(filter(lambda p : p.requires_grad, resnet.parameters()), lr=0.001)
# filter와 lambda를 통해 requires_grad=True인 layer만 파라미터 업데이트

from torch.optim import lr_scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)
# 에폭에 따라 lr변경 -> 7에폭마다 0.1씩 곱해짐

ct = 0
for child in resnet.children():  # model.children() -> 모델의 layer정보
  ct += 1
  if ct < 6:
    for param in child.parameters():
      param.requires_grad = False



def train_resnet(model, criterion, optimizer, scheduler, num_epochs = 25):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    
    for epoch in range(num_epochs):                           # 에폭마다 for문
        print(f"---------- epoch {epoch + 1} ----------")
        since = time.time()
        
        for phase in ["train", "val"]:                        # train과 val데이터 동시에-> for문인거 신경 안써도됨
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, labels in loader[phase]:             # 배치단위마다 for문
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    x, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == "train":
                        loss.backward() 
                        optimizer.step()       
                        
                running_loss += loss.item() * inputs.size(0)       # 교차엔트로피 계산 deafualt값이 mean이므로 각 데이터 마다의 손실 평균이 저장되있음
                                                                   # 따라서 배치 사이즈를 곱해줘 한 배치 사이즈의 loss 총합을 계산!
                running_corrects += torch.sum(preds == labels.data)  # -----------------여기까진 base model과 구조가 동일----------------------------
            if phase == "train":
                scheduler.step()
                l_r = [x["lr"] for x in optimizer_ft.param_groups]
                print("learning rate : ", l_r)
                 
            epoch_loss = running_loss/dataset_sizes[phase]          # 전체 데이터 loss합을 각 데이터셋 전체 크기로 나눠주어 loss계산
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print("{} Loss: {:4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        time_elapsed = time.time() - since
        print("Completed in {:.0f}m {:0f}s".format(time_elapsed // 60, time_elapsed % 60))
        
    print("Best val Acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    
    return model

model_resnet50 = train_resnet(resnet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epoch)
torch.save(model_resnet50, "resnet50.pt")