# VLL_Task1
## Goal
Acheiving accuracy 75% at CIFAR-100 dataset by using ResNet-50</br>
The implementation should be done from scratch

## Model
ResNet-50, pretrained with ImageNet

## Experiment Details
 - learning rate: 0.001
 - epoch: 30
 - batch size = 64
 - optimizer: Adam
 - scheduler: StepLR(step_size=5, gamma=0.5)
 - As a warm-up strategy, StepLR was applied once the test accuracy exceeded 70%

## GPU resource
1 NVIDIA GeForce RTX 3090 GPU

## Result
Best Accuracy: 80.1%

## Limitation
Using test accuracy to determine when to end the warm-up phase can be considered cheating
