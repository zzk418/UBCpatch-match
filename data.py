import torch
from torchvision import datasets, transforms
from dataloader import patchSet
from torch.utils.data import random_split, DataLoader

transform = transforms.Compose([    
        transforms.ToTensor(),
        transforms.Normalize(mean=128,std=160)
    ])
dataset = patchSet('./', transforms=transform)

train_set_size = int(len(dataset) * 0.4)
valid_set_size = int(len(dataset) * 0.1)
abandon_size = len(dataset) - (train_set_size + valid_set_size)
train_set, valid_set, abandon_set = random_split(dataset, [train_set_size, valid_set_size, abandon_size])

batch_size = 64
train_loader = DataLoader(
    train_set, batch_size=batch_size, pin_memory=True, drop_last=True)

vaild_loader = DataLoader(
    valid_set, batch_size=batch_size, pin_memory=True, drop_last=True) 

# train_transforms = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
#     ])


# test_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
#     ])

# trainset = datasets.CIFAR10(
#     root='./exp1/CIFAR/', train=True, download=True, transform=train_transforms)
# testset = datasets.CIFAR10(
#     root='./exp1/CIFAR/', train=False, download=True, transform=test_transforms)

# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size = 128, shuffle=True, num_workers=4)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size = 128, shuffle=True, num_workers=4)