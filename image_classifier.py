"""
DataLoader
__init__() : 클래스 생성자로 서 데이터에 대한 transform(데이터형 변환, augmentation등)을 설정하고
            데이터를 읽기위한 기초적인 초기화 작업들을 수행하도록 정의

__getitem()__ : 본 class에 존재하는 데이터를 읽고 반환하는 함수, 따라서 본인이 어떤 작업을 수행하는지에 
                따라 반환하는 값들이 달라질수 있다. 
                ex) image classification -> 이미지와 해당 이미지가 어떤 클래스에 속하는지 대한 값 반환

__len()__ : 데이터셋의 크기를 반환하는 함수이다.
            ex) image classifier -> 우리가 가진 이미지와 개수가 곧 데이터 셋의 크기, 즉 40장을 가졌으면 
                __len()__함수는 40을 반환

__next__()[1]/[2]/[3] : [0] -> base path , [1] -> 하위 디렉토리, [2] -> 파일이 리스트로 반환
"""
import os
import numpy as np
import time
import copy
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CustomImageDataset(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length


hyper_param_epoch = 20
hyper_param_batch = 8

transforms_train = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor()])

transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor()])

train_data_set = CustomImageDataset(data_set_path="animal/data/train", transforms=transforms_train)
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

test_data_set = CustomImageDataset(data_set_path="animal/data/test", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

if not (train_data_set.num_classes == test_data_set.num_classes):
   print("error: Numbers of class in training set and test set are not equal")
   exit()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = train_data_set.num_classes

# Model Train
def train(model, train_loader, optimizer, epoch):
    model.train()
    for i_batch, item in enumerate(train_loader):
        images = item['image'].to(device)
        labels = item['label'].to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i_batch + 1) % hyper_param_batch == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch, hyper_param_epoch, loss.item()))
# Model Test
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for item in test_loader:
            images = item['image'].to(device)
            labels = item['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
    
    #test_loss /= len(test_loader.dataset)
    test_loss = F.cross_entropy(outputs, labels)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False # 현재 weight들이 변하지 않도록 frozen 시키기
num_ftrs = model.fc.in_features 
model.fc = nn.Linear(num_ftrs, 2) # 해당 부분만 업데이트

model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("[{}] Test Loss: {:.4f}, accuracy: {:.2f}%\n".format(epoch, test_loss, test_accuracy))