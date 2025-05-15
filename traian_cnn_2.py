from torchvision import transforms
from PIL import ImageFilter
from ultralytics import YOLO
import cv2
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
# 训练集增强（强化边缘和纹理）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomApply(
        [transforms.Lambda(lambda x: x.filter(ImageFilter.EDGE_ENHANCE))],
        p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 验证集预处理（仅标准化）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import torch
from torch import nn, optim
from torchvision import models, datasets
from torch.utils.data import DataLoader


# 自定义数据集加载
class DefectDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):
        self.classes = ['dong', 'que', 'normal']
        self.image_paths = []
        self.labels = []
        for label_idx, label in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(label_idx)
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.image_paths)


# 加载数据
train_dataset = DefectDataset('./cnn_dataset/train', transform=train_transform)
val_dataset = DefectDataset('./cnn_dataset/val', transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 定义模型（基于ResNet18微调）
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # 三分类输出
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(15):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # 验证步骤
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch+1}, Val Acc: {100*correct/total:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'defect_cnn.pth')