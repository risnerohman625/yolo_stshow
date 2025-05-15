import torch
from torch import nn, optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.metrics import classification_report
import numpy as np
from torchvision.models.resnet import ResNet18_Weights

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class DefectDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.classes = ['dong', 'que', 'normal']  # 根据实际类别调整
        self.image_paths = []
        self.labels = []
        for label_idx, label_name in enumerate(self.classes):
            label_dir = os.path.join(root_dir, label_name)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


if __name__ == '__main__':
    # 实例化数据集
    train_dataset = DefectDataset("./cnn_dataset/train",
                                  transform=train_transform)
    val_dataset = DefectDataset("./cnn_dataset/val", transform=val_transform)

    # 定义DataLoader
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    # 加载预训练模型并修改输出层
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 三分类：dong/que/normal

    # 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # 验证步骤
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {running_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"Val Acc: {100 * correct / total:.2f}%")

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted)

    print(
        classification_report(y_true,
                              y_pred,
                              target_names=['dong', 'que', 'normal']))
    torch.save(model.state_dict(), "cnn_defect_classifier.pth")
