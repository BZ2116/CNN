"""
author:Bruce Zhao
date: 2025/3/30 15:23
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义简化版AlexNet（Alex-5）
class Alex5(nn.Module):
    def __init__(self, num_classes=10):
        super(Alex5, self).__init__()
        # 特征提取器（卷积层+池化）
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # 输入1通道，输出64通道
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 池化后尺寸：64@7x7

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 192@3x3
        )
        # 分类器（全连接层）
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(192 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = torch.flatten(x, 1)  # 展平为向量 [batch, 192*3*3]
        x = self.classifier(x)  # 分类决策
        return x


# 数据预处理（适配AlexNet输入）
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 原始MNIST是28x28，调整为32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
])

# 加载数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Alex5(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环（10个epoch）
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch + 1}/10 | Loss: {running_loss / len(train_dataset):.4f} | "
          f"Test Acc: {100 * correct / total:.2f}%")