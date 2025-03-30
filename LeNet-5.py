"""
author:Bruce Zhao
date: 2025/3/30 15:23
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义LeNet-5模型结构
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 特征提取器（卷积层）
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # 输入通道1（灰度图），输出通道6
            nn.Tanh(),  # 原论文使用Sigmoid，此处用Tanh更高效
            nn.AvgPool2d(kernel_size=2, stride=2),  # 池化层：6@28x28 → 6@14x14
            nn.Conv2d(6, 16, kernel_size=5),  # 输出通道16
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)  # 16@10x10 → 16@5x5
        )
        # 分类器（全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 展平后输入维度16*5*5=400
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)  # 输出类别数
        )

    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = torch.flatten(x, 1)  # 展平为向量（保留batch维度）
        x = self.classifier(x)  # 分类决策
        return x


# 数据预处理（以MNIST为例）
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # LeNet-5输入要求32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值和标准差
])

# 加载数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
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

    print(f"Epoch {epoch + 1}/10 | Train Loss: {running_loss / len(train_dataset):.4f} | "
          f"Test Acc: {100 * correct / total:.2f}%")