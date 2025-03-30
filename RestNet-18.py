"""
author:Bruce Zhao
date: 2025/3/30 15:23
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models


# 自定义ResNet-18（适配单通道输入）
class MNIST_ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 修改原始ResNet的输入层（原为3通道）
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # 移除第一个maxpool层（避免尺寸过小）
        self.resnet.fc = nn.Linear(512, num_classes)  # 修改分类头

    def forward(self, x):
        return self.resnet(x)


# 数据预处理（适配ResNet）
transform = transforms.Compose([
    transforms.Resize(32),  # 放大到32x32（适应ResNet下采样）
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
])

# 数据集加载
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

# 数据加载器（增大batch_size）
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环（5个epoch即可收敛）
for epoch in range(5):
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

    print(f"Epoch {epoch + 1}/5 | Loss: {running_loss / len(train_dataset):.4f} | "
          f"Test Acc: {100 * correct / total:.2f}%")