import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
import random
# from torchvision.models import resnet18
from torchvision.models.resnet import ResNet, BasicBlock

# 定义 LeNet 网络
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_acc_list = []
test_acc_list = []
train_acc_var_list = []
test_acc_var_list = []
batch_count_list = []

num_samples = 50

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # 随机选取100个样本
        indices = random.sample(range(len(loader.dataset)), num_samples)
        subset = torch.utils.data.Subset(loader.dataset, indices)
        subset_loader = DataLoader(subset, batch_size=num_samples, shuffle=False)
        for images, labels in subset_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

total_batches = 0

model.train()

train_acc_prev = 0.0
test_acc_prev = 0.0
train_acc_var = 0.0
test_acc_var = 0.0

prev_weight = 0.0

for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    total_batches += 1

    if total_batches % 5 == 0:
        train_acc = evaluate(train_loader)
        test_acc = evaluate(test_loader)
        # 为了图像的美观，使用指数衰减平均做平滑处理
        prev_weight *= 0.95
        prev_weight *= 0.95
        c0 = prev_weight / (1 + prev_weight)
        c1 = 1 - c0
        train_acc_avg = c0 * train_acc_prev + c1 * train_acc
        test_acc_avg = c0 * test_acc_prev + c1 * test_acc
        train_acc_var = c0**2 * train_acc_var + c1**2 * train_acc_avg * (1 - train_acc_avg) / num_samples
        test_acc_var = c0**2 * test_acc_var + c1**2 * test_acc_avg * (1 - test_acc_avg) / num_samples
        prev_weight += 1.0
        prev_weight += 1.0
        train_acc_prev = train_acc_avg
        test_acc_prev = test_acc_avg
        
        train_acc_list.append(train_acc_avg)
        test_acc_list.append(test_acc_avg) 
        train_acc_var_list.append(train_acc_var)
        test_acc_var_list.append(test_acc_var)
        
        batch_count_list.append(total_batches)
        # print(f"Batch {total_batches}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

plt.rcParams["font.size"] = 20

batch_count_list = np.array(batch_count_list)
train_acc_list = np.array(train_acc_list)
test_acc_list = np.array(test_acc_list)

train_acc_var_list = np.array(train_acc_var_list)
test_acc_var_list = np.array(test_acc_var_list)

train_ci = np.sqrt(train_acc_var_list) * 1.96
test_ci = np.sqrt(test_acc_var_list) * 1.96

# 绘制准确率曲线
plt.figure(figsize=(10,6))
plt.plot(batch_count_list, train_acc_list, label='训练集准确率')
plt.plot(batch_count_list, test_acc_list, label='测试集准确率')
plt.fill_between(batch_count_list, train_acc_list - train_ci, train_acc_list + train_ci, alpha=0.2, edgecolor='none')
plt.fill_between(batch_count_list, test_acc_list - test_ci, test_acc_list + test_ci, alpha=0.2, edgecolor='none')

plt.xlabel('数据批次(Batch)数量')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)
plt.savefig('img/lenet5_mnist.png')
plt.close()