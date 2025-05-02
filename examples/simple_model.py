import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x

x = np.linspace(-5, 5, 200)
y = np.sin(x * 2)

x = torch.from_numpy(x).float().reshape(-1, 1)
y = torch.from_numpy(y).float().reshape(-1, 1)
    
criterion: nn.Module = nn.MSELoss()
model = SimpleModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

loss_record = []
outputs = []

model.train()

# 用于显示进度条
pbar = tqdm(range(2000))
for epoch in pbar:
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    loss_record.append(loss.detach().item())
    optimizer.step()
    
    if epoch % 100 == 0:
        # 记录训练过程中的输出
        outputs.append(output.detach().numpy())
    
    pbar.set_description(f"Loss: {loss.item():.4f}")

# 记录最后的输出
outputs.append(output.detach().numpy())

# 将所有输出绘制在同一张图上，由浅到深
size_output = len(outputs)
normalize = Normalize(vmin=-1, vmax=size_output-1)
cmap = cm.coolwarm

for i, output in enumerate(outputs):
    t = normalize(i) ** 0.7
    color = cmap(t)
    plt.plot(x.detach().numpy(), output, color=color, alpha=t)
    
plt.plot(
    x.detach().numpy(), y.detach().numpy(), color="#5a5", linestyle="--", label='True Output'
)
plt.title('Model Output vs True Output')

plt.show()

plt.plot(loss_record)
plt.title('Loss Record')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()