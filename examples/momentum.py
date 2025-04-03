import numpy as np
import matplotlib.pyplot as plt

# 定义函数和其梯度
def func(x, y):
    return 2*x**2 + y**2

def grad_x(x, y):
    return 4*x

def grad_y(x, y):
    return 2*y

# 梯度下降参数
learning_rate = 0.5
momentum = 0
iterations = 10

# 初值
x_val, y_val = 1, 1
velocity_x, velocity_y = 0, 0  # 初始化动量

# 存储每一步的结果
x_history = [x_val]
y_history = [y_val]

# 梯度下降迭代
for i in range(iterations):
    grad_x_val = grad_x(x_val, y_val)
    grad_y_val = grad_y(x_val, y_val)

    # 更新动量
    velocity_x = momentum * velocity_x - learning_rate * grad_x_val
    velocity_y = momentum * velocity_y - learning_rate * grad_y_val

    # 更新位置
    x_val += velocity_x
    y_val += velocity_y

    # 存储当前结果
    x_history.append(x_val)
    y_history.append(y_val)

# 绘制等值线图
x = np.linspace(-1.5, 1.5, 30)
y = np.linspace(-1.5, 1.5, 30)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

plt.figure(figsize=(8, 6))

# 绘制等值线
contours = plt.contour(X, Y, Z, levels=np.arange(0, 8, 0.5), cmap='coolwarm')
plt.clabel(contours, inline=True, fontsize=8)

# 绘制梯度下降路径
x_history = np.array(x_history)
y_history = np.array(y_history)
plt.plot(x_history, y_history, marker='o', color='black', markersize=5, label='Gradient Descent Path')

plt.title(r'Gradient Descent on $f(\theta) = 2\theta_1^2 + \theta_2^2$')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.axis('equal')
plt.colorbar(contours, label='Function Value')
plt.legend(loc='lower right')

plt.show()
