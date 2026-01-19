# Pytorch Tutorial

## 1. PyTorch 是什么

可以先把 PyTorch 理解成两部分：

1. **张量计算库**  
   有点像带 GPU 版的 NumPy。

   - NumPy: `ndarray` 在 CPU 上算
   - PyTorch: `Tensor` 可以在 CPU 或 GPU 上算
2. **自动求导 + 神经网络框架**  
   可以自动对计算过程求导数。  
   这对梯度下降训练神经网络非常关键。

一句话总结：

> PyTorch \= 支持自动求导和 GPU 加速的 "NumPy + 深度学习工具箱"。

---

## 2. 深度学习训练的三大要素

要训练一个网络，基本总是要先想清楚三件事：

1. **模型结构 (Model)**   
   输入是什么，输出是什么，中间有几层，每层怎么连接。
2. **损失函数 (Loss Function)**   
   用一个数值来衡量“模型预测得好不好”。  
   比如回归用均方误差，分类用交叉熵。
3. **优化器 (Optimizer)**   
   用来更新参数，让损失越来越小。  
   比如 SGD、Adam。

整体流程通常是：

```text
训练 (train) → 验证 (validation) → 测试 (test)
```

- 训练集: 用来更新参数
- 验证集: 用来调超参数、看是否过拟合
- 测试集: 只在最后评估一次

---

## 3. Dataset 和 DataLoader

你可以把它们想成：

- ​**Dataset**: 像一个“数组”，定义了数据“怎么按索引拿到一条”。
- ​**DataLoader**: 像一个“迭代器”，每次从 Dataset 取一批 (batch) 数据出来。

### 3.1 Dataset 要做什么

自定义 Dataset 只需要实现两个方法：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, file):
        # 在这里把数据读进来，存在内存或索引里
        self.data = [...]  

    def __getitem__(self, index):
        # 返回第 index 条样本，通常是 (x, y)
        return self.data[index]

    def __len__(self):
        # 返回总共有多少条数据
        return len(self.data)
```

所以可以总结为:

- ​`__len__` 告诉 PyTorch: “我有多少条数据”。
- ​`__getitem__` 告诉 PyTorch: “给定一个索引，我怎么拿出一条数据”。

### 3.2 DataLoader 要做什么

DataLoader 会把 Dataset 打包成一个可以用 `for` 循环迭代的对象。

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset=my_dataset,
    batch_size=5,
    shuffle=True,
)
```

它会帮你：

- 每次取 `batch_size`​ 条数据: `(x_batch, y_batch)`
- 训练时打乱顺序: `shuffle=True`
- 可以用多进程预取数据: `num_workers` 参数控制

使用方式：

```python
for x_batch, y_batch in dataloader:
    # 在这里把 batch 丢给模型
    pred = model(x_batch)
```

一句话总结：

> Dataset 定义“单条样本如何获取”，DataLoader 定义“如何按批次、按顺序（或随机）取样本”。

---

## 4. Tensor（张量）

可以先把 Tensor 当作“PyTorch 版的数组”，类似 NumPy 的 `ndarray`。

### 4.1 常见维度含义

- 1 维: 长度为 N 的向量，比如音频波形 `[N]`
- 2 维: 矩阵，比如灰度图 `[H, W]`
- 3 维: 彩色图像，通常用 `[C, H, W]` 格式

  - ​`C` 是通道数, 比如 RGB 就是 3
  - ​`H`​ 高，`W` 宽

在训练图片模型时常见的是 4 维 `[N, C, H, W]`:

- ​`N` 是 batch 大小
- 每个样本是一个 `[C, H, W]` 的图像

### 4.2 Tensor 基本创建与操作

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])   # 1D
A = torch.ones(2, 3)                # 2x3 全 1
B = torch.zeros(3, 4)               # 3x4 全 0

s = A.sum()
m = A.mean()
```

和 NumPy 的感觉很像。

常见操作：

- ​`.sum()`: 求和
- ​`.mean()`: 求平均
- ​`.transpose(dim0, dim1)`: 转置维度
- ​`.squeeze()`: 去掉长度为 1 的维度
- ​`.unsqueeze(dim)`: 在指定位置增加一个维度
- ​`torch.cat([t1, t2], dim=...)`: 在某个维度上拼接

### 4.3 Tensor 与 NumPy 转换

```python
import numpy as np
import torch

a = np.array([1, 2, 3])
t = torch.from_numpy(a)        # numpy -> tensor

b = t.numpy()                  # tensor -> numpy
```

几乎所有的基础操作名字都和 NumPy 类似: `reshape`​, `squeeze`, 等等。

### 4.4 Tensor 的设备 (device)

Tensor 可以放在 CPU 或 GPU 上：

```python
x = torch.randn(3, 4)

if torch.cuda.is_available():
    x = x.to("cuda")   # 把张量拷贝到 GPU 上

x = x.to("cpu")        # 再拷回 CPU
```

在深度学习中，关键是:

- 模型和数据必须在同一个设备上。
- GPU 能显著加速大规模矩阵运算。

### 4.5 自动求导 autograd（非常关键）

autograd 能帮你自动算导数。

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x.pow(2).sum()    # y = 2^2 + 3^2 = 13

y.backward()          # 对 x 求导

print(x.grad)         # dy/dx = [2*2, 2*3] = [4, 6]
```

你只需要:

1. 创建张量时设 `requires_grad=True`
2. 用这些张量构建一些计算
3. 在标量 loss 上调用 `backward()`
4. PyTorch 会自动把每个参数的梯度写到 `param.grad` 里

这一点是训练神经网络的根基。

---

## 5. 神经网络模型 (nn.Module)

PyTorch 中所有网络结构都继承自 `nn.Module`。

### 5.1 核心设计思想

- 在 `__init__` 中定义网络的“层”和“模块”，即需要学习的参数。
- 在 `forward` 中定义前向计算，也就是数据如何流过这些层。

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
```

这里:

- 输入是长度为 10 的向量
- 中间有 32 维隐藏层，加 Sigmoid 激活
- 最后输出 1 维, 可以用来做回归或二分类的 "logits"

### 5.2 常见层

- 全连接层: `nn.Linear(in_features, out_features)`
- 卷积层: `nn.Conv2d(in_channels, out_channels, kernel_size)`
- 激活函数: `nn.ReLU()`​, `nn.Sigmoid()`​, `nn.Tanh()`
- 归一化层: `nn.BatchNorm2d(num_features)`
- 池化层: `nn.MaxPool2d(kernel_size)`

### 5.3 Linear 层的计算

​`nn.Linear(in, out)` 本质上就是:

```text
y = W x + b
```

- ​`W`​ 的形状是 `[out, in]`
- ​`b`​ 的形状是 `[out]`
- 对每个样本，将输入向量 `[in]`​ 变换到输出向量 `[out]`

---

## 6. 损失函数 (Loss Functions)

损失函数把“预测值”和“真实标签”变成一个“误差标量”。

常见设计:

- ​**回归任务**:

  - ​`nn.MSELoss()` 均方误差 (Mean Squared Error)
- ​**分类任务**:

  - ​`nn.CrossEntropyLoss()` 适用于多分类

    - 输入: 未经 softmax 的 logits (shape: [N, num\_classes])
    - 标签: 每个样本是一个类别编号 (shape: [N])

用法例子:

```python
criterion = nn.MSELoss()
loss = criterion(pred, target)
```

---

## 7. 优化器 (Optimizer)

优化器负责“根据梯度更新参数”。

最常见例子:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

每一步的标准流程是:

```python
optimizer.zero_grad()   # 清空上一次的梯度
loss.backward()         # 自动反向传播, 计算新的梯度
optimizer.step()        # 用梯度更新参数
```

注意:

- 不调用 `zero_grad` 的话, 梯度会累积。
- 这有时是有意为之 (梯度累积), 但初学时一般每 step 都要清空。

---

## 8. 训练流程 (核心框架)

这是最标准的 PyTorch 训练循环结构:

```python
for epoch in range(num_epochs):
    model.train()   # 让模型处于训练模式
    for x_batch, y_batch in dataloader:
        # 1. 把数据放到设备上 (CPU or GPU)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 2. 前向传播
        pred = model(x_batch)

        # 3. 计算损失
        loss = criterion(pred, y_batch)

        # 4. 清空梯度
        optimizer.zero_grad()

        # 5. 反向传播
        loss.backward()

        # 6. 更新参数
        optimizer.step()
```

每一层逻辑基本是:

1. 从 DataLoader 拿一批数据
2. 模型计算预测
3. 通过损失函数计算误差
4. 通过 autograd 得到梯度
5. 优化器根据梯度更新参数

---

## 9. 验证 / 测试流程

验证和测试时有两个关键点:

1. 不需要计算梯度, 所以用 `torch.no_grad()`
2. 某些层在训练和测试时行为不同, 比如 Dropout 和 BatchNorm  
   所以需要用 `model.eval()` 切换到评估模式。

典型写法:

```python
model.eval()           # 切到评估模式
total_loss = 0

with torch.no_grad():  # 关闭梯度计算
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item()
```

验证完记得切回 `model.train()` 再继续训练。

---

## 10. 模型保存与加载

训练完通常要把模型参数保存起来:

```python
torch.save(model.state_dict(), "model.pth")
```

之后想重新使用模型:

```python
model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.to(device)
model.eval()
```

这里只保存了参数 (`state_dict`), 而不是整个对象。  
优点是更灵活, 兼容性更好, 通常也更推荐。
