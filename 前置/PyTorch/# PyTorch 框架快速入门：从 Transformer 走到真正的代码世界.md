# PyTorch 框架快速入门：从 Transformer 走到真正的代码世界

这一章和前面的 Transformer 章节有一个非常明显的区别。

前面我们主要在研究：

> **Transformer 的数学和模型结构到底在干嘛？**

比如你已经学过：

- hidden vector；
- Q / K / V；
- Multi-Head Attention；
- FFN / SwiGLU；
- RoPE；
- RMSNorm；
- Residual；
- Decoder Block；
- KV Cache；
- Prefill / Decode；
- FLOPs 和显存。

而从这一章开始，我们要回答另一个问题：

> **这些东西在真正的深度学习代码里，到底是怎么组织、怎么训练、怎么放到 GPU 上跑的？**

PyTorch 就是干这个的。

这一篇原文的主线其实很清楚：

```text
Tensor
  ↓
Autograd
  ↓
nn.Module
  ↓
Training Loop
  ↓
GPU / Mixed Precision
  ↓
Profiler
```

你可以把它理解成：

```text
Tensor
= 数据本身

Autograd
= 自动帮你算梯度

nn.Module
= 把参数和计算组织成模型

Training Loop
= 真正训练模型

GPU
= 把计算搬到 GPU 上

Profiler
= 看看到底哪里慢
```

对于 AI Infra 来说，这一章非常重要。

因为以后你看：

```text
Megatron-LM
DeepSpeed
FSDP
vLLM
PyTorch Distributed
torch.compile
CUDA Extension
```

底下基本都逃不开这些 PyTorch 基础。

---

# 一、Tensor 到底是什么？

先从最基础的开始。

PyTorch 里最核心的数据结构就是：

```python
torch.Tensor
```

你可以把 Tensor 理解成：

> **带有 shape、dtype、device 等信息的多维数组。**

比如：

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
```

这是一个一维 Tensor。

shape：

```python
x.shape
```

得到：

```text
torch.Size([3])
```

---

## 二、Tensor 和你前面学的东西怎么对应？

其实你前面已经一直在和 Tensor 打交道，只不过以前是数学写法。

比如 Transformer 输入：

$$
X \in \mathbb{R}^{B \times S \times d}
$$

到了 PyTorch 里，就是：

```python
x.shape == (B, S, d)
```

例如：

```python
x = torch.randn(2, 128, 4096)
```

表示：

```text
B = 2
S = 128
d = 4096
```

也就是：

```text
2 个 sequence
每个 sequence 128 个 token
每个 token 是 4096 维 hidden vector
```

所以以后你可以建立一个非常重要的对应关系：

```text
数学：
X ∈ R^(B × S × d)

PyTorch：
x.shape == (B, S, d)
```

实际上 AI Infra 代码里，你会天天在追：

```text
这个 Tensor 现在 shape 是什么？
dtype 是什么？
在哪张 GPU？
内存连续不连续？
```

---

# 三、Tensor 的创建

原文列了一些最常用的。

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])

zeros = torch.zeros(3, 4)

ones = torch.ones(2, 3, 4)

rand_normal = torch.randn(3, 4)

seq = torch.arange(0, 10, 2)
```

其中你以后特别常见的是：

```python
torch.randn(...)
```

比如：

```python
x = torch.randn(B, S, d)
```

经常被用来：

- 写测试；
- 检查 shape；
- benchmark kernel；
- 写最小复现代码。

---

# 四、zeros_like 和 ones_like 为什么有用？

假设：

```python
x.shape == (2, 128, 4096)
```

如果你想创建一个和 x：

- shape 一样；
- device 一样；
- dtype 一样；

的全零 Tensor，可以：

```python
y = torch.zeros_like(x)
```

而不是重新手写：

```python
torch.zeros(2, 128, 4096)
```

这在 CUDA / AI Infra 代码里很常见。

因为真实 Tensor 可能是：

```text
shape = (B, H, S, Dh)
device = cuda:3
dtype = bfloat16
```

这时候：

```python
torch.zeros_like(x)
```

就很方便。

---

# 五、这一章第一个真正重要的东西：shape 操作

原文讲：

```text
view
reshape
permute
squeeze
unsqueeze
```

这几个你一定要熟。

尤其是：

```text
reshape
view
permute
```

因为 Transformer 代码里全是它们。

---

# 六、view / reshape：改变 shape

比如：

```python
x = torch.arange(12)
```

现在：

```text
shape = (12,)
```

我们可以：

```python
x.view(3, 4)
```

变成：

```text
shape = (3, 4)
```

数据没有变，只是重新解释这 12 个元素。

数学上就是：

$$
12 = 3 \times 4
$$

---

## 七、-1 是什么意思？

比如：

```python
x.view(-1, 4)
```

因为一共有：

$$
12
$$

个元素，

最后一维指定：

$$
4
$$

所以 PyTorch 自动推断第一维：

$$
\frac{12}{4} = 3
$$

所以得到：

```text
(3, 4)
```

因此：

```python
-1
```

在 reshape / view 里通常表示：

> **这一维你帮我自动算。**

---

# 八、这和 Multi-Head Attention 有什么关系？

这个你现在应该非常熟悉。

Attention 一开始：

$$
Q \in \mathbb{R}^{B \times S \times d}
$$

比如：

```text
(B, S, 4096)
```

假设：

```text
num_heads = 32
head_dim = 128
```

因为：

$$
4096 = 32 \times 128
$$

我们会把：

```text
(B, S, 4096)
```

reshape 成：

```text
(B, S, 32, 128)
```

PyTorch 代码很可能就是：

```python
q = q.view(B, S, num_heads, head_dim)
```

或者：

```python
q = q.reshape(B, S, num_heads, head_dim)
```

所以你前面学的 Multi-Head：

$$
d = H \times D_h
$$

现在终于对应到真实代码了。

---

# 九、permute：真正需要认真理解

假设：

```python
q.shape
```

是：

```text
(B, S, H, Dh)
```

但 Attention 计算通常喜欢：

```text
(B, H, S, Dh)
```

怎么办？

不是 reshape。

而是：

```python
q = q.permute(0, 2, 1, 3)
```

意思是：

原来的轴：

```text
0  1  2  3
B  S  H  D
```

现在变成：

```text
0  2  1  3
B  H  S  D
```

所以：

```python
permute(0, 2, 1, 3)
```

本质是：

> **重新排列维度的顺序。**

---

# 十、reshape 和 permute 千万别混

比如：

```text
(B, S, H, D)
```

变成：

```text
(B, H, S, D)
```

这不是简单地“把数字重新分组”。

而是：

> S 维和 H 维真的交换位置。

所以要：

```python
permute
```

而不是单纯：

```python
reshape
```

你可以记：

```text
reshape / view：
改变每一维怎么拆

permute：
改变维度的排列顺序
```

---

# 十一、这就是 Transformer 里最经典的一条 shape 链

你前面已经学过：

$$
(B,S,d)
$$

先 reshape：

$$
(B,S,H,D_h)
$$

再 permute：

$$
(B,H,S,D_h)
$$

然后：

$$
QK^T
$$

得到：

$$
(B,H,S,S)
$$

所以 PyTorch 的 shape 操作不是无聊的 API。

它其实就是：

> **把你前面学的数学 shape 真正翻译成程序。**

---

# 十二、view 和 reshape 有什么区别？

原文给了一个入门版区别：

```text
view：
要求底层内存布局满足要求

reshape：
更灵活，如果必要可以复制数据
```

你现在先这样理解就行。

更重要的是记住：

```python
permute()
```

之后 Tensor 的逻辑 shape 虽然变了，但底层内存布局可能变得不连续。

因此后面你会经常看到：

```python
x = x.permute(...).contiguous()
```

这和后面 CUDA、Memory Layout、Stride 会直接连起来。

现在先建立这个意识：

$$
\text{Tensor Shape}
\neq
\text{Tensor 在内存中的物理排列}
$$

这一点以后非常重要。

---

# 十三、squeeze 和 unsqueeze

假设：

```python
x.shape == (1, 3, 1, 4)
```

那么：

```python
x.squeeze()
```

会把 size 为 1 的维度去掉：

```text
(3, 4)
```

反过来：

```python
x.shape == (3, 4)
```

调用：

```python
x.unsqueeze(0)
```

得到：

```text
(1, 3, 4)
```

也就是插入一个新的维度。

你以后会看到很多类似：

```python
mask.unsqueeze(1)
```

或者：

```python
scale.unsqueeze(-1)
```

通常都是为了：

$$
\text{Broadcasting}
$$

也就是广播。

---

# 十四、第二个非常重要的属性：device

Tensor 不只是有 shape。

它还有：

```python
x.device
```

比如：

```text
cpu
```

或者：

```text
cuda:0
```

假设：

```python
x = torch.randn(3, 4)
```

默认通常在 CPU。

搬到 GPU：

```python
x = x.to("cuda")
```

或者：

```python
x = x.cuda()
```

---

# 十五、为什么 AI Infra 特别在乎 CPU → GPU 搬运？

因为 CPU RAM 和 GPU HBM 是不同的内存。

CPU 数据要送给 GPU，一般需要经过：

```text
CPU Memory
    ↓
PCIe / NVLink 等互联
    ↓
GPU HBM
```

这个过程不是免费的。

所以如果代码写成：

```python
for ...:
    x = x.cpu()
    ...
    x = x.cuda()
```

不断来回搬，

性能很可能直接炸掉。

你前面刚刚学过：

$$
\text{Memory Bound}
$$

这里已经开始真正碰到：

> **数据搬运本身就是性能成本。**

所以 AI Infra 里一个非常重要的思想就是：

$$
\text{能不搬的数据，就别搬}
$$

---

# 十六、为什么最好直接在 GPU 创建 Tensor？

比如：

```python
x = torch.randn(3, 4)
x = x.cuda()
```

相当于：

```text
先 CPU 创建
↓
再搬 GPU
```

如果本来就只需要 GPU Tensor，可以直接：

```python
x = torch.randn(3, 4, device="cuda")
```

也就是：

```text
直接在 GPU 上创建
```

省掉一次搬运。

---

# 十七、dtype 也非常重要

Tensor 还有：

```python
x.dtype
```

常见：

```text
float32
float16
bfloat16
```

原文重点介绍了这三个。

每个元素占用大概：

```text
FP32   → 4 Bytes
FP16   → 2 Bytes
BF16   → 2 Bytes
```

所以如果有：

$$
N
$$

个元素，

显存大概就是：

$$
\text{Memory}
=
N \times \text{Bytes Per Element}
$$

这个你已经很熟了。

---

# 十八、为什么 BF16 现在这么常见？

FP16 和 BF16 都是：

$$
16\text{ bit}
$$

也就是：

$$
2\text{ Bytes}
$$

但它们的 exponent / mantissa 分配不一样。

你现在最需要记：

```text
FP16：
精度不错
动态范围比较小
更容易 overflow / underflow

BF16：
有效精度比 FP16 粗一些
但是动态范围接近 FP32
```

所以大模型训练里 BF16 很常见。

你后面学 GPU 时会再详细研究。

---

# 十九、到这里 Tensor 可以总结成四个核心属性

以后看到一个 Tensor，你脑子里应该下意识问：

```text
1. shape 是什么？

2. dtype 是什么？

3. device 在哪里？

4. 内存 layout / contiguous 情况怎么样？
```

比如：

```text
Q:
shape  = (B, H, S, Dh)
dtype  = bf16
device = cuda:0
```

这已经开始很 AI Infra 了。

---

# 二十、第二大部分：Autograd

你前面已经学过 Backprop。

数学上你知道：

$$
L
\rightarrow
\frac{\partial L}{\partial W}
$$

然后优化器根据梯度更新：

$$
W
$$

问题是：

> 真实神经网络几十亿个参数，难道我要自己手推每个导数？

当然不会。

PyTorch 有一个系统：

```text
autograd
```

自动帮你算。

---

# 二十一、Autograd 最核心的一句话

$$
\text{Forward 时记录计算关系}
$$

然后：

$$
\text{Backward 时沿计算图反向计算梯度}
$$

这就是 Autograd。

---

# 二十二、计算图是什么？

比如：

```python
x = torch.tensor([2.0], requires_grad=True)

y = x * 3

z = y ** 2
```

数学上：

$$
y = 3x
$$

$$
z = y^2
$$

所以计算关系：

```text
x
↓ ×3
y
↓ square
z
```

这就是计算图。

PyTorch 在 forward 的过程中，知道：

```text
y 是怎么由 x 来的
z 是怎么由 y 来的
```

所以当你：

```python
z.backward()
```

它可以从：

$$
z
$$

反推到：

$$
x
$$

---

# 二十三、这就是你前面学的 Chain Rule

比如：

$$
z=y^2
$$

$$
y=3x
$$

那么：

$$
\frac{dz}{dx}
=
\frac{dz}{dy}
\frac{dy}{dx}
$$

PyTorch 干的就是：

> 自动执行这整条 Chain Rule。

所以：

```python
z.backward()
```

背后就是反向传播。

---

# 二十四、requires_grad=True 是什么意思？

比如：

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
```

意思不是：

> x 现在已经有梯度。

而是：

> **请 PyTorch 跟踪和 x 有关的计算，为后面求梯度做准备。**

比如：

```python
y = x * 3
z = y.sum()

z.backward()
```

那么：

```python
x.grad
```

会得到：

```text
[3, 3]
```

因为：

$$
z = 3x_1 + 3x_2
$$

所以：

$$
\frac{\partial z}{\partial x_1}=3
$$

$$
\frac{\partial z}{\partial x_2}=3
$$

---

# 二十五、x.grad 到底是什么？

你前面学过：

$$
\nabla_x L
$$

在 PyTorch 里，就是：

```python
x.grad
```

所以一个非常直接的对应关系：

```text
数学：

∂L / ∂W


PyTorch：

W.grad
```

这特别重要。

---

# 二十六、为什么训练时参数会自动 requires_grad=True？

假设：

```python
linear = nn.Linear(4096, 4096)
```

里面有：

```text
weight
bias
```

这些属于模型的可训练参数。

PyTorch 会把它们注册成：

```python
nn.Parameter
```

而 Parameter 默认会参与梯度计算。

所以后面：

```python
loss.backward()
```

以后，

就可以得到：

```python
linear.weight.grad
```

也就是：

$$
\frac{\partial L}{\partial W}
$$

---

# 二十七、这和你之前学的矩阵梯度正好对应

比如 Linear：

$$
Y=XW
$$

假设 upstream gradient：

$$
G=\frac{\partial L}{\partial Y}
$$

你之前已经推过：

$$
\frac{\partial L}{\partial X}=GW^T
$$

以及：

$$
\frac{\partial L}{\partial W}=X^TG
$$

在 PyTorch 里：

```python
loss.backward()
```

Autograd 就会自动执行这些东西。

你不需要手写：

```python
dW = X.T @ G
```

但从 AI Infra 角度：

> **你最好知道它底层大概在算什么。**

因为以后你会发现：

```text
Backward 里也有很多 GEMM
```

甚至训练 FLOPs 为什么大约是 forward 的数倍，也和这里直接相关。

---

# 二十八、一个很重要的 PyTorch 特性：动态计算图

PyTorch 的计算图是：

```text
Define-by-Run
```

也就是说：

> 程序跑到哪，图就动态构建到哪。

所以可以写：

```python
if condition:
    y = f(x)
else:
    y = g(x)
```

甚至：

```python
for ...:
```

这种 Python 控制流。

PyTorch 会按照这一次真正执行过的路径建立计算图。

---

# 二十九、为什么 backward 后计算图通常会释放？

因为训练时计算图里会保存大量中间信息。

如果每一步都永久留着：

> 显存很快就炸了。

所以通常一次：

```python
loss.backward()
```

之后，这次 forward 对应的图会被释放。

这和你后面要学的：

```text
activation memory
activation checkpointing
```

其实是直接相关的。

---

# 三十、梯度为什么会累加？

这个是 PyTorch 新手特别容易踩的坑。

假设：

```python
x = torch.tensor([1.0], requires_grad=True)

(x * 2).sum().backward()
```

得到：

```python
x.grad == 2
```

然后你又：

```python
(x * 3).sum().backward()
```

你可能以为：

```text
x.grad = 3
```

结果是：

```text
x.grad = 5
```

因为：

$$
2+3=5
$$

PyTorch 默认是：

$$
\text{grad}_{new}
=
\text{grad}_{old}
+
\text{new gradient}
$$

也就是说：

> **梯度默认累加，不会自动覆盖。**

---

# 三十一、为什么 PyTorch 要故意这么设计？

因为梯度累积本身很有用。

比如你 GPU 显存不够。

原本想用：

```text
Batch Size = 1024
```

但 GPU 最多只能放：

```text
Batch Size = 256
```

那么可以：

```text
第 1 个 mini batch：算 gradient
第 2 个 mini batch：继续累加
第 3 个 mini batch：继续累加
第 4 个 mini batch：继续累加

然后再 optimizer.step()
```

于是相当于模拟更大的 batch。

所以默认 gradient accumulation 是有意设计的。

---

# 三十二、正常训练为什么要 zero_grad？

标准训练通常每一个 step：

```python
optimizer.zero_grad()

output = model(x)

loss = criterion(output, y)

loss.backward()

optimizer.step()
```

其中：

```python
optimizer.zero_grad()
```

就是：

> 把上一轮参数的 gradient 清掉。

否则上一轮 gradient 会和这一轮混起来。

---

# 三十三、optimizer.step() 又是什么？

假设最简单的 Gradient Descent：

$$
W_{new}
=
W_{old}
-
\eta
\frac{\partial L}{\partial W}
$$

其中：

$$
\eta
$$

是 learning rate。

PyTorch 里：

```python
optimizer.step()
```

就是：

> 读取每个参数的 `.grad`，然后按照 optimizer 的规则更新参数。

如果是 AdamW，公式当然复杂一些。

但思想没变：

```text
parameter.grad
        ↓
optimizer
        ↓
更新 parameter
```

---

# 三十四、为什么更新参数时要 no_grad？

原文手写 Gradient Descent：

```python
with torch.no_grad():
    w -= 0.01 * w.grad
```

为什么？

因为：

```python
w -= ...
```

本身也是一个数学操作。

但参数更新这个操作：

> 我们不希望它进入下一轮计算图。

否则 PyTorch 会继续追踪：

```text
w 是怎么被 optimizer 更新的
```

这通常不是普通训练想要的。

所以：

```python
torch.no_grad()
```

意思就是：

> **这里面的计算不要建立 Autograd graph。**

---

# 三十五、第三大块：nn.Module

如果 Tensor 是数据，

Autograd 是梯度系统，

那么：

```python
nn.Module
```

就是：

> **PyTorch 用来组织模型的基本单位。**

几乎所有模型都继承：

```python
nn.Module
```

比如：

```python
class SimpleModel(nn.Module):
    ...
```

---

# 三十六、一个 Module 通常有两个核心部分

```python
class Model(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ...
```

可以这样理解：

### `__init__`

定义：

> 我有哪些层、哪些参数？

### `forward`

定义：

> 数据进来以后按照什么顺序计算？

---

# 三十七、比如一个简单 MLP

```python
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
```

数学上：

$$
h = \operatorname{ReLU}(XW_1+b_1)
$$

然后：

$$
Y=hW_2+b_2
$$

所以 Module 其实就是：

> **把数学计算写成一个有参数管理能力的 Python 对象。**

---

# 三十八、为什么用 model(x)，而不是 model.forward(x)？

一般正常使用：

```python
output = model(x)
```

而不是：

```python
output = model.forward(x)
```

因为：

```python
model(x)
```

实际上会经过 `nn.Module.__call__` 的完整流程，再调用 `forward()`。

这样 PyTorch 才能正确处理各种：

- hook；
- autocast；
- module 机制；
- tracing / compile 相关逻辑。

所以日常记：

```python
model(x)
```

就行。

---

# 三十九、Transformer 其实也是一大堆 nn.Module 嵌套

比如你前面学的 Decoder Block，可以写成：

```python
class DecoderBlock(nn.Module):

    def __init__(self):
        self.norm1 = RMSNorm(...)
        self.attn = Attention(...)
        self.norm2 = RMSNorm(...)
        self.ffn = SwiGLU(...)

    def forward(self, x):
        h = x + self.attn(self.norm1(x))
        y = h + self.ffn(self.norm2(h))
        return y
```

你看。

以前你学的是：

$$
h=x+\operatorname{Attention}(\operatorname{RMSNorm}(x))
$$

$$
y=h+\operatorname{FFN}(\operatorname{RMSNorm}(h))
$$

现在变成 PyTorch：

```python
h = x + self.attn(self.norm1(x))
out = h + self.ffn(self.norm2(h))
```

所以：

$$
\text{数学结构}
\leftrightarrow
\text{nn.Module.forward()}
$$

---

# 四十、parameters() 是什么？

模型里可能有：

```text
WQ
WK
WV
WO
Wup
Wgate
Wdown
RMSNorm gamma
Embedding
LM Head
...
```

难道你自己一个一个找吗？

不用。

PyTorch：

```python
model.parameters()
```

会递归找到模型里的所有参数。

所以：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=...
)
```

就可以把整个模型的参数交给 optimizer。

---

# 四十一、怎么统计模型参数量？

原文给了非常重要的一行：

```python
total_params = sum(
    p.numel()
    for p in model.parameters()
)
```

其中：

```python
p.numel()
```

表示：

> 这个 Tensor 一共有多少个元素。

如果：

```python
W.shape == (4096, 4096)
```

那么：

$$
\text{numel}(W)
=
4096 \times 4096
$$

所以：

```python
sum(p.numel() for p in model.parameters())
```

就是模型总参数量。

这和你刚刚手算：

```text
7B model
```

完全是同一件事。

---

# 四十二、named_parameters() 是什么？

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

可能看到：

```text
layers.0.attn.q_proj.weight
layers.0.attn.k_proj.weight
layers.0.attn.v_proj.weight
layers.0.ffn.up_proj.weight
...
```

这个以后调模型特别有用。

比如：

> 到底哪一层参数 shape 不对？

或者：

> 我要冻结哪些参数？

或者：

> Tensor Parallel 到底切了哪个参数？

都会大量看：

```python
named_parameters()
```

---

# 四十三、state_dict() 是什么？

这个也特别重要。

```python
model.state_dict()
```

可以理解成：

> **模型所有需要保存状态的 Tensor 的字典。**

例如：

```text
linear.weight
linear.bias
...
```

所以保存模型经常：

```python
torch.save(model.state_dict(), ...)
```

加载：

```python
model.load_state_dict(...)
```

---

# 四十四、Parameter 和 state_dict 不完全是一回事

入门阶段可以先这样理解：

```text
parameters()
→ 主要给 optimizer 用

state_dict()
→ 主要给保存 / 加载 checkpoint 用
```

以后会发现：

```text
state_dict
```

除了 parameter，也可以包含 buffer。

比如某些不训练但需要保存的 Tensor。

现在先知道这个区别即可。

---

# 四十五、几个 Transformer 里特别常见的 Module

原文讲了：

```python
nn.Linear
nn.Embedding
nn.LayerNorm
nn.Dropout
```

它们和你前面知识可以直接一一对应。

---

# 四十六、nn.Linear

比如：

```python
linear = nn.Linear(512, 256)
```

数学上：

$$
Y=XW^T+b
$$

注意 PyTorch 的 Linear weight 通常存储 shape 是：

```text
(out_features, in_features)
```

比如：

```text
(256, 512)
```

这一点和我们数学里经常写：

$$
W \in \mathbb{R}^{512 \times 256}
$$

方向可能看起来相反。

因为 PyTorch 实际调用可以理解成：

$$
Y=XW^T+b
$$

这个以后看 weight shape 时要注意。

---

# 四十七、为什么 Transformer 里 Linear 特别多？

你现在应该已经可以秒答了。

Attention：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

还有：

$$
OW_O
$$

FFN：

$$
XW_{\text{gate}}
$$

$$
XW_{\text{up}}
$$

$$
ZW_{\text{down}}
$$

LM Head：

$$
HW_{\text{LM}}
$$

所以：

$$
\text{Transformer 绝大多数参数和 FLOPs 都集中在 Linear / GEMM}
$$

这也正是为什么：

```python
nn.Linear
```

对 AI Infra 特别重要。

---

# 四十八、nn.Embedding

比如：

```python
embedding = nn.Embedding(
    num_embeddings=50000,
    embedding_dim=768
)
```

这就对应：

$$
E \in \mathbb{R}^{V \times d}
$$

其中：

$$
V=50000
$$

$$
d=768
$$

输入：

```python
token_id = 42
```

本质上就是：

> 从 Embedding Matrix 里取第 42 行。

所以 Embedding 主要是：

$$
\text{Lookup}
$$

而不是像 Linear 那样把整个矩阵做 GEMM。

这也正好解释了为什么你之前学：

$$
FLOPs \approx 2Ns
$$

时，说它只是经验近似。

因为 Embedding 有很多参数，但不会对每一个 token 把整个 Embedding Matrix 乘一遍。

---

# 四十九、第四大块：训练循环

这一部分必须吃透。

无论模型多大：

```text
MLP
CNN
Transformer
7B LLM
70B LLM
```

训练的核心逻辑其实一直是：

```text
拿数据
↓
Forward
↓
计算 Loss
↓
Backward
↓
Optimizer 更新参数
↓
清梯度
↓
下一步
```

也就是原文总结的五个核心步骤：

```text
forward
→ loss
→ backward
→ optimizer.step
→ zero_grad
```

---

# 五十、Dataset 和 DataLoader 是什么关系？

可以这样理解：

```text
Dataset：
数据集是什么

DataLoader：
怎么一批一批把数据喂出来
```

Dataset 负责：

```python
__len__()
```

和：

```python
__getitem__(idx)
```

也就是：

> 有多少条数据，以及第 idx 条数据是什么。

---

# 五十一、DataLoader 负责什么？

比如：

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

它负责：

- batching；
- shuffle；
- 多 worker 加载；
- 数据预取相关能力。

所以最终：

```python
for x, y in loader:
```

每次拿到的是一个 batch。

---

# 五十二、这和你前面的 B 是什么关系？

以前：

$$
X \in \mathbb{R}^{B \times S \times d}
$$

这里的：

$$
B
$$

很多时候就是 DataLoader 帮你组出来的：

```python
batch_size=B
```

例如：

```python
batch_size=32
```

那么每次 forward：

```text
32 个 samples
```

一起进模型。

---

# 五十三、为什么 DataLoader 也会成为 AI Infra 瓶颈？

假设 GPU 一次计算只需要：

```text
10 ms
```

但是 CPU：

- 读磁盘；
- 解压；
- 数据预处理；
- 组 batch；

花了：

```text
30 ms
```

那么：

```text
GPU 算 10 ms
↓
等 CPU 20 ms
↓
再算
```

GPU 大量时间都在空等。

所以：

$$
\text{GPU 利用率低}
$$

不一定是 GPU kernel 写得差。

也可能：

> **数据根本没及时喂过来。**

这就是：

$$
\text{Data Loading Bottleneck}
$$

---

# 五十四、num_workers 是干嘛的？

DataLoader 可以启动多个 worker 去并行准备数据：

```python
num_workers=4
```

意思大概就是：

> 不要只让主 Python 进程一个人慢慢准备。

而是让多个 worker 并行。

但也不是越大越好。

这个以后需要 profiler / benchmark 来调。

---

# 五十五、pin_memory 是什么？

原文提到：

```python
pin_memory=True
```

这是为了让 CPU → GPU 数据传输更高效。

你现在先记：

> **Pinned Memory 是不会被 OS 随意换出的固定页内存，可以更方便 GPU 做 DMA 数据传输。**

以后学 CUDA 的：

```text
Pinned Host Memory
Async Memory Copy
CUDA Stream
```

你会重新碰到。

所以这里是一个提前埋下的伏笔。

---

# 五十六、标准训练循环一定要读懂

最核心可以写成：

```python
for x, y in dataloader:

    x = x.to(device)
    y = y.to(device)

    logits = model(x)

    loss = criterion(logits, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
```

把它逐行翻成人话。

> 注意看代码，我们是每次，从数据集里拿出来一份 x,y 进行训练，更新参数的；
>
> 一个 batch 就是我们拿出来的一份 x,y。
>
> 那么假设数据集有 N 条样本；batch size 也确定了（就算一个 batch 有几条样本）；那么 batch 个数也确定了。
>
> epoch 就叫训练轮次；1 epoch = 我们把整个训练 data  set 完整过了一次。

---

## 1. 数据搬到设备

```python
x = x.to(device)
```

如果：

```text
device = cuda
```

就是 CPU → GPU。

---

## 2. Forward

```python
logits = model(x)
```

这一步就是我们前面学的整个：

```text
Embedding
→ Transformer Blocks
→ LM Head
```

之类的 forward。

---

## 3. Loss

```python
loss = criterion(logits, y)
```

比如 Cross Entropy。

你前面已经学过：

$$
L=-\log p_y
$$

所以：

> 看模型预测得有多差。

---

## 4. Backward

```python
loss.backward()
```

Autograd 开始根据计算图反向算所有 parameter gradients。

---

## 5. 更新参数

```python
optimizer.step()
```

比如 AdamW 根据：

```python
param.grad
```

更新参数。

---

## 6. 清 gradient

```python
optimizer.zero_grad()
```

准备下一轮。

---

# 五十七、Epoch 和 Step 要区分

假设数据集有：

$$
10000
$$

条数据。

Batch Size：

$$
100
$$

那么一个 epoch 大约：

$$
\frac{10000}{100}=100
$$

个 training steps。

所以：

```text
Step：
处理一个 batch

Epoch：
整个数据集大致完整走一遍
```

LLM 训练里因为数据量巨大，很多时候更习惯直接讨论：

```text
training steps
tokens trained
```

---

# 五十八、Learning Rate Scheduler 是什么？

Learning Rate 不一定永远固定。

例如训练开始：

```text
lr 大一点
```

后来：

```text
逐渐变小
```

PyTorch 提供：

```python
scheduler
```

例如：

```python
CosineAnnealingLR
```

你现在只需要知道：

$$
\eta = \eta(t)
$$

也就是说：

> learning rate 会随着训练进度变化。

以后大模型训练会学：

```text
Warmup
Cosine Decay
```

这些。

---

# 五十九、Checkpoint 为什么不只保存 model？

如果只保存：

```python
model.state_dict()
```

你确实保存了模型参数。

但如果训练跑到：

```text
第 50000 step
```

机器挂了。

你想原地恢复训练，还需要：

- optimizer state；
- 当前 epoch / step；
- scheduler state；
- 其他随机状态等。

所以原文示例保存：

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...
}
```

这就是：

$$
\text{Checkpoint}
$$

---

# 六十、为什么 Optimizer State 很大？

你前面估训练显存已经碰过了。

比如 AdamW 对每个 parameter 还可能维护：

$$
m
$$

和：

$$
v
$$

也就是一阶动量和二阶动量。

所以 checkpoint 不只是：

> 模型那几十 GB。

训练 checkpoint 可能大得多。

这就是为什么后面分布式训练里：

```text
Checkpoint Save / Load
```

本身也是一个 AI Infra 问题。

---

# 六十一、第五大块：GPU 训练

最基础规则：

> **参与同一个运算的 Tensor 必须位于兼容的 device 上。**

比如：

```python
model = model.to("cuda")
```

那么输入也应该：

```python
x = x.to("cuda")
```

如果：

```text
model 在 GPU
x 在 CPU
```

直接计算通常会报 device mismatch。

---

# 六十二、model.to(device) 到底做了什么？

比如：

```python
model.to("cuda")
```

会把 model 里面注册的：

```text
parameters
buffers
```

搬到 GPU。

所以：

```python
model.to(device)
```

不是只是修改一个字符串。

它是真的把大量 Tensor 搬过去。

如果 7B 模型 BF16：

$$
7\times10^9\times2\text{ Bytes}
$$

大约就是：

$$
14\text{ GB}
$$

的数据。

所以对于大型模型：

> “把模型放到 GPU”本身就是一个真实的显存分配和数据搬运过程。

---

# 六十三、混合精度训练是什么？

原文的核心思想是：

> 大量 forward / backward 计算使用低精度，让 Tensor Core 更快、activation 更省显存；某些需要稳定性的地方仍保持更高精度。

比如：

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16
):
    loss = model(...)
```

`autocast` 会让许多适合低精度的算子自动使用 BF16 / FP16。

---

# 六十四、为什么叫 Mixed Precision？

因为 `autocast` 不是简单地：

```text
整个程序所有东西全部 BF16
```

而是：

```text
某些操作 BF16
某些操作 FP32
某些状态 FP32

让 PyTorch 自动决定哪些算子用 BF16，哪些继续用 FP32。
dtype=torch.bfloat16 主要是在告诉 autocast，适合降精度的算子，优先用 BF16。
```

也就是：

$$
\text{Mixed Precision}
$$

混合精度。

---

# 六十五、为什么 FP16 常需要 GradScaler？

FP16 的动态范围比较小。

反向传播时，有些 gradient 可能非常小。

例如：

$$
10^{-8}
$$

可能小到 FP16 无法正常表示。

于是发生：

$$
\text{underflow}
$$

变成 0。

所以思路是：

> 先把 loss 放大。

比如：

$$
L'=SL
$$

那么 gradient：

$$
\frac{\partial L'}{\partial W}
=
S
\frac{\partial L}{\partial W}
$$

gradient 也一起被放大。

等 optimizer 更新时再把 scale 正确处理回来。

这就是：

$$
\text{Loss Scaling}
$$

PyTorch 的：

```python
GradScaler
```

就是帮你做这件事。

---

# 六十六、为什么 BF16 通常不需要 GradScaler？

因为 BF16 的 exponent 范围接近 FP32。

所以它对极大值 / 极小值的表达范围比 FP16 强得多。

因此通常没那么容易因为动态范围不足发生 gradient underflow。

所以原文给 BF16 的代码是：

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16
):
    loss = ...
```

然后直接：

```python
loss.backward()
optimizer.step()
```

不需要和 FP16 一样使用 GradScaler。

---

# 六十七、混合精度是不是所有显存直接减半？

不是。

这个原文特别值得注意。

假设 activation：

```text
FP32 → BF16
```

确实：

$$
4\text{ Bytes}
\rightarrow
2\text{ Bytes}
$$

activation 显存可以明显下降。

但训练显存还有：

```text
模型权重
Gradient
Optimizer State
其他 Buffer
```

有些仍然可能是 FP32。

所以最终训练显存：

> 不一定直接减半。

原文甚至专门强调：

> 如果模型 / batch 太小，固定的 FP32 权重和 optimizer state 占主导，你可能看不出很大的显存节省。

这个思路很重要。

---

# 六十八、如何查看 GPU 显存？

PyTorch 可以：

```python
torch.cuda.memory_allocated()
```

查看当前已分配显存。

还可以：

```python
torch.cuda.max_memory_allocated()
```

查看峰值显存。

这个在 AI Infra 里非常实用。

比如你要问：

```text
这个 Transformer Block forward 峰值显存多少？

FlashAttention 到底省了多少？

BF16 到底省多少 activation？

Batch Size 最大能开多少？
```

都需要真正测。

---

# 六十九、reset_peak_memory_stats() 要特别理解

```python
torch.cuda.reset_peak_memory_stats()
```

只是：

> 重置“峰值统计数字”。

它不是：

> 清空 GPU 显存。

这两个完全不同。

所以不要看到：

```text
reset
```

就以为数据被释放了。

---

# 七十、empty_cache() 也不要误解

```python
torch.cuda.empty_cache()
```

并不是：

> 强制删除所有还活着的 Tensor。

如果某个 Tensor 还被 Python 变量引用：

```python
x
model
optimizer
```

它对应的数据不能凭空被删掉。

所以原文实验里先：

```python
del model, opt, loss
```

再：

```python
torch.cuda.empty_cache()
```

就是这个原因。

---

# 七十一、第六大块：Profiler

这部分对 AI Infra 来说可能是整篇最“对口”的。

因为：

> **能跑 ≠ 跑得快。**

Profiler 要解决：

```text
到底哪里慢？
```

而不是靠猜。

---

# 七十二、torch.profiler 在看什么？

原文例子：

```python
with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA
    ],
    record_shapes=True,
    profile_memory=True
) as prof:
    ...
```

它可以记录：

- CPU 时间；
- CUDA 时间；
- operator；
- shape；
- memory；
- 调用次数。

然后：

```python
prof.key_averages().table(...)
```

让你找热点。

---

# 七十三、Profiler 最重要的是建立“不要猜性能”的习惯

比如模型很慢。

你可能第一反应：

> GPU 太慢。

结果 profiler 一看：

```text
GPU kernel 总共只跑 30ms

CPU DataLoader 卡了 200ms
```

那你优化 CUDA kernel 半天都没用。

真正瓶颈是：

$$
\text{CPU / Data Loading}
$$

所以 AI Infra 很重要的思维是：

$$
\text{Measure First}
$$

而不是：

$$
\text{Guess First}
$$

---

# 七十四、CPU total 和 CUDA total 怎么看？

原文让你重点关注：

```text
CPU total
CUDA total
# of Calls
```

例如：

### CPU 很高、CUDA 很低

可能意味着：

```text
GPU 在等 CPU
```

比如：

- Python overhead；
- DataLoader；
- 数据预处理；
- CPU → GPU 搬运。

---

### CUDA 很高

说明：

> 真正 GPU kernels 很耗时。

这时候再去研究：

- GEMM；
- Attention；
- Memory Bandwidth；
- kernel fusion；
- Tensor Core utilization。

---

# 七十五、大量小 Kernel 为什么不好？

假设你需要做：

```text
100 个很小的操作
```

如果每一个都单独 launch 一个 CUDA kernel：

```text
Kernel 1 launch
Kernel 2 launch
Kernel 3 launch
...
Kernel 100 launch
```

即使每个 kernel 算得非常快，

launch 本身也有 overhead。

所以最后可能是：

$$
\text{Kernel Launch Overhead}
$$

占了很多时间。

这就是为什么你前面学 RMSNorm、Residual 时，我一直说：

> 虽然 FLOPs 小，但可能值得 fusion。

例如原来：

```text
Residual Add
↓
写 HBM
↓
RMSNorm
↓
再读 HBM
```

可以考虑 fuse：

```text
Residual + RMSNorm
```

减少：

- kernel launch；
- HBM traffic。

---

# 七十六、为什么 .item() 可能让性能突然变差？

这个特别有意思。

GPU 计算通常是异步的。

CPU 发出：

```text
kernel A
kernel B
kernel C
```

不一定每次都原地等 GPU 跑完。

但是：

```python
loss.item()
```

意思是：

> 我要把 GPU 上这个 loss 的具体数值拿回 CPU。

那 CPU 必须知道：

```text
GPU 算完了吗？
```

于是很可能需要：

$$
\text{CPU-GPU Synchronization}
$$

CPU 得等 GPU。

这样 pipeline 被打断。

所以：

```python
for step:
    print(loss.item())
```

每一步都打印，

可能形成隐式同步开销。

---

# 七十七、这就是为什么性能优化不能只看 FLOPs

这和你前面学的内容已经完全接起来了。

真实 runtime 可能包含：

```text
GEMM FLOPs

HBM Traffic

CPU → GPU Copy

Kernel Launch

Synchronization

Python Overhead

Data Loading

Communication
```

所以：

$$
\text{Runtime}
\neq
\frac{\text{FLOPs}}{\text{Peak FLOPs}}
$$

现实远没有这么简单。

这就是 AI Infra 真正要研究的东西。

---

# 七十八、Profiler 为什么要 Warmup？

原文 profiler 示例先：

```python
for _ in range(3):
    ...
```

然后才真正测。

原因是第一次运行时可能发生：

- CUDA context 初始化；
- kernel loading；
- memory allocation；
- library initialization；
- algorithm selection。

所以：

```text
第一次
```

通常不能代表稳定状态性能。

因此 benchmark 经常：

```text
Warmup
↓
正式测量
```

这个习惯以后写 CUDA benchmark 也会一直出现。

---

# 七十九、把 PyTorch 和你前面的 Transformer 知识整个接起来

现在你已经可以这样看代码：

```python
q = self.q_proj(x)
```

脑子里不是只有：

> “调用一个 Python 函数。”

而应该知道：

$$
Q=XW_Q
$$

本质大概率是：

$$
\text{GEMM}
$$

---

看到：

```python
q = q.view(B, S, H, D).transpose(1, 2)
```

想到：

$$
(B,S,d)
\rightarrow
(B,H,S,D_h)
$$

---

看到：

```python
attn = q @ k.transpose(-2, -1)
```

想到：

$$
QK^T
$$

shape：

$$
(B,H,S,S)
$$

以及：

$$
O(S^2)
$$

---

看到：

```python
loss.backward()
```

想到：

```text
Autograd
↓
VJP / Chain Rule
↓
Linear backward
↓
大量 GEMM
```

### 啥是 VJP

VJP = **Vector-Jacobian Product**，中文一般叫：

$$
\boxed{\text{向量-雅可比矩阵乘积}}
$$

假设：

$$
y=f(x)
$$

它的 Jacobian 是：

$$
J=\frac{\partial y}{\partial x}
$$

反向传播时，如果上游梯度是：

$$
g=\frac{\partial L}{\partial y}
$$

那么我们真正需要的是：

$$
\boxed{
\frac{\partial L}{\partial x}
=
gJ
}
$$

这就是 VJP。

关键点是：

$$
\boxed{
\text{PyTorch 不会真的把巨大的 Jacobian }J\text{ 整个造出来}
}
$$

而是直接计算：

$$
gJ
$$

这样可以省很多内存和计算。

所以你可以把 VJP 记成一句话：

> **反向传播时，把“上游梯度向量”乘上当前函数的 Jacobian，得到继续往前传的梯度。**

`loss.backward()` 背后大量做的，本质就是 VJP。

---

看到：

```python
model.to("cuda")
```

想到：

```text
Parameters / Buffers
↓
CPU Memory
↓
PCIe / 互联
↓
GPU HBM
```

---

看到：

```python
with torch.autocast(...):
```

想到：

```text
BF16 / FP16
↓
Tensor Core
↓
减少部分显存
↓
数值稳定性 trade-off
```

---

看到：

```python
torch.profiler
```

想到：

```text
不要猜瓶颈
↓
直接测 CPU / CUDA / Memory / Shapes
```

这才是这一章对你最重要的意义。

---

# 八十、这一章你现在最需要吃透什么？

我建议你分成三个层级。

## 第一层：必须真正掌握

### Tensor 的四个属性

```text
shape
dtype
device
layout / contiguous
```

---

### shape 操作

```text
reshape / view
permute
squeeze
unsqueeze
```

尤其必须会追：

$$
(B,S,d)
\rightarrow
(B,H,S,D_h)
$$

---

### Autograd

必须知道：

```text
requires_grad
forward 构图
loss.backward()
param.grad
gradient accumulation
zero_grad
```

---

### nn.Module

必须理解：

```text
__init__
forward
parameters()
state_dict()
```

---

### 标准训练循环

```text
Data
↓
Forward
↓
Loss
↓
Backward
↓
Optimizer Step
↓
Zero Grad
```

这五个必须熟。

---

# 八十一、第二层：要建立 AI Infra 直觉

包括：

```text
CPU / GPU device

CPU-GPU 数据搬运

FP32 / FP16 / BF16

Autocast

GradScaler

显存统计

DataLoader

pin_memory

num_workers
```

需要知道：

> 它们为什么会影响性能和显存。

---

# 八十二、第三层：这一篇最 AI Infra 的内容

就是：

```text
torch.profiler
```

以及：

```text
CPU Bottleneck

GPU Bottleneck

Data Loading Bottleneck

Kernel Launch Overhead

CPU-GPU Synchronization

Memory Traffic
```

你现在不用成为 profiler 专家。

但一定要形成一个习惯：

$$
\text{性能问题先测量，再优化}
$$

---

# 八十三、把整篇压缩成一张图

```text
                       PyTorch
                          │
          ┌───────────────┼───────────────┐
          │               │               │
        Tensor         Autograd       nn.Module
          │               │               │
          │               │               │
    数据怎么表示       梯度怎么算       模型怎么组织
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
                    Training Loop
                          │
              forward → loss → backward
                          │
                    optimizer.step
                          │
                     zero_grad
                          │
                          ▼
                      GPU Training
                          │
               ┌──────────┼──────────┐
               │          │          │
             device      dtype      memory
               │          │          │
               ▼          ▼          ▼
           CPU↔GPU     BF16/FP16   显存管理
               │          │          │
               └──────────┼──────────┘
                          │
                          ▼
                       Profiler
                          │
                          ▼
                 到底哪里真的慢？
                          │
             ┌────────────┼────────────┐
             │            │            │
            CPU          GPU         Memory
             │            │            │
       DataLoader      Kernel      HBM / Copy
```

---

# 八十四、你可以把 PyTorch 看成四层

以后读代码时，我建议你脑子里这样分。

## 第一层：数学

比如：

$$
Y=XW
$$

$$
QK^T
$$

$$
L=-\log p_y
$$

---

## 第二层：PyTorch API

对应：

```python
nn.Linear
torch.matmul
cross_entropy
loss.backward()
```

---

## 第三层：Kernel

PyTorch API 最后会调用：

```text
GEMM Kernel
Softmax Kernel
Norm Kernel
Elementwise Kernel
...
```

---

## 第四层：Hardware

Kernel 最后跑在：

```text
SM
Tensor Core
CUDA Core
HBM
L2 Cache
Shared Memory
Register
```

上。

所以后面整个 AI Infra 学习路线其实就是：

```text
数学
↓
PyTorch
↓
CUDA Kernel
↓
GPU Hardware
```

你现在刚好走到第二层。

---

# 八十五、这一章和你前面 Transformer 的最大连接点

你前面一直在问：

> “这个公式真实代码到底长什么样？”

PyTorch 就是在回答这件事。

比如：

$$
Q=XW_Q
$$

变成：

```python
q = self.q_proj(x)
```

---

$$
(B,S,d)
\rightarrow
(B,H,S,D_h)
$$

变成：

```python
q = q.view(B, S, H, D).transpose(1, 2)
```

---

$$
QK^T
$$

变成：

```python
scores = q @ k.transpose(-2, -1)
```

---

$$
\operatorname{softmax}
$$

变成：

```python
attn = torch.softmax(scores, dim=-1)
```

---

$$
L
$$

反向传播：

```python
loss.backward()
```

所以从这一章开始，你其实是在：

$$
\text{把数学模型翻译成真实程序}
$$

---

# 八十六、我建议你这篇学完以后，能回答下面 15 个问题

```text
1. Tensor 的 shape、dtype、device 分别是什么意思？

2. reshape / view 和 permute 有什么区别？

3. 为什么 Multi-Head Attention 里一定会大量出现 reshape 和 permute？

4. requires_grad=True 到底告诉 PyTorch 什么？

5. loss.backward() 背后在干嘛？

6. param.grad 存的是什么？

7. 为什么 PyTorch 的梯度默认是累加而不是覆盖？

8. 为什么正常训练要 zero_grad()？

9. nn.Module 的 __init__ 和 forward 分别负责什么？

10. model.parameters() 和 state_dict() 分别主要用来干什么？

11. 一个标准 training step 有哪几个核心步骤？

12. 为什么 CPU → GPU 数据搬运可能成为瓶颈？

13. FP16 和 BF16 为什么只占 FP32 一半显存，但数值特性不同？

14. 为什么 loss.item() 可能导致 CPU-GPU 同步？

15. 为什么做 AI Infra 性能优化前应该先跑 profiler？
```

如果这些你都能讲清楚，

那么这篇 PyTorch 入门最重要的部分基本就拿下了。

---

# 八十七、最后只记住一句主线

前面的 Transformer 章节是在教你：

$$
\text{模型“应该怎么算”}
$$

而 PyTorch 这一章开始教你：

$$
\text{这些计算“在真实程序里怎么表达和执行”}
$$

再往后进入 GPU / CUDA 时，你会继续追问：

$$
\text{PyTorch 写下来的这些操作，最终在 GPU 上到底是怎么跑起来的？}
$$

所以整个学习路径现在已经非常清楚：

```text
数学基础
   ↓
Transformer
   ↓
PyTorch
   ↓
GPU Hardware
   ↓
CUDA Kernel
   ↓
分布式训练 / 推理优化
```

这就是这一章在整个 AI Infra 学习路线里的位置。