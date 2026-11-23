# GPU 基础知识：从硬件架构到 AI 计算

你现在 PyTorch 基本学完以后，再进 GPU Basics，顺序是很合适的。

前面你已经知道：

```text
Transformer
↓
PyTorch
↓
各种 Tensor 运算
```

比如：

```python
q = x @ Wq
```

```python
scores = q @ k.transpose(-2, -1)
```

```python
loss.backward()
```

但是以前我们只是知道：

> 这些东西最终会“放到 GPU 上算”。

现在开始往下一层钻：

> GPU 到底为什么能算得这么快？

> PyTorch 的一个矩阵乘，最终在 GPU 里面发生了什么？

这一篇最重要的主线可以先压成：

```text
CPU vs GPU
    ↓
SM
    ↓
Warp / Thread
    ↓
CUDA Core / Tensor Core
    ↓
Register / Shared Memory / L2 / HBM
    ↓
Compute Bound / Memory Bound
    ↓
Grid / Block / Thread
    ↓
GPU 性能指标
```

你学这一篇不是为了背硬件名词，而是为了以后理解：

```text
为什么 FlashAttention 快？

为什么 Kernel Fusion 有用？

为什么 Decode 是 Memory Bound？

为什么 GEMM 喜欢 Tensor Core？

为什么 CUDA 要搞 Shared Memory？

为什么多 GPU 以后通信会成为瓶颈？
```

---

# 一、CPU 和 GPU 为什么设计得完全不一样？

最核心的区别可以先记成：

```text
CPU：

少量很强的核心
擅长复杂控制逻辑
追求单个任务低延迟


GPU：

大量相对简单的计算单元
擅长规则的大规模并行
追求整体高吞吐
```

也就是：

```text
CPU → Latency Oriented

GPU → Throughput Oriented
```

所谓 Latency，就是：

> 一个任务多久完成。

所谓 Throughput，就是：

> 单位时间总共能完成多少任务。

---

# 二、为什么深度学习特别适合 GPU？

因为深度学习里面大量都是这种东西：

```text
同一种操作
+
大量不同的数据
+
同时计算
```

例如矩阵乘：

$$
C = AB
$$

其中：

$$
C_{ij} = \sum_k A_{ik}B_{kj}
$$

矩阵里面很多不同的输出元素：

```text
C00
C01
C02
...
```

都可以并行计算。

Transformer 里面更是如此：

```text
QKV Projection
FFN
LM Head
QK^T
AV
```

核心都有大量矩阵乘。

所以：

$$
Y = XW
$$

这种工作负载非常符合 GPU 的设计哲学。

你可以记一句：

> GPU 并不是“什么都比 CPU 快”，而是特别擅长大量规则、重复、可并行的计算。

---

# 三、GPU 里面最重要的单位：SM

NVIDIA GPU 内部有很多层级。

文章里面会看到：

```text
GPU
↓
GPC
↓
TPC
↓
SM
```

你现在不用背 GPC 和 TPC。

最应该抓住的是：

```text
SM
```

SM 全称：

```text
Streaming Multiprocessor
```

可以把一张 GPU 想象成：

```text
GPU
├── SM 0
├── SM 1
├── SM 2
├── SM 3
├── ...
```

每个 SM 都像一个“小型计算工厂”。

里面有：

```text
CUDA Core
Tensor Core
Warp Scheduler
Load / Store Unit
Register
Shared Memory
L1 Cache
```

所以现在先记：

> SM 是理解 GPU 执行模型最关键的一层。

---

# 四、CUDA Core 是什么？

可以先把 CUDA Core 理解成：

> 执行普通算术运算的计算单元。

比如：

$$
a+b
$$

或者：

$$
a\times b
$$

以及非常重要的：

$$
a\times b+c
$$

最后这个叫：

```text
FMA
```

全称：

```text
Fused Multiply-Add
```

一次 FMA 包含：

```text
一次乘法
+
一次加法
```

所以在 FLOPs 统计中通常算：

$$
2
$$

次浮点运算。

这就是为什么你前面计算 GEMM FLOPs 的时候经常写：

$$
2MKN
$$

因为矩阵乘里面大量都是乘加。

---

# 五、Tensor Core 又是什么？

Tensor Core 和普通 CUDA Core 不完全是一回事。

你可以先把 CUDA Core 理解成：

```text
普通算术工人
```

而 Tensor Core 更像：

```text
专门处理矩阵乘加的硬件专家
```

它特别擅长：

$$
D = AB + C
$$

这个操作叫：

```text
MMA
```

也就是：

```text
Matrix Multiply-Accumulate
```

所以对于 AI 来说，Tensor Core 非常重要。

因为 Transformer 最重的计算恰好就是矩阵乘。

---

# 六、从 Transformer 到 Tensor Core 的链条

比如你写：

```python
y = x @ W
```

数学上：

$$
Y = XW
$$

对于大矩阵，这通常会走到高性能 GEMM 实现。

你可以先粗略建立：

```text
PyTorch Linear / MatMul
        ↓
      GEMM
        ↓
   切成很多 Tile
        ↓
Tensor Core 做很多小块 MMA
        ↓
拼出完整结果
```

所以你前面学的 GEMM，开始和硬件真正接上了。

---

# 七、为什么 BF16 / FP16 不只是为了省显存？

你之前已经知道：

```text
FP32 → 4 Bytes

FP16 → 2 Bytes

BF16 → 2 Bytes
```

所以低精度可以省显存。

但是还有一个非常重要的原因：

> Tensor Core 对 BF16、FP16 等低精度矩阵运算通常能够提供非常高的吞吐。

因此使用 BF16 有两个重要收益：

```text
1. 数据体积更小

2. 可以更好利用 AI 专用矩阵计算硬件
```

所以你前面学的：

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16
):
    ...
```

现在可以从硬件角度重新理解：

```text
PyTorch
↓
部分适合的计算使用 BF16
↓
大型矩阵乘走适合的 GPU 矩阵计算路径
↓
Tensor Core 得到更好的利用
```

---

# 八、Warp 是什么？

这个是 GPU 里面非常重要的概念。

NVIDIA GPU 并不是把所有 Thread 完全独立地一个一个执行。

它会把线程按组组织。

一个 Warp 有：

$$
32
$$

个线程。

可以想象成：

```text
Warp

Thread 0
Thread 1
Thread 2
...
Thread 31
```

这 32 个线程会以非常紧密的方式一起执行。

---

# 九、SIMT 是什么意思？

SIMT：

```text
Single Instruction, Multiple Threads
```

比如我们有：

$$
y_i = x_i + 1
$$

那么可以：

```text
Thread 0 处理 x0

Thread 1 处理 x1

Thread 2 处理 x2

...

Thread 31 处理 x31
```

它们执行的是：

```text
同一种指令
```

但是操作：

```text
不同的数据
```

所以 GPU 最舒服的情况就是：

> 很多线程走一样的计算流程，只是每个线程处理不同的数据。

---

# 十、为什么 GPU 不太喜欢复杂 if / else？

假设一个 Warp 里面有 32 个线程。

代码是：

```python
if x > 0:
    y = x * 2
else:
    y = x / 2
```

如果：

```text
16 个线程走 if

16 个线程走 else
```

那就麻烦了。

因为一个 Warp 的线程应该尽量“齐步走”。

实际执行时可能类似：

```text
先执行 if 分支

走 else 的线程暂时不干活


再执行 else 分支

走 if 的线程暂时不干活
```

于是部分计算资源被浪费。

这叫：

```text
Warp Divergence
```

也就是：

```text
Warp 分支发散
```

所以 GPU 喜欢：

```text
规则计算
少分支
大量数据并行
```

---

# 十一、接下来是 AI Infra 特别重要的一部分：Memory Hierarchy

GPU 不只有“显存”。

内部其实有很多层存储。

可以先画成：

```text
快、小
 ↑

Register

Shared Memory / L1

L2 Cache

HBM

 ↓
慢、大
```

核心规律：

> 越靠近计算单元，通常越快，但是容量越小。

---

# 十二、Register：寄存器

Register 是最靠近计算单元的存储。

可以想成：

> 每个线程手边自己的小纸条。

特点：

```text
非常快

容量很小

线程私有
```

CUDA Kernel 里的很多局部变量会尽量放这里。

所以如果数据能一直待在 Register 里参与计算，就非常舒服。

---

# 十三、Register Spilling 是什么？

问题是 Register 数量有限。

如果一个线程需要的寄存器太多：

```text
Register 不够
```

那么一些值可能会被“挤出去”。

这个叫：

```text
Register Spilling
```

被 spill 出去的数据可能进入所谓的：

```text
Local Memory
```

但特别注意：

> Local Memory 这个名字非常骗人。

它并不是一个特别快的“小本地内存”。

它实际上位于 GPU 显存一侧，访问代价很高。

所以：

```text
寄存器用太多
↓
Spill
↓
更多慢内存访问
↓
性能可能下降
```

---

# 十四、Shared Memory 是什么？

Shared Memory 是：

> 一个 Thread Block 内的线程可以共同使用的一块高速片上存储。

你可以把它理解成：

```text
Block 里面所有线程共享的小白板
```

例如：

```text
Thread 0 把数据写进 Shared Memory

Thread 1 可以读

Thread 2 也可以读
```

它比 HBM 快很多。

所以 CUDA 优化里面一个非常经典的思想就是：

```text
先从 HBM 把数据搬进 Shared Memory

然后反复利用

最后再写回 HBM
```

这叫：

```text
Data Reuse
```

也就是数据复用。

---

# 十五、L1、L2、HBM

可以粗略理解成：

```text
Register
    ↓
Shared Memory / L1
    ↓
L2
    ↓
HBM
```

其中：

```text
L1

更靠近单个 SM
```

而：

```text
L2

整张 GPU 上的多个 SM 可以共享
```

再往外就是：

```text
HBM
```

---

# 十六、HBM 是什么？

HBM：

```text
High Bandwidth Memory
```

就是现代 AI GPU 使用的高带宽显存。

你在 PyTorch 里：

```python
x = torch.randn(..., device="cuda")
```

这种大型 Tensor，主要就是放在 GPU Global Memory，也就是 HBM 这一层。

比如：

```text
Model Weights

Activations

Gradients

Optimizer States

KV Cache
```

所以你前面一直说的：

```text
GPU 显存
```

主要就是这一层。

---

# 十七、为什么 HBM 已经很快了，还会成为瓶颈？

因为 GPU 的计算单元也非常快。

假设 Tensor Core 已经准备好了：

```text
我要算矩阵乘！
```

但是数据还在 HBM，没有及时搬过来。

那 Tensor Core 只能：

```text
等数据。
```

所以程序性能不只是：

```text
算得快不快
```

还取决于：

```text
数据送得快不快
```

这就开始进入两个重要概念：

```text
Compute Bound

Memory Bound
```

---

# 十八、Arithmetic Intensity 是什么？

Arithmetic Intensity 中文叫：

```text
算术强度
```

公式是：

$$
AI = \frac{\mathrm{FLOPs}}{\mathrm{Memory\ Bytes}}
$$

不要把公式想复杂。

它只是问：

> 每从内存搬 1 Byte 数据过来，我能用它做多少次浮点运算？

---

# 十九、例子：算术强度很低

假设搬了：

$$
1000\ \mathrm{Bytes}
$$

但是只做：

$$
100\ \mathrm{FLOPs}
$$

那么：

$$
AI = \frac{100}{1000} = 0.1
$$

意思就是：

> 搬了一大堆数据过来，结果只算了一点点。

这种 workload 很容易：

```text
Memory Bound
```

因为真正耗时间的是：

```text
搬数据
```

而不是计算。

---

# 二十、例子：算术强度很高

假设仍然只搬：

$$
1000\ \mathrm{Bytes}
$$

但是做了：

$$
1000000\ \mathrm{FLOPs}
$$

那么：

$$
AI = \frac{1000000}{1000} = 1000
$$

意思是：

> 数据搬进来以后，被疯狂重复利用，做了很多计算。

这种 workload 更容易：

```text
Compute Bound
```

也就是最终可能被 GPU 算力限制。

---

# 二十一、Compute Bound 和 Memory Bound 怎么区分？

简单版本：

```text
Compute Bound

GPU 在等计算完成
```

也就是：

> 计算单元已经比较忙了，再提高内存带宽帮助不大。

而：

```text
Memory Bound

GPU 在等数据
```

也就是：

> 计算单元没吃饱，数据搬运跟不上。

所以以后看到一个算子，你首先可以问：

> 它主要是在等“算”，还是在等“数据”？

这是 GPU 性能分析里非常核心的问题。

---

# 二十二、为什么 GEMM 更容易 Compute Bound？

比如：

$$
C = AB
$$

矩阵 A、B 的一块数据从 HBM 搬进来以后，

我们不会只使用一次。

而是会把它放到：

```text
Shared Memory
Register
```

然后让它参与很多次乘加。

也就是说：

```text
搬一次
↓
反复计算很多次
```

数据复用率非常高。

所以 GEMM 的 Arithmetic Intensity 可以很高。

这也是为什么大型 GEMM 非常适合 Tensor Core。

---

# 二十三、为什么 RMSNorm 更容易 Memory Bound？

比如 RMSNorm 要处理：

$$
x_1,x_2,\ldots,x_d
$$

主要就是：

```text
读取很多元素

算平方

做 reduction

再乘 scale

写回
```

每一个元素真正做的 FLOPs 并没有特别多。

所以它更像：

```text
搬很多数据
+
做少量计算
```

因此：

```text
Arithmetic Intensity 低
```

所以更容易：

```text
Memory Bound
```

这也解释了我们前面说的：

> RMSNorm 的 FLOPs 虽然很少，不代表它实际 runtime 就一定完全可以忽略。

---

# 二十四、Kernel Fusion 为什么有用？

假设原来是：

```text
Residual Add
↓
写 HBM

RMSNorm
↓
重新读 HBM
↓
写 HBM

Activation
↓
再读 HBM
↓
再写 HBM
```

这几个算子本身 FLOPs 都不算特别高。

但是你疯狂：

```text
读显存
写显存
读显存
写显存
```

很浪费。

如果能融合成一个 Kernel：

```text
Residual
+
Norm
+
Activation
```

就可能让很多中间结果留在：

```text
Register

Shared Memory
```

而不是反复写回 HBM。

所以 Kernel Fusion 的一个核心目标就是：

```text
减少 HBM Traffic
```

---

# 二十五、现在终于真正理解 FlashAttention

普通 Attention 可以粗略想象成：

```text
QK^T
↓
巨大 Attention Matrix
↓
写入 HBM

再从 HBM 读出来
↓
Softmax

再写 HBM

再读出来
↓
乘 V
```

问题就是：

```text
中间矩阵太大

HBM 读写太多
```

FlashAttention 的核心思想：

```text
把 Q / K / V 分块

↓
搬一小块进片上高速存储

↓
在里面计算 QK^T

↓
Online Softmax

↓
直接和 V 结合

↓
中间完整 Attention Matrix 不落 HBM
```

所以 FlashAttention 很重要的一点不是：

```text
少算了特别多 FLOPs
```

而是：

```text
大幅减少了 HBM IO
```

你现在有 Memory Hierarchy 以后，这件事就很好理解了。

---

# 二十六、Roofline Model 先理解直觉就行

Roofline Model 想回答的是：

> 一个 Kernel 理论上到底是被计算能力限制，还是被内存带宽限制？

可以粗略写成：

$$
P_{\mathrm{actual}}
\le
\min
\left(
P_{\mathrm{compute}},
\;
BW \times AI
\right)
$$

其中：

```text
P_compute

GPU 的峰值计算能力
```

```text
BW

Memory Bandwidth
```

```text
AI

Arithmetic Intensity
```

所以：

```text
AI 很低
↓
BW × AI 很小
↓
Memory Bound
```

而：

```text
AI 很高
↓
Memory 不再是主要限制
↓
更容易 Compute Bound
```

现在不需要继续钻 Roofline 数学。

先懂这个判断逻辑即可。

---

# 二十七、重新理解 Prefill 为什么偏 Compute Bound

Prefill 的时候：

```text
一次有很多 token
```

比如：

$$
S = 2048
$$

Linear 可能类似：

$$
(2048,d)(d,d)
$$

这是一个很大的 GEMM。

一个权重矩阵被：

```text
很多 token
```

共同使用。

因此：

```text
同样一份 Weight 搬进来
↓
可以参与大量计算
```

数据复用高。

所以 Arithmetic Intensity 高。

因此 Prefill 通常更容易：

```text
Compute Bound
```

---

# 二十八、重新理解 Decode 为什么偏 Memory Bound

Decode 每次可能只有：

```text
1 个新 token
```

于是 Linear 更像：

$$
(1,d)(d,d)
$$

虽然只处理一个 token，

但那一大坨：

```text
Wq
Wk
Wv
Wo
Wup
Wgate
Wdown
```

仍然需要被读取。

同时还要读取越来越大的：

```text
KV Cache
```

于是变成：

```text
搬很多数据

但每次只有很少 token 可以复用这些数据
```

Arithmetic Intensity 较低。

所以更容易：

```text
Memory Bound
```

这就是你前面一直记的：

```text
Prefill → 偏算力

Decode → 偏带宽
```

背后的硬件原因。

---

# 二十九、CUDA 的线程层级

CUDA 里面最基本的逻辑层级是：

```text
Grid
  ↓
Block
  ↓
Thread
```

此外还有一个非常重要的硬件执行分组：

```text
Warp
```

所以整体可以想成：

```text
Grid
├── Block 0
│   ├── Warp 0
│   │   ├── Thread
│   │   ├── Thread
│   │   └── ...
│   └── Warp 1
│
├── Block 1
│
└── ...
```

---

# 三十、Grid 是什么？

一次 CUDA Kernel 启动时创建出来的全部线程，整体属于一个 Grid。

kernel=GPU 上并行执行的一段程序。

可以理解成：

```text
这一次 Kernel 的所有工作
```

---

# 三十一、Block 是什么？

一个 Block 是：

> 一组能够协作的线程。

同一个 Block 的线程可以：

```text
共享 Shared Memory

进行同步
```

一个 Thread Block 会被调度到某一个 SM 上执行。

所以：

```text
Block
```

是 CUDA 编程里非常重要的协作单位。

---

# 三十二、Thread 是什么？

Thread 就是逻辑上的单个 CUDA 线程。

比如向量加法：

$$
C_i = A_i + B_i
$$

我们可以让：

```text
Thread 0 → 算 C0

Thread 1 → 算 C1

Thread 2 → 算 C2

...
```

---

# 三十三、Block 和 Warp 千万别混

假设一个 Block 有：

$$
256
$$

个线程。

一个 Warp 有：

$$
32
$$

个线程。

那么这个 Block 会包含：

$$
\frac{256}{32}=8
$$

个 Warp。

也就是：

```text
Block

Warp 0 → Thread 0 ~ 31

Warp 1 → Thread 32 ~ 63

...

Warp 7 → Thread 224 ~ 255
```

所以：

```text
Block

是程序员配置的逻辑线程组织单位
```

而：

```text
Warp

是 GPU 执行调度中的重要硬件分组
```

---

# 三十四、一个最基本的 CUDA Kernel

比如向量加法：

```cpp
__global__ void vecAdd(
    float* A,
    float* B,
    float* C,
    int N
) {
    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

最重要的是：

```cpp
blockIdx.x
```

表示：

```text
我是第几个 Block
```

```cpp
blockDim.x
```

表示：

```text
一个 Block 有多少个 Thread
```

```cpp
threadIdx.x
```

表示：

```text
我是当前 Block 里的第几个 Thread
```

所以全局线程编号：

$$
idx
=
blockIdx.x \times blockDim.x
+
threadIdx.x
$$

---

# 三十五、举个例子

假设：

$$
blockDim.x = 256
$$

当前：

```text
Block 3
```

里面：

```text
Thread 10
```

那么：

$$
idx = 3 \times 256 + 10
$$

所以：

$$
idx = 778
$$

这个线程就负责：

$$
C_{778}=A_{778}+B_{778}
$$

这就是最基础的 CUDA 数据并行。

---

# 三十六、Kernel 到底是什么？

Kernel 可以先理解成：

> 在 GPU 上由大量线程并行执行的一段函数。

比如：

```cpp
vecAdd<<<blocks, threads>>>(...)
```

意思可以翻译成：

```text
CPU：

GPU，你启动这么多线程。

这些线程全部执行 vecAdd。
```

所以：

```text
Kernel
```

就是 GPU 上的一段并行程序。

---

# 三十七、这和 PyTorch 有什么关系？

比如你写：

```python
C = A + B
```

底层可以粗略想象成：

```text
Python
↓
PyTorch
↓
CUDA Backend
↓
启动 Elementwise Kernel
↓
Grid
↓
Block
↓
Warp
↓
Thread
↓
GPU 执行
```

而：

```python
C = A @ B
```

可能进入：

```text
PyTorch
↓
高性能矩阵乘库
↓
GEMM Kernel
↓
Tensor Core
```

所以 PyTorch 并没有绕开 CUDA。

它只是把底层很多复杂工作封装掉了。

---

# 三十八、为什么一个 model.forward() 会看到很多 Kernel？

比如一个 Transformer Block：

```text
RMSNorm
↓
QKV Projection
↓
RoPE
↓
Attention
↓
Output Projection
↓
Residual
↓
RMSNorm
↓
SwiGLU
↓
Residual
```

这些操作底下可能对应：

```text
很多不同 CUDA Kernel
```

所以：

```python
model(x)
```

看起来只是一个函数调用。

但 GPU 那边可能已经启动了很多 Kernel。

这就是为什么：

```text
Kernel Launch Overhead
```

和：

```text
Kernel Fusion
```

会成为性能问题。

---

# 三十九、GPU 的四个关键性能指标

以后看一张 AI GPU，不要只问：

```text
有多少 CUDA Core？
```

更重要的是看四件事。

---

## 1. Compute

也就是计算能力。

常见单位：

```text
TFLOPS
```

一 TFLOPS 表示：

$$
10^{12}
$$

次浮点运算每秒。

但要特别注意：

> 说 TFLOPS 时一定要问是什么 dtype、什么计算路径。

因为：

```text
FP32

TF32

FP16

BF16

FP8
```

峰值可能差非常多。

---

## 2. Memory Bandwidth

表示：

> GPU 每秒能从 HBM 搬多少数据。

常见单位：

```text
GB/s

TB/s
```

对于：

```text
Decode

KV Cache

RMSNorm

Elementwise
```

这种偏 Memory Bound 的 workload，非常重要。

---

## 3. Memory Capacity

就是：

```text
GPU 有多少 GB 显存。
```

它首先决定：

> 东西装不装得下。

比如一个 70B 模型，假设每个参数占 2 Bytes，仅参数就大约：

$$
70 \times 10^9 \times 2
$$

Bytes。

也就是大约：

$$
140\ \mathrm{GB}
$$

这还没算：

```text
KV Cache

Activation

其他 Buffer
```

所以：

> 算力够，但显存装不下，照样跑不了。

---

## 4. Interconnect

也就是 GPU 和 GPU 之间通信能力。

比如：

```text
PCIe

NVLink

NVSwitch

InfiniBand
```

单卡放不下模型以后，就得切多卡。

一旦切多卡：

```text
通信
```

就会成为新的成本。

---

# 四十、PCIe、NVLink、InfiniBand 先怎么区分？

简单理解：

```text
PCIe

CPU / GPU / 设备之间常见的通用高速互联
```

```text
NVLink

NVIDIA GPU 之间的高速互联
```

```text
NVSwitch

把多张 GPU 通过高速 NVLink 组织起来的交换结构
```

```text
InfiniBand

多台服务器之间常见的高速集群网络
```

可以粗略记成：

```text
GPU 内部
→ Register / Shared / HBM

同一台机器 GPU 之间
→ NVLink / NVSwitch / PCIe

不同机器之间
→ 高速网络 / InfiniBand
```

---

# 四十一、为什么多 GPU 后通信很重要？

假设一个矩阵被分到四张卡：

```text
GPU 0 算一部分

GPU 1 算一部分

GPU 2 算一部分

GPU 3 算一部分
```

如果每张卡：

```text
计算只花 1 ms
```

但互相交换结果：

```text
花 5 ms
```

那瓶颈就根本不在 Tensor Core。

而在：

```text
Communication
```

所以多 GPU 性能要同时看：

```text
Compute
+
Communication
```

---

# 四十二、重新理解 model.cuda()

你之前写：

```python
model = model.cuda()
```

可能只觉得：

> 模型被放到 GPU 了。

现在应该看到真实数据流：

```text
CPU DRAM

↓ PCIe / 互联

GPU HBM
```

比如一个 BF16 7B 模型仅参数大约：

$$
7 \times 10^9 \times 2
$$

Bytes。

约为：

$$
14\ \mathrm{GB}
$$

所以 `model.cuda()` 背后是真的要进行大量：

```text
显存分配
+
数据传输
```

---

# 四十三、为什么 GPU Benchmark 要 synchronize？

GPU Kernel 通常是异步启动的。

也就是说 CPU 可能做：

```text
提交 Kernel
↓
继续运行自己的 Python
```

而 GPU：

```text
在后台继续算
```

所以如果写：

```python
start = time.time()

C = A @ B

end = time.time()
```

这个时间可能不是真正的 GPU GEMM 时间。

因为 CPU 可能只是：

> 把任务提交出去以后就继续往下走了。

所以 benchmark 经常需要：

```python
torch.cuda.synchronize()
```

意思就是：

> CPU 等一下，确认 GPU 前面的工作真的做完。

这时候再计时才靠谱。

---

# 四十四、为什么 Mixed Precision 现在又更有意义了？

以前你知道：

```text
BF16 比 FP32 小
```

现在又多一层：

```text
低精度
↓
数据更小
↓
HBM Traffic 更少

同时：

低精度矩阵乘
↓
更容易使用高吞吐 Tensor Core 路径
```

所以：

```text
Mixed Precision
```

同时在优化：

```text
Memory
+
Compute
```

这就是为什么它对 AI 特别重要。

---

# 四十五、把 Transformer 每个部分重新用 GPU 眼光看

## QKV Projection

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

核心：

```text
GEMM
```

重点：

```text
Tensor Core
Data Reuse
Tiling
```

---

## FFN

例如：

$$
XW_{up}
$$

$$
XW_{gate}
$$

$$
ZW_{down}
$$

同样主要是：

```text
GEMM
```

所以也是 Transformer 中非常重的 Compute。

---

## QK^T 和 AV

$$
QK^T
$$

以及：

$$
AV
$$

也是矩阵乘。

但是 Attention 又会产生很大的中间数据。

所以既要考虑：

```text
FLOPs
```

也要考虑：

```text
Memory Traffic
```

---

## Softmax

主要是：

```text
exp

reduction

divide
```

相对 GEMM，数据搬运的重要性会更明显。

---

## RMSNorm

主要：

```text
read

reduce

elementwise

write
```

更容易：

```text
Memory Bound
```

---

# 四十六、现在前面那些 AI Infra 技术终于有了硬件解释

## FlashAttention

主要减少：

```text
HBM Traffic
```

---

## Kernel Fusion

主要减少：

```text
Kernel Launch

中间 Tensor 写回 HBM

再次从 HBM 读取
```

---

## GQA / MQA

减少：

```text
KV Cache
```

于是减少：

```text
显存占用

HBM 读取量
```

---

## Mixed Precision

减少：

```text
Bytes
```

同时提高：

```text
Tensor Core 吞吐
```

---

## Tensor Parallel

解决：

```text
单卡显存 / 算力不够
```

代价：

```text
GPU-GPU Communication
```

---

## Activation Checkpointing

减少：

```text
Activation Memory
```

代价：

```text
Backward 时重新计算
```

这就是典型：

```text
Memory 换 Compute
```

---

# 四十七、这一篇现在最应该真正吃透的内容

我建议你先把下面这些钉死。

```text
1. CPU vs GPU

CPU 偏低延迟
GPU 偏高吞吐
```

```text
2. SM

GPU 执行线程、分配片上资源的核心单位
```

```text
3. Warp

32 Threads
```

```text
4. SIMT

同一条指令
不同线程处理不同数据
```

```text
5. Warp Divergence

一个 Warp 内走不同分支会浪费执行资源
```

```text
6. CUDA 线程组织

Grid
↓
Block
↓
Thread
```

```text
7. Memory Hierarchy

Register
↓
Shared Memory / L1
↓
L2
↓
HBM
```

```text
8. Tensor Core

专门做高吞吐矩阵乘累加
```

```text
9. Arithmetic Intensity

FLOPs / Bytes
```

```text
10. Compute Bound vs Memory Bound

到底是在等计算还是等数据
```

---

# 四十八、目前可以先不深挖的东西

这些先知道名字即可：

```text
GPC

TPC

具体某代 GPU 有几个 SM

具体 Tensor Core 指令格式

Register Bank 细节

Shared Memory Bank Conflict

Occupancy 精确计算

NVSwitch 拓扑

InfiniBand RDMA
```

后面的 CUDA 章节还会重新讲。

现在没必要一次全部吃掉。

---

# 四十九、你现在最重要的一张脑图

```text
PyTorch

    ↓

CUDA Kernel

    ↓

Grid

    ↓

Block

    ↓

Warp

    ↓

Thread

    ↓

SM

    ├── CUDA Core
    │
    ├── Tensor Core
    │
    ├── Register
    │
    └── Shared Memory / L1

    ↓

L2

    ↓

HBM
```

计算的时候核心问题就是：

```text
数据从哪里来？

↓
搬到哪里？

↓
能重复用多少次？

↓
谁来计算？

↓
最后写回哪里？
```

---

# 五十、以后看任何 GPU 算子，先问两个问题

第一个：

> 我现在是在等计算，还是在等数据？

也就是：

```text
Compute Bound

还是

Memory Bound
```

第二个：

> 如果慢，我应该少搬数据，还是提高计算单元利用率？

如果是 Memory Bound：

```text
减少 HBM Traffic

增加 Data Reuse

做 Kernel Fusion

使用更低精度

优化 Memory Access
```

如果是 Compute Bound：

```text
提高 Tensor Core 利用率

改善矩阵尺寸

增加并行度

使用更高效 GEMM
```

很多 GPU 性能优化，本质都可以从这里开始分析。

---

# 五十一、最后把整篇压缩成 10 句话

1. CPU 更偏低延迟，GPU 更偏高吞吐。

2. 深度学习的大量规则矩阵运算天然适合 GPU。

3. SM 是理解 NVIDIA GPU 执行模型最关键的硬件单位。

4. 一个 Warp 有 32 个线程。

5. GPU 使用 SIMT，大量线程执行相同指令处理不同数据。

6. Warp 内出现不同控制流会导致 Warp Divergence。

7. GPU 核心存储层级可以先记为 Register → Shared/L1 → L2 → HBM。

8. Tensor Core 是矩阵乘加的重要 AI 加速硬件。

9. Arithmetic Intensity 衡量每搬 1 Byte 数据能够做多少 FLOPs。

10. GPU 优化的核心问题，经常就是“到底在等计算，还是在等数据”。

---

# 五十二、学完以后你最好能回答这 12 个问题

```text
1. 为什么不能简单说 GPU 一定比 CPU 快？

2. 为什么 Transformer 特别适合 GPU？

3. SM 是什么？

4. CUDA Core 和 Tensor Core 有什么区别？

5. Warp 是什么？

6. SIMT 是什么意思？

7. Warp Divergence 为什么会影响性能？

8. Grid、Block、Thread 是什么关系？

9. Register、Shared Memory、L2、HBM 有什么区别？

10. Arithmetic Intensity 是什么意思？

11. 为什么 Prefill 更容易 Compute Bound？

12. 为什么 Decode 更容易 Memory Bound？
```

如果这 12 个能自己讲清楚，这篇 GPU Basics 的主干就已经拿下来了。

---

# 五十三、你现在的学习位置

```text
数学基础
   ↓
Transformer
   ↓
PyTorch
   ↓
GPU Hardware       ← 现在
   ↓
CUDA Programming
   ↓
CUDA Memory Model
   ↓
Kernel Optimization
   ↓
GEMM / Reduce / Softmax
   ↓
Attention / FlashAttention
   ↓
分布式训练和推理
```

前面你是从“模型的角度”看：

```text
GEMM
RMSNorm
FlashAttention
KV Cache
Prefill
Decode
```

从现在开始，会逐渐切换到：

```text
SM
Warp
Tensor Core
Shared Memory
Register
HBM
Memory Bandwidth
```

也就是开始真正从 GPU 硬件视角理解：

> 为什么这些 AI Infra 优化会有效。