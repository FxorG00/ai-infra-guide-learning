# 1.3 CUDA 内存模型：真正理解 Register、Shared Memory、Global Memory 和 Tiling

这一篇非常重要。

上一章 1.2 解决的是：

> **这么多 CUDA Thread 到底怎么组织？**

也就是：

```text
Kernel
↓
Grid
↓
Block
↓
Thread
↓
Warp
↓
SM
```

而这一章开始回答另一个问题：

> **这些 Thread 要处理的数据到底放在哪里？**

以及：

> **为什么同样做完全一样的计算，有的 CUDA Kernel 能快几十倍？**

这篇真正的核心不是让你背：

```text
Global Memory
Shared Memory
Register
Constant Memory
Texture Memory
Unified Memory
```

而是建立一个最重要的 GPU 性能思想：

> **数据离计算单元越近，通常访问越快；但是越快的存储通常也越小。**

所以 CUDA 优化大量时候都在想：

```text
数据现在在哪里？

↓

我要不要把它搬近一点？

↓

搬进来以后能不能多用几次？

↓

能不能减少访问 HBM？
```

这一章最终会把你带到一个非常重要的概念：

```text
Tiling
```

也就是：

> **从 Global Memory 分块搬数据到 Shared Memory，然后重复利用。**

这就是以后 GEMM、FlashAttention 等高级优化里反复出现的核心思想。

---

# 一、为什么 CUDA 内存模型这么重要？

GPU 的计算能力非常强。

尤其是 Tensor Core，做矩阵乘的时候速度非常夸张。

但是有一个问题：

> **计算单元再快，也得先拿到数据才能算。**

你可以把 GPU 想象成：

```text
Tensor Core / CUDA Core
=
超级快的工厂


HBM
=
远处的大仓库
```

如果工厂：

```text
1 秒能加工一亿件产品
```

但是仓库：

```text
每秒只能送一百万件原材料
```

那工厂再牛也没用。

因为它大部分时间都在：

```text
等数据
```

所以 CUDA 优化里非常重要的一句话就是：

> **很多 Kernel 的瓶颈不是计算，而是内存访问。**

---

# 二、GPU 为什么要有这么多层 Memory？

如果只设计一种内存，会出现一个矛盾：

```text
我们想要：

容量巨大
+
特别快
+
特别便宜
```

现实里这三个很难同时满足。

所以 GPU 采用了：

```text
Memory Hierarchy
```

也就是多层内存结构。

可以先记成：

```text
            快
            ↑

        Register
            │
    Shared Memory / L1
            │
          L2
            │
   Global Memory / HBM

            ↓
            慢
```

同时：

```text
            小
            ↑

        Register
            │
    Shared Memory / L1
            │
          L2
            │
           HBM

            ↓
            大
```

所以最核心的规律就是：

> **越靠近计算单元的存储，通常越快，但容量越小。**

---

# 三、先把各种 Memory 的作用域搞清楚

这个比背速度数字更重要。

可以先记：

```text
Thread
│
├── Register
└── Local Memory


Block
│
└── Shared Memory


整个 Grid / GPU
│
├── Global Memory
├── Constant Memory
└── Texture Memory
```

什么意思？

比如：

```text
Register
```

是：

> 一个 Thread 自己用。

而：

```text
Shared Memory
```

是：

> 同一个 Block 的 Thread 可以共享。

而：

```text
Global Memory
```

是：

> 整张 GPU 上所有 Thread 都可以访问。

这三个是你现在最应该真正吃透的。

---

# 四、Global Memory 是什么？

Global Memory 可以先理解成：

> **GPU 的主存。**

物理上主要就是你一直说的：

```text
HBM
```

比如 PyTorch：

```python
x = torch.randn(
    1024,
    1024,
    device="cuda"
)
```

这个大型 Tensor 的主体数据通常就在：

```text
GPU Global Memory / HBM
```

里面。

模型的：

```text
Weights

Activations

Gradients

Optimizer States

KV Cache
```

也主要放这里。

---

# 五、为什么 Global Memory 很重要？

因为它：

```text
容量大
```

比如几十 GB。

但是问题是：

```text
距离 SM 比较远

访问延迟高
```

所以 CUDA 性能优化里经常出现一个目标：

> **少访问 Global Memory。**

不是说完全不用。

那当然不可能。

而是：

> **一份数据既然从 HBM 搬过来了，就尽可能多利用几次。**

这个思想后面会一直出现。

---

# 六、cudaMalloc 是在干什么？

例如：

```cpp
float* d_data;

size_t size = N * sizeof(float);

cudaMalloc(&d_data, size);
```

可以先理解成：

> **在 GPU Global Memory 里申请一块空间。**

注意这里经常用命名：

```text
h_x
```

表示：

```text
Host 上的数据
```

而：

```text
d_x
```

表示：

```text
Device 上的数据
```

这只是程序员常用命名习惯。

---

# 七、cudaMemcpy 又是在干什么？

例如：

```cpp
cudaMemcpy(
    d_data,
    h_data,
    size,
    cudaMemcpyHostToDevice
);
```

意思就是：

```text
CPU Memory
    ↓
GPU Memory
```

也就是：

```text
Host
↓
Device
```

反过来：

```cpp
cudaMemcpy(
    h_result,
    d_result,
    size,
    cudaMemcpyDeviceToHost
);
```

就是：

```text
GPU
↓
CPU
```

所以一个最基本 CUDA 程序经常是：

```text
CPU 创建数据
↓
cudaMalloc
↓
cudaMemcpy Host → Device
↓
Launch Kernel
↓
GPU 计算
↓
cudaMemcpy Device → Host
↓
CPU 拿到结果
```

---

# 八、Global Memory 优化的第一大原则：Coalesced Access

这一块非常重要。

中文一般叫：

```text
合并访问
```

这是你现在必须建立的 CUDA 性能直觉之一。

我们已经知道：

> 一个 Warp 有 32 个 Thread。

假设：

```text
Thread 0
Thread 1
Thread 2
...
Thread 31
```

都要去 Global Memory 读取数据。

GPU 最喜欢的是：

```text
Thread 0  → data[0]
Thread 1  → data[1]
Thread 2  → data[2]
Thread 3  → data[3]
...
Thread 31 → data[31]
```

也就是说：

> **相邻 Thread 访问相邻的内存地址。**

这就是最理想的访问模式之一。

---

# 九、为什么连续访问会快？

因为 GPU 从 HBM 取数据的时候，并不是：

```text
Thread 0 要 4 Bytes

那我只取 4 Bytes
```

然后：

```text
Thread 1 再单独取 4 Bytes
```

它会按照一定大小的：

```text
Memory Transaction
```

成块地取数据。

假设一个 Warp 的 32 个 Thread 都读取连续的 float。

一个 float：

$$
4\ \mathrm{Bytes}
$$

32 个 float：

$$
32 \times 4 = 128\ \mathrm{Bytes}
$$

如果这 128 Bytes 在内存中连续而且对齐得很好，

GPU 就有机会用很少的内存事务把需要的数据一起拿回来。

所以：

```text
连续 Thread
+
连续地址
```

通常是非常好的模式。

---

# 十、什么叫糟糕的访问？

假设：

```text
Thread 0  → data[0]

Thread 1  → data[100]

Thread 2  → data[200]

Thread 3  → data[300]

...
```

这些地址散得到处都是。

那么 GPU 就可能需要发起很多独立 Memory Transaction。

结果就是：

```text
真正想要的数据不多

但是为了拿这些数据
却搬了很多没用的数据
```

于是：

```text
Memory Bandwidth 利用率下降
```

---

# 十一、所以 Coalesced Access 可以怎么记？

一句话：

> **让同一个 Warp 中相邻的 Thread，尽可能访问连续的 Global Memory 地址。**

这句话以后会反复出现。

---

# 十二、为什么 Thread → Data 的映射这么重要？

你上一章学了：

```cpp
int idx =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

然后：

```cpp
float value = data[idx];
```

你现在应该发现：

这种最朴素的写法其实有一个很好的特点：

```text
Thread 0 → data[0]
Thread 1 → data[1]
Thread 2 → data[2]
...
```

所以天然容易形成：

```text
Coalesced Access
```

这就是为什么：

> **线程索引怎么映射到数据索引，不只是代码正确性问题，也是性能问题。**

---

# 十三、AoS 和 SoA 是什么？

原文这里用了一个很经典的例子。

假设我们有很多粒子，每个粒子有：

```text
x
y
z
vx
vy
vz
```

一种数据结构叫：

```text
AoS
```

全称：

```text
Array of Structures
```

例如：

```cpp
struct Particle {
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

Particle particles[N];
```

内存里大概是：

```text
particle0:
x y z vx vy vz

particle1:
x y z vx vy vz

particle2:
x y z vx vy vz
```

---

# 十四、为什么 AoS 可能不适合某些 GPU 访问？

假设每个 Thread 只需要：

```text
x
```

那么：

```text
Thread 0 → particles[0].x

Thread 1 → particles[1].x

Thread 2 → particles[2].x
```

但是两个 x 之间夹着：

```text
y
z
vx
vy
vz
```

所以地址不是紧挨着的。

这会让合并访问变得不那么理想。

---

# 十五、SoA 又是什么？

SoA：

```text
Structure of Arrays
```

例如：

```cpp
struct Particles {
    float x[N];
    float y[N];
    float z[N];

    float vx[N];
    float vy[N];
    float vz[N];
};
```

这时候内存更像：

```text
x0 x1 x2 x3 x4 ...

y0 y1 y2 y3 y4 ...

z0 z1 z2 z3 z4 ...
```

如果每个 Thread 都读取 x：

```text
Thread 0 → x[0]

Thread 1 → x[1]

Thread 2 → x[2]
```

地址就连续了。

于是更容易：

```text
Coalesced Access
```

所以你现在不用记：

```text
SoA 永远比 AoS 好
```

真正应该理解的是：

> **数据布局会影响 Warp 的内存访问模式。**

---

# 十六、接下来进入这篇最重要的 Memory：Shared Memory

Shared Memory 是：

> **位于 SM 上的高速片上存储，并且同一个 Block 内的 Thread 可以共享。**

可以继续用之前那个比喻：

```text
Global Memory
=
城市外的大仓库


Shared Memory
=
办公室里的共享文件柜


Register
=
每个人自己桌上的纸
```

所以 Shared Memory：

```text
比 Global Memory 小很多

但是快很多
```

---

# 十七、为什么 Shared Memory 这么重要？

因为它可以让我们做：

```text
HBM 数据
↓
搬到 Shared Memory
↓
很多 Thread 重复使用
↓
最后再写回 HBM
```

也就是：

> **把远处的数据搬近一点，然后反复利用。**

这几乎就是 CUDA 高性能编程最核心的套路之一。

---

# 十八、Shared Memory 怎么声明？

静态 Shared Memory 可以写：

```cpp
__shared__ float tile[256];
```

意思是：

> 这个 Block 拥有一块 256 个 float 的 Shared Memory。

注意：

```text
不是每个 Thread 一份
```

而是：

> **整个 Block 共享这一份。**

---

# 十九、每个 Thread 可以一起往 Shared Memory 搬数据

例如：

```cpp
int idx =
    blockIdx.x * blockDim.x
    + threadIdx.x;

tile[threadIdx.x] = input[idx];
```

假设一个 Block 有：

```text
256 Threads
```

那么：

```text
Thread 0
→ 把 input[某个位置] 搬到 tile[0]

Thread 1
→ 搬到 tile[1]

Thread 2
→ 搬到 tile[2]

...
```

于是整个 Block：

> **大家一起合作，把一块 Global Memory 数据搬进 Shared Memory。**

这叫：

```text
Cooperative Loading
```

后面 GEMM 会疯狂使用这个思想。

---

# 二十、为什么搬完以后需要 __syncthreads()？

假设：

```text
Thread 0 很快
Thread 1 比较慢
Thread 2 更慢
```

Thread 0 搬完自己的数据以后：

```text
tile[0]
```

已经准备好了。

但 Thread 200 可能：

```text
tile[200]
```

还没写完。

如果 Thread 0 马上开始：

```cpp
读取 tile[200]
```

就可能读到：

```text
还没准备好的数据
```

所以需要：

```cpp
__syncthreads();
```

它的意思大致是：

> **同一个 Block 的所有 Thread 都先到这里集合，所有人都到了以后，再一起继续。**

可以理解成：

```text
Thread 0 到了
↓
等

Thread 1 到了
↓
等

...

最后一个 Thread 到了
↓
大家继续
```

---

# 二十一、所以 Shared Memory 经常和同步一起出现

典型模式：

```cpp
shared[...] = global[...];

__syncthreads();

// 使用 shared memory

__syncthreads();
```

第一道同步：

> 确保所有数据都搬进来了。

第二道同步有时是为了：

> 确保大家都使用完当前 tile，才能覆盖这块 Shared Memory 加载下一批数据。

这个模式后面 GEMM 会非常常见。

---

# 二十二、动态 Shared Memory 是什么？

有时候 Shared Memory 大小不想在编译时写死。

可以：

```cpp
extern __shared__ float sharedData[];
```

然后 Kernel Launch 时：

```cpp
kernel<<<
    grid,
    block,
    sharedMemSize
>>>(...);
```

这里第三个参数就是：

```text
动态 Shared Memory 大小
```

比如：

```cpp
int sharedMemSize =
    blockSize * sizeof(float);
```

这样同一个 Kernel 可以在运行时决定 Shared Memory 大小。

现在知道这个语法就够了。

```text
这里的 extern __shared__ 是 CUDA 的固定语法；表示这里只声明这块动态 shared memory，不在编译器确定他的大小；实际空间是运行时，kernel launch 提供。

对比 cpp 的 extern，是一个 declaration 表示这个变量实际的 definition 在别的文件。
```



---

# 二十三、Shared Memory 很快，是不是随便访问都行？

也不是。

Shared Memory 内部还被划分成多个：

```text
Bank
```

原文用的模型是：

```text
32 个 Bank
```

这和一个 Warp 的：

```text
32 个 Thread
```

正好能很好对应。

---

# 二十四、什么是 Shared Memory Bank？

可以把 Shared Memory 想成：

```text
Bank 0
Bank 1
Bank 2
...
Bank 31
```

如果一个 Warp 的 32 个 Thread：

```text
Thread 0 → Bank 0
Thread 1 → Bank 1
Thread 2 → Bank 2
...
Thread 31 → Bank 31
```

那么大家可以很好地并行访问。

---

# 二十五、什么是 Bank Conflict？

假设：

```text
Thread 0
```

和：

```text
Thread 16
```

在同一个时刻都访问：

```text
同一个 Bank
```

但访问的是：

```text
不同地址
```

那这个 Bank 没办法同时把两个不同位置都服务掉。

于是访问需要被拆开。

这就叫：

```text
Bank Conflict
```

也就是：

> **同一个 Warp 中多个 Thread 同时访问同一 Bank 的不同地址。**

结果：

```text
原本应该并行
↓
变成部分串行
↓
性能下降
```

---

# 二十六、连续访问 Shared Memory 通常很好

例如：

```cpp
float val = shared[threadIdx.x];
```

那么：

```text
Thread 0 → shared[0]

Thread 1 → shared[1]

...

Thread 31 → shared[31]
```

通常能够很好地分散到不同 Bank。

所以访问效率高。

---

# 二十七、为什么 stride 可能制造 Bank Conflict？

例如：

```cpp
shared[threadIdx.x * 2]
```

访问：

```text
Thread 0 → shared[0]

Thread 1 → shared[2]

Thread 2 → shared[4]

...
```

由于 Bank 编号会周期性循环，

某些 Thread 最终会落到同一个 Bank。

于是产生：

```text
Bank Conflict
```

---

# 二十八、所有 Thread 读同一个地址会怎样？

这是一个特殊情况。

比如：

```cpp
float val = shared[0];
```

整个 Warp 的 32 个 Thread 全部读取同一个地址。

这种情况 GPU 可以进行：

```text
Broadcast
```

所以并不按照普通的 Bank Conflict 处理。

可以把它理解成：

> “既然大家都要同一个东西，那我拿一次然后广播给你们。”

---

# 二十九、为什么矩阵转置喜欢加 Padding？

这是 Shared Memory 优化里一个很经典的技巧。

假设：

```cpp
__shared__ float tile[32][32];
```

当按某个方向访问矩阵列时，

可能导致很多 Thread 映射到相同 Bank。

于是发生严重 Bank Conflict。

经典技巧是：

```cpp
__shared__ float tile[32][33];
```

注意只多了一列：

```text
32 → 33
```

这不是为了真的存更多有意义的数据。

它只是：

> **故意把每一行起始位置错开一点。**

从而让不同 Thread 访问列时，不再老撞到相同 Bank。

这叫：

```text
Padding
```

---

# 三十、这个 Padding 思想很重要

它告诉你：

> **GPU 优化有时候不是减少数据，而是故意改变数据布局，让硬件访问模式更舒服。**

以后你会看到很多：

```text
Padding

Alignment

Layout Transformation

Swizzle
```

本质都和这种思想有关：

> **数学上数据一样，但物理布局不同，性能可以差很多。**

---

# 三十一、接下来是最快的一层：Register

Register 是：

> **每个 Thread 私有的最快存储。**

比如：

```cpp
float acc = 0.0f;
```

这种局部标量变量通常希望待在 Register 里。

例如：

```cpp
for (...) {
    acc += ...
}
```

如果：

```text
acc 一直在 Register
```

那就非常舒服。

因为不用每次跑去 HBM。

---

# 三十二、Register 为什么特别适合 accumulator？

例如矩阵乘：

$$
C_{ij}
=
\sum_k A_{ik}B_{kj}
$$

一个 Thread 或一个线程组可能需要不断做：

```text
sum += a * b
```

这个：

```text
sum
```

如果一直存在 Register：

```text
读取快
更新快
不用来回访问 HBM
```

所以高性能 GEMM 会非常重视：

```text
Register Accumulation
```

后面你会重新碰到。

---

# 三十三、Register 是不是越多越好？

不是。

因为一个 SM 的 Register 总量是有限的。

假设：

```text
一个 Thread 使用 20 个 Register
```

那么一个 SM 可能容纳很多 Thread。

但是如果：

```text
一个 Thread 使用 200 个 Register
```

那么 Register 很快就被吃光。

于是一个 SM 同时能驻留的：

```text
Thread
Warp
Block
```

数量会下降。

也就是你上一章刚认识的：

```text
Occupancy
```

可能下降。

所以这里开始出现一个经典 trade-off：

```text
更多 Register
↓
Thread 自己计算更舒服
↓
减少内存访问


但是


更多 Register
↓
SM 能同时放的 Thread / Warp 变少
↓
Occupancy 可能下降
```

所以不是越多越好。

---

# 三十四、什么是 Register Spilling？

如果编译器发现：

```text
这个 Thread 需要太多局部数据
```

Register 放不下。

一些变量就会被：

```text
Spill
```

出去。

这叫：

```text
Register Spilling
```

它们会进入所谓的：

```text
Local Memory
```

---

# 三十五、Local Memory 最容易被名字骗

你可能看到：

```text
Local
```

会觉得：

> “是不是非常靠近 Thread，所以很快？”

不是。

这里一定要记：

> **CUDA 的 Local Memory 在逻辑上是 Thread 私有，但物理上并不是 Register 那种片上高速存储。**

它实际会落在 Global Memory 一侧，并通过 Cache 加速。

所以：

```text
Register
```

和：

```text
Local Memory
```

性能完全不是一个级别。

---

# 三十六、所以 Thread 私有不等于快

这是一个很重要的概念。

```text
Register
```

是：

```text
Thread 私有
+
片上
+
快
```

而：

```text
Local Memory
```

也是：

```text
Thread 私有
```

但它可能在：

```text
Global Memory 那一侧
```

所以：

```text
作用域
```

和：

```text
物理位置
```

是两个不同概念。

---

# 三十七、Constant Memory 是什么？

Constant Memory 可以理解成：

> **一小块所有 Thread 都能读取，但 GPU Kernel 不能随便修改的全局只读内存。**

原文给出的容量是：

```text
64 KB
```

它有专门的：

```text
Constant Cache
```

---

# 三十八、Constant Memory 什么情况下特别舒服？

假设一个 Warp：

```text
32 个 Thread
```

全部要读取：

```text
同一个参数
```

例如：

```text
某个卷积系数

某个常数表中的同一个位置
```

这时候 Constant Cache 可以很好地：

```text
读取一次
↓
广播给整个 Warp
```

所以这种模式非常适合 Constant Memory。

---

# 三十九、什么时候不适合 Constant Memory？

如果：

```text
Thread 0 → constant[0]

Thread 1 → constant[17]

Thread 2 → constant[53]

Thread 3 → constant[91]
```

也就是：

> 一个 Warp 里的 Thread 各读各的地址。

那 Constant Memory 的广播优势就没了。

甚至访问可能被串行化。

所以：

```text
Constant Memory
```

最喜欢：

> **同一个 Warp 的 Thread 读取相同地址。**

---

# 四十、你现在怎么判断要不要用 Constant Memory？

简单记：

```text
数据比较小

+

只读

+

很多 Thread 会读取相同值
```

这时候可以考虑：

```text
Constant Memory
```

否则很多场景直接使用 Global Memory + Cache 就可以了。

---

# 四十一、Texture Memory 是什么？

这一部分你现在可以浅学。

Texture Memory 是通过专门的 Texture Cache 去访问数据。

它历史上特别适合：

```text
图像
二维空间数据
插值
不规则空间访问
```

比如：

```text
线性插值

双线性插值

边界 Clamp / Wrap
```

这些硬件都能提供一些支持。

---

# 四十二、为什么 AI / LLM 里 Texture Memory 没那么核心？

因为现代 GPU：

```text
L1 / L2 Cache
```

已经很强。

而深度学习主流 Kernel 更多关注：

```text
Global Memory
Shared Memory
Register
Tensor Core
```

所以你当前阶段对 Texture Memory：

> **知道它存在，知道主要用于特殊空间访问 / 图像场景即可。**

不用深挖。

---

# 四十三、Unified Memory 是什么？

Unified Memory 中文一般叫：

```text
统一内存
```

它解决的是一个编程上的麻烦。

传统 CUDA：

```text
CPU 一份指针

GPU 一份指针

自己 cudaMalloc

自己 cudaMemcpy
```

Unified Memory 想做：

> **CPU 和 GPU 使用同一个地址空间和同一个指针。**

例如：

```cpp
float* data;

cudaMallocManaged(
    &data,
    N * sizeof(float)
);
```

然后：

```text
CPU 可以 data[i]

GPU Kernel 也可以 data[i]
```

看起来方便很多。

---

# 四十四、Unified Memory 的“魔法”是什么？

它不代表：

> CPU 和 GPU 真的同时拥有一块神奇的零延迟内存。

背后仍然需要：

```text
页面迁移
```

例如：

```text
CPU 先访问
↓
Page 在 CPU Memory

GPU 开始访问
↓
触发迁移
↓
Page 搬到 GPU

CPU 后面又访问
↓
可能再搬回来
```

所以：

```text
Unified Memory
```

主要是：

> **让系统帮你管理数据迁移。**

而不是：

> **消灭数据迁移。**

这两个一定不要混。

---

# 四十五、Unified Memory 为什么方便但不一定最快？

因为系统自动迁移的时候：

```text
什么时候迁？

迁多少？

Kernel 什么时候突然需要？
```

可能没有你手动控制得精准。

于是可能出现：

```text
Kernel 正在运行
↓
突然缺页
↓
开始迁移
↓
等
```

这会产生额外开销。

所以性能敏感的代码通常更喜欢：

> **明确知道数据什么时候该在哪，然后主动控制传输。**

---

# 四十六、Prefetch 又是什么？

既然问题是：

```text
GPU 用到的时候才发现 Page 还在 CPU
```

那我们可以提前说：

> “GPU 等会儿要用这批数据，你现在就先搬过去。”

这就是：

```text
Prefetch
```

例如：

```cpp
cudaMemPrefetchAsync(
    data,
    size,
    device
);
```

意思就是：

> 提前把 Unified Memory 的页面迁移到指定设备附近。

这样真正 Kernel 开始的时候，就少一点突然 Page Fault 的情况。

---

# 四十七、这一堆 Memory 到底怎么选？

现在先把最常见的几类记成：

```text
单个 Thread 自己频繁使用的小变量

→ Register
```

```text
同一个 Block 内 Thread 要共享数据

→ Shared Memory
```

```text
大型数组 / 模型数据 / 普通 Tensor

→ Global Memory
```

```text
小型只读数据
并且很多 Thread 读取同一个值

→ Constant Memory
```

```text
想快速开发
CPU / GPU 访问模式比较复杂

→ Unified Memory
```

这基本就是你现在最需要的内存选型直觉。

---

# 四十八、但这一章真正的大 Boss 是 Tiling

前面所有内容最后其实都在铺垫：

```text
Tiling
```

Tiling 中文：

```text
分块
```

它是 GPU 优化里极其重要的模式。

---

# 四十九、为什么需要 Tiling？

假设做矩阵乘：

$$
C = AB
$$

最朴素的方法可能是：

> 每算一个输出元素，就不停从 Global Memory 读取 A 和 B。

例如：

$$
C_{ij}
=
\sum_k A_{ik}B_{kj}
$$

为了算一个：

```text
C[i][j]
```

要读取很多：

```text
A[i][k]

B[k][j]
```

问题是：

> 同一个 A、B 元素其实会被很多不同输出重复使用。

如果每次都从 HBM 重新读：

```text
太浪费了。
```

---

# 五十、Tiling 的核心想法

不要：

```text
需要一次
↓
去 HBM 读一次
```

而是：

```text
从 HBM 搬一块数据
↓
放 Shared Memory
↓
这个 Block 的 Thread 反复使用
↓
算很多次
↓
再处理下一块
```

所以 Tiling 最核心的一句话：

> **搬一次，用多次。**

这个你一定要记住。

---

# 五十一、Tiling 的数据流

可以画成：

```text
Global Memory / HBM
        │
        │ cooperative load
        ▼
   Shared Memory
        │
        │ reuse many times
        ▼
      Register
        │
        ▼
     Compute
```

相比：

```text
HBM
↓
算一下

HBM
↓
再算一下

HBM
↓
再算一下
```

前者的 HBM 访问次数少很多。

---

# 五十二、为什么这会提高 Arithmetic Intensity？

你之前学：

$$
AI
=
\frac{\mathrm{FLOPs}}
{\mathrm{Memory\ Bytes}}
$$

假设一块数据从 HBM 搬进来以后只算一次：

```text
FLOPs 少

Bytes 多
```

那么 AI 低。

但如果：

```text
同样搬一次

然后在 Shared Memory / Register 里重复算很多次
```

那么：

```text
Memory Bytes 没明显增加

FLOPs 增加很多
```

所以：

$$
AI
$$

提高。

于是 Kernel 更有机会从：

```text
Memory Bound
```

往：

```text
Compute Bound
```

移动。

---

# 五十三、这就是为什么 GEMM 特别适合 Tiling

矩阵乘里天然有巨大的：

```text
Data Reuse
```

例如一个：

```text
A[i][k]
```

可以参与很多不同：

```text
C[i][j]
```

的计算。

一个：

```text
B[k][j]
```

也会被很多不同：

```text
C[i][j]
```

使用。

所以只要合理地把 A、B Tile 搬到 Shared Memory，

就可以：

```text
搬一次
↓
整个 Block 反复使用
```

这就是高性能 GEMM 的基础。

---

# 五十四、Tiled GEMM 大概在干什么？

原文给出的代码思想可以先不逐行啃。

先看流程：

```text
Block 负责 C 的一小块 Tile
        ↓
Thread 协作加载 A 的一个 Tile
        ↓
Thread 协作加载 B 的一个 Tile
        ↓
__syncthreads()
        ↓
大家使用 Shared Memory 中的 A / B
        ↓
做很多乘加
        ↓
__syncthreads()
        ↓
加载下一批 Tile
        ↓
继续
```

最后每个 Thread 得到自己的：

```text
accumulator
```

然后写回：

```text
C
```

---

# 五十五、为什么中间要两次 __syncthreads()？

第一道：

```text
Load A / B Tile
↓
__syncthreads()
```

是为了保证：

> 所有人都把 Tile 搬好了，才能开始用。

第二道：

```text
计算完当前 Tile
↓
__syncthreads()
```

是为了保证：

> 所有人都用完当前 Shared Memory 数据以后，才能把它覆盖成下一批 Tile。

所以：

```text
Shared Memory
+
Thread Collaboration
+
Synchronization
```

是绑定在一起的。

---

# 五十六、FlashAttention 为什么也叫 Tiling 思想？

你前面已经学过 FlashAttention。

现在你可以重新看：

```text
Q
K
V
```

都很大。

普通 Attention 容易：

```text
HBM
↔
巨大 Attention Matrix
↔
HBM
```

FlashAttention 则：

```text
把 Q/K/V 分块
↓
搬到片上高速存储
↓
处理一个 Tile
↓
Online Softmax
↓
直接更新输出
↓
不把完整 Attention Matrix 落 HBM
```

所以它和 GEMM Tiling 背后有同一条思想：

> **把数据分块搬近，然后尽可能在片上完成更多计算。**

---

# 五十七、所以这一章最重要的不是背 6 种内存

真正应该建立的是：

```text
Global Memory
=
容量大
但是远


Shared Memory
=
Block 内共享
很适合做 Tile Cache


Register
=
Thread 私有
最快
适合 accumulator / 临时变量
```

然后：

```text
HBM
↓
Shared Memory
↓
Register
↓
Compute
```

这条数据移动路径。

这才是以后 CUDA Kernel 优化的主线。

---

# 五十八、把 Global、Shared、Register 放在一起比较

可以先这么记：

| Memory | 谁能访问 | 大小 | 速度 | 典型用途 |
|---|---|---|---|---|
| Register | 单 Thread | 最小 | 最快 | accumulator、局部变量 |
| Shared Memory | 同 Block | 小 | 很快 | Tile、Block 内数据交换 |
| Global Memory | 全 GPU | 很大 | 较慢 | 大数组、Tensor、模型权重 |

这个表你现在最好能自己默写出来。

---

# 五十九、Constant / Texture / Unified 现在学到什么程度？

我建议：

```text
Constant Memory

知道：
小、只读、Warp 同地址广播舒服
```

```text
Texture Memory

知道：
针对特殊空间访问和图像场景
当前 LLM 学习里不是重点
```

```text
Unified Memory

知道：
CPU / GPU 可以用同一个指针
系统自动迁移页面
方便但不代表最快
```

够了。

不要现在在这三种 Memory 上花大量时间。

---

# 六十、这一篇现在必须真正吃透的第一件事：Coalescing

你必须能判断：

```cpp
data[idx]
```

为什么通常访问比较好。

以及：

```cpp
data[idx * stride]
```

为什么 stride 很大时容易变差。

本质就是：

```text
同一个 Warp
↓
相邻 Thread
↓
最好访问相邻地址
```

---

# 六十一、第二件事：Shared Memory + __syncthreads()

你必须理解：

```text
为什么 Shared Memory 是 Block 共享

为什么 Thread 可以协作搬数据

为什么搬完要同步

为什么使用完有时候还要再次同步
```

这会直接决定你后面能不能学懂 Reduce 和 GEMM。

---

# 六十二、第三件事：Bank Conflict

现在至少要理解：

> **Shared Memory 虽然快，但如果同一个 Warp 的 Thread 访问 Bank 的模式很差，也会发生串行化。**

以及知道经典方法：

```text
Padding
```

可以改变 Bank 映射。

目前不用自己推复杂 Bank Index。

---

# 六十三、第四件事：Register 和 Occupancy 的 Trade-off

必须知道：

```text
Register 多
↓
单 Thread 可能更舒服
↓
减少 Memory Access
```

但：

```text
Register 多
↓
SM 能同时放的 Thread / Warp 变少
↓
Occupancy 可能下降
```

所以：

> **Register 不是越多越好。**

---

# 六十四、第五件事：Tiling

这是这一篇最重要的性能思想。

直接记：

> **Tiling = 分块加载 + 本地复用 + 减少 Global Memory 访问。**

或者更短：

> **搬一次，用多次。**

以后你学：

```text
Reduce

GEMM

Softmax

FlashAttention
```

都会不停回来用这句话。

---

# 六十五、现在把上一章和这一章接起来

上一章学：

```text
Grid
↓
Block
↓
Thread
```

这一章补上 Memory：

```text
Thread
│
└── Register

Block
│
└── Shared Memory

Grid / GPU
│
└── Global Memory
```

然后硬件上：

```text
Thread
↓
Warp
↓
SM
├── Register
├── Shared Memory / L1
└── Compute Unit

          ↕
         L2

          ↕
         HBM
```

到这里 CUDA 的基础地图才真正开始完整。

---

# 六十六、用一个 Block 的视角看整个数据流

假设一个 Block 要处理一块数据。

最经典流程：

```text
1. 每个 Thread 根据 idx 找到自己负责的数据

2. Thread 从 Global Memory 读取

3. 大家一起把数据放进 Shared Memory

4. __syncthreads()

5. 在 Shared Memory 上反复计算 / 交换数据

6. Thread 用 Register 保存自己的局部 accumulator

7. __syncthreads()

8. 最终结果写回 Global Memory
```

这个流程以后会出现无数次。

---

# 六十七、你现在可以重新理解 CUDA Kernel 优化在干什么

朴素 Kernel：

```text
HBM
↓
Compute
↓
HBM
↓
Compute
↓
HBM
↓
Compute
```

优化 Kernel：

```text
HBM
↓
Shared Memory
↓
Register
↓
Compute
↓
Compute
↓
Compute
↓
Compute
↓
HBM
```

核心就是：

> **尽量把昂贵的远距离访问，变成便宜的本地重复访问。**

---

# 六十八、这篇目前可以先跳浅的部分

你现在不用深挖：

```text
Texture Object API

__ldg 的各种细节

Shared / L1 配置 API

Unified Memory 页面迁移的底层机制

cudaMemPrefetchAsync 的高级调优

精确 Bank 映射的复杂情况
```

以后需要时再补。

---

# 六十九、这一篇你现在应该能回答这 12 个问题

```text
1. 为什么 GPU 需要多层 Memory Hierarchy？

2. Register、Shared Memory、Global Memory 分别属于谁？

3. Global Memory 为什么又大又慢？

4. 什么叫 Coalesced Access？

5. 为什么相邻 Thread 最好访问相邻地址？

6. Shared Memory 为什么比 Global Memory 快？

7. 为什么 Shared Memory 通常要配合 __syncthreads()？

8. 什么叫 Bank Conflict？

9. 为什么 tile[32][33] 有时候比 tile[32][32] 更快？

10. Register Spilling 是什么？

11. 为什么 Local Memory 名字里有 Local，却不代表它很快？

12. Tiling 为什么能减少 Global Memory Traffic？
```

如果这些都能讲明白，这篇主干就吃下来了。

---

# 七十、最后把整篇压成 8 句话

第一句：

> GPU 的计算能力很强，所以很多 Kernel 真正卡的是数据搬运。

第二句：

> GPU Memory 越靠近计算单元通常越快，但容量越小。

第三句：

> Global Memory 容量最大，但访问昂贵，所以要尽量减少不必要的访问。

第四句：

> Global Memory 的第一大优化原则之一，是让 Warp 内相邻 Thread 尽量访问连续地址。

第五句：

> Shared Memory 是 Block 内 Thread 共享的高速片上暂存空间。

第六句：

> Register 是单 Thread 私有的最快存储，但用太多会影响 Occupancy，甚至发生 Spilling。

第七句：

> Shared Memory 访问还要注意 Bank Conflict。

第八句：

> Tiling 的核心就是“搬一次，用多次”，这是后面 GEMM 和 FlashAttention 的核心优化思想之一。

---

# 七十一、你现在真正应该记住的最终数据流

```text
                Global Memory / HBM
                        │
                        │  expensive
                        ▼
                Shared Memory
                        │
                        │  fast reuse
                        ▼
                    Register
                        │
                        ▼
                CUDA / Tensor Core
                        │
                        ▼
                    Compute
```

对于高性能 CUDA Kernel，我们经常想做的就是：

```text
减少 HBM 访问

↓

提高 Shared Memory / Register 数据复用

↓

让计算单元一直有数据可算
```

这就是 CUDA 内存优化最核心的思想。

---

# 七十二、下一篇 1.4 为什么非常值得亲手写？

因为你到现在已经学了：

```text
1.2：

Thread 到底是谁？
Grid / Block / Thread 怎么组织？


1.3：

Thread 要处理的数据放在哪里？
Global / Shared / Register 有什么区别？
```

下一篇：

```text
1.4 第一个实用 Kernel：向量加法
```

会第一次把：

```text
cudaMalloc

cudaMemcpy

Kernel

Grid

Block

Thread Index

Global Memory

Boundary Check

cudaFree
```

全部放在一个真正能运行的程序里。

所以 1.4 我建议不要只看。

最好自己亲手：

```text
编译一次
运行一次
改一次 blockSize
故意删一次边界判断
再看一次结果
```

这样前面 1.2 和 1.3 的很多概念才会真正从“我听懂了”变成：

> **“哦，原来代码里就是这个东西。”**

```` :chatgpt-content-reference{index="0"}