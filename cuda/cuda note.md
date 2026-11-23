# 1.2 cuda 编程模型

## cuda 程序的执行流程

```text
CPU
│
├── 创建 / 初始化数据
│
├── 在 GPU 上申请显存
│
├── 把数据从 CPU 搬到 GPU
│
├── 启动 Kernel
│
│       ↓
││     GPU
││      ↓
││   大量线程并行计算
││
├── 等待 GPU
│
├── 把结果从 GPU 搬回 CPU
│
└── 后续处理
```

## 函数修饰符

![image-20261123175019425](cuda%20note.assets/image-20261123175019425.png)

## Kernel

一段会让大量 GPU 线程并行执行的函数。

```cpp
__global__ void doubleArray(float* x) {
    ...
}
```

例如这个，CPU 可以启动这个 kernel，让 GPU 产生很多 Thread 去执行这个函数，但是每个 Thread 负责不同数据。达到一个并行的效果。

## Gird/Block/Thread

```text
Grid（网格） = 整个学校——一次 Kernel 启动产生的所有线程的集合
Block（线程块） = 一个班级——Block 内线程可以协作（共享内存、同步）
Thread（线程） = 一个学生——最小的执行单位

一次 kernel 启动，产生一个 Grid。
然后 Grid 往下是 Block
Block 往下是 Thread。
```

## threadIdx,blockIdx

![image-20261123175348541](cuda%20note.assets/image-20261123175348541.png)

```text
blockDim 指的是 block 的维度，也就是一个 block 下面对应的 thread 长啥样；比如 128，就是一个 block 下面有 128 个 thread。

都是从 0 开始编号。
```

## 一维索引

最常用的模式——将线程映射到一维数组：

```cpp
// 全局线程 ID = Block 编号 × Block 大小 + Block 内线程编号
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
```

对于一个 `gridDim=4, blockDim=8` 的配置，全局索引分布如下：

```cpp
Block 0: [ 0  1  2  3  4  5  6  7]
Block 1: [ 8  9 10 11 12 13 14 15]
Block 2: [16 17 18 19 20 21 22 23]
Block 3: [24 25 26 27 28 29 30 31]
```

## Grid-stride Loop

当数据量大于线程总数时，使用循环让每个线程处理多个元素：

```cpp
__global__ void processLargeArray(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // 总线程数

    // 每个线程以 stride 为步长遍历
    for (int i = idx; i < N; i += stride) {
        data[i] = data[i] * 2.0f;
    }
}
```

```text
就是比如只有 128 个 thread，编号 0-127。
那么第 i 个 thread 就负责 mod 128=i 的那些数据。
```

## Kernel 函数

## dim3

```cpp
// 1D：处理向量
dim3 grid(256);        // 256 个 Block
dim3 block(256);       // 每个 Block 256 个线程

// 2D：处理图像/矩阵
dim3 grid(16, 16);     // 16x16 = 256 个 Block
dim3 block(16, 16);    // 每个 Block 16x16 = 256 个线程

// 3D：处理体数据
dim3 grid(8, 8, 8);    // 8x8x8 = 512 个 Block
dim3 block(8, 8, 4);   // 每个 Block 8x8x4 = 256 个线程

```



### [4.1 Kernel 定义规则](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型/#41-kernel-定义规则)

```cpp
__global__ void myKernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
    }
}
```

Kernel 函数的约束：

- 返回类型必须是 `void`
- 不能使用可变参数（variadic）
- 不能是类的虚函数
- 不能递归（Compute Capability < 2.0，现代 GPU 均已支持）
- 参数通过值传递，指针必须指向设备内存

### [4.2 启动语法 `<<<...>>>`](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型/#42-启动语法-)

```cpp
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

| 参数             | 类型            | 必填 | 说明                       |
| :--------------- | :-------------- | :--- | :------------------------- |
| `gridDim`        | `dim3` 或 `int` | 是   | Grid 中 Block 的数量       |
| `blockDim`       | `dim3` 或 `int` | 是   | 每个 Block 中线程的数量    |
| `sharedMemBytes` | `size_t`        | 否   | 动态共享内存大小（默认 0） |
| `stream`         | `cudaStream_t`  | 否   | 执行流（默认 0，即默认流） |

### [4.3 计算启动配置](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型/#43-计算启动配置)

给定数据大小 N 和 Block 大小，计算需要多少个 Block：

```cpp
int N = 1000000;
int blockSize = 256;
int gridSize = (N + blockSize - 1) / blockSize;  // 向上取整

myKernel<<<gridSize, blockSize>>>(input, output, N);
```

📌 **关键点**：`(N + blockSize - 1) / blockSize` 是整数除法向上取整的标准写法。当 N 不是 blockSize 整数倍时，最后一个 Block 中会有一些线程”越界”，因此 Kernel 内必须做边界检查 `if (idx < N)`。

### [4.4 错误检查](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型/#44-错误检查)

Kernel 启动是异步的，错误不会立即被捕获：

```cpp
myKernel<<<grid, block>>>(args);

// 检查启动错误
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}

// 等待执行完成并检查运行时错误
err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    printf("Kernel execution error: %s\n", cudaGetErrorString(err));
}
```

## Block 大小选择

### [5.2 为什么必须是 32 的倍数](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型/#52-为什么必须是-32-的倍数)

GPU 的执行以 **Warp**（32 个连续线程）为最小调度单位。如果 Block 大小不是 32 的倍数，最后一个 Warp 中的部分线程会被浪费：

```
Block 大小 = 100 → 4 个 Warp，但第 4 个 Warp 中只有 4 个线程活跃
Block 大小 = 128 → 4 个 Warp，全部线程都活跃 ✅
Copy
```

### [5.3 Occupancy API（自动选择）](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型/#53-occupancy-api自动选择)

CUDA 提供 API 来自动计算最优 Block 大小：

```cpp
int blockSize;
int minGridSize;

// 自动计算使占用率最大化的 Block 大小
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    myKernel,       // kernel 函数指针
    0,              // 动态共享内存大小
    0               // Block 大小上限（0 = 不限制）
);

int gridSize = (N + blockSize - 1) / blockSize;
myKernel<<<gridSize, blockSize>>>(args);
```

## SM,Block,Warp,Thread 的关系

Block 是：

> CUDA 编程模型里的逻辑协作单位。

Warp 则更接近：

> **GPU 硬件真正调度线程时的重要执行单位。**

可以粗略画：

```text
GPU
│
├── SM 0
│
├── SM 1
│
├── SM 2
│
└── ...
```

当 Kernel 启动以后：

```text
Grid
↓
很多 Block
```

这些 Block 会被调度到不同 SM 上。

例如：

```text
Block 0 → SM 0

Block 1 → SM 1

Block 2 → SM 2

Block 3 → SM 0

...
```

一个 Block 一旦被调度到某个 SM：

> 这个 Block 会在这个 SM 上完成。

然后 Block 内：

```text
Thread
↓
按 Warp 分组
↓
SM 调度 Warp 执行
```

所以可以粗略记：

```text
Grid
↓
Block
↓
分配到 SM
↓
Block 内 Thread 组成 Warp
↓
SM 调度 Warp
```

这张图特别重要。

## Warp

### [6.1 什么是 Warp](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型/#61-什么是-warp)

如果说 Block 是逻辑上的协作单位，那 Warp 就是**物理上的执行单位**。SM（Streaming Multiprocessor）以 Warp 为粒度调度执行——一个 Warp 包含 32 个连续线程，它们在同一时钟周期内执行**同一条指令**（SIMT，Single Instruction Multiple Threads）。

比如：

$$
y_i=x_i\times2
$$

Warp 里面：

```text
Thread 0:
y[0] = x[0] * 2


Thread 1:
y[1] = x[1] * 2


...


Thread 31:
y[31] = x[31] * 2
```

这就是 GPU 特别擅长的模式。

### [6.2 Warp 分歧（Divergence）](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型/#62-warp-分歧divergence)

当 Warp 内的线程走不同的分支路径时，发生 **Warp 分歧**——GPU 不得不串行执行两个分支：

我们希望一个 warp(32 个 thread) 是进入同一个分支的。

```cpp
// ❌ 容易导致分歧
if (threadIdx.x % 2 == 0) {
    // 偶数线程执行这里
    result = pathA(data);
} else {
    // 奇数线程执行这里
    result = pathB(data);
}

// ✅ 以 Warp 为边界分支，避免分歧
if (threadIdx.x / 32 < 2) {
    // 前两个 Warp 执行 pathA
    result = pathA(data);
} else {
    // 后续 Warp 执行 pathB
    result = pathB(data);
}
```

💡 **提示**：只要同一个 Warp 内的 32 个线程走相同分支，就不会有分歧惩罚。分歧发生在 Warp 内部，不同 Warp 之间走不同分支完全没有代价。

### warp 分歧慢的原因

你可以先粗略理解成：

```text
先执行 A 分支
↓
B 那一半线程暂时不干活

再执行 B 分支
↓
A 那一半线程暂时不干活
```

本来希望：

```text
32 个线程齐步走
```

现在变成：

```text
一部分工作
另一部分等着
```

所以效率下降。

# 1.3 cuda 内存模型

## 各种 memory

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

## 本章核心

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

数据移动路线：
HBM
↓
Shared Memory
↓
Register
↓
Compute
```

## Global Memory

### 合并访问 Coalesced Access

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

**核心：GPU 从 HBM 取数据，不是一个一个取；而是按照一定大小的 Memory Transaction 成块地取数据。**

所以：

```text
连续 Thread
+
连续地址
```

通常是非常好的模式。

## Shared Memory

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

### Shared Memory Bank

“**每 32 个元素分成一组 Bank**”，而是 **Shared Memory 本身有 32 个 Bank，连续的 32-bit word 会轮流映射到这 32 个 Bank 上**。

如果先假设每个元素是 `float`，也就是 4 Bytes，那么可以粗略理解成：

$$
\text{bank}(i)=i\bmod 32
$$

所以：

```text
shared[0]  → Bank 0
shared[1]  → Bank 1
...
shared[31] → Bank 31

shared[32] → Bank 0
shared[33] → Bank 1
...
```

所以你刚才说的：

```text
shared[0]
shared[32]
```

确实都会映射到：

$$
\text{Bank 0}
$$

如果**同一个 Warp 里的两个线程，在同一条内存访问指令中，访问 Bank 0 里的不同地址**，例如：

```text
Thread 0 → shared[0]

Thread 1 → shared[32]
```

那么就会发生：

$$
\text{Bank Conflict}
$$

因为它们访问的是：

```text
同一个 Bank
+
不同地址
```

于是这个 Bank 的访问需要部分串行化，性能会下降。

但是有一个重要例外。

如果多个线程读取的是**完全相同的地址**，例如：

```text
Thread 0  → shared[0]
Thread 1  → shared[0]
Thread 2  → shared[0]
...
```

这种情况通常可以进行：

```text
Broadcast
```

所以不按照普通 Bank Conflict 来理解。

你现在可以直接记成：

$$
\text{Bank 编号}
\approx
\text{word index}\bmod 32
$$

以及：

$$
\text{同一个 Warp}
+
\text{同一个 Bank}
+
\text{不同地址}
\Rightarrow
\text{Bank Conflict}
$$

所以你刚才举的：

```text
shared[0]
shared[32]
```

这个例子就是非常标准的 Bank Conflict 直觉。

## Register

对应函数里的局部变量。

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

## Constant Memory

**一小块所有 Thread 都能读取，但 GPU Kernel 不能随便修改的全局只读内存。**

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

# 1.4 vector add

## 为啥 `cudaMalloc` 传的是二重指针

对，关键就在于：**`cudaMalloc` 需要修改你这个指针变量本身。**

你现在有：

```cpp
float* d_A;
```

此时 `d_A` 里面还没有有效地址。

`cudaMalloc` 要做的是：

```text
在 GPU 上申请一块内存
↓
拿到这块内存的地址
↓
把这个地址写进 d_A
```

所以它必须拿到：

```cpp
&d_A
```

而 `d_A` 是 `float*`，那么：

```cpp
&d_A
```

自然就是：

```cpp
float**
```

也就是二级指针。

你可以把它和普通 `malloc` 对比：

```cpp
d_A = (float*)malloc(size);
```

`malloc` 是**直接 return 新地址**。

而 CUDA 的设计是：

```cpp
cudaMalloc(&d_A, size);
```

它的返回值拿去返回：

```cpp
cudaError_t
```

也就是“申请成功还是失败”，所以真正申请出来的地址就通过参数 `&d_A` 写回去。

而 `cudaMemcpy`：

```cpp
cudaMemcpy(d_A, h_A, size, ...);
```

只是说：

> 从 `h_A` 指向的地方复制到 `d_A` 指向的地方。

它**不需要修改 `d_A` 这个指针变量本身**，所以传一级指针就够了。

一句话：

> `cudaMalloc` 要改“指针本身”，所以传 `&d_A`；`cudaMemcpy` 只使用“指针指向的内存”，所以传 `d_A`。

## CUDA_CHECK

```cpp
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

```



所以你可以记：

```cpp
do { ... } while (0)

不是为了循环
而是为了让多行宏包装成“一条语句”
```

这样用户就可以很自然地写：

```cpp
CUDA_CHECK(cudaMalloc(...));
```

比如你写：

```
CUDA_CHECK(cudaMalloc(&d_A, size));
```

预处理器会把宏参数 `call` 直接替换成：

```
cudaMalloc(&d_A, size)
```

所以宏里面这句：

```
cudaError_t err = call;
```

展开后其实就是：

```
cudaError_t err = cudaMalloc(&d_A, size);
```

而 `cudaMalloc(...)` 的返回值本来就是：

```
cudaError_t
```

所以正好赋给 `err`。

## 计时

### `cudaEvent_t`

关键就在这：**`cudaEvent_t` 本身已经是一个“句柄/类似指针”的东西。**

```cpp
cudaEvent_t start;
```

可以粗略理解成：

```cpp
Event* start;
```

所以：

```cpp
cudaEventCreate(&start);
```

需要 `&start`，因为 `Create` 要**给 start 赋一个新地址/句柄**，相当于：

```cpp
void create(Event** p);
```

而：

```cpp
cudaEventRecord(start);
```

这里 `start` 已经指向那个 Event 了。`Record` 是去**修改 start 指向的 Event 对象内部状态**，不是修改 `start` 这个变量本身。

所以一句话：

> `Create(&start)`：要修改 **start 里面存的值**。  
> `Record(start)`：要修改 **start 指向的对象**。

这和：

```cpp
int* p;
malloc(&p);   // 类似这种“给 p 一个地址”的概念
use(p);       // 用 p 指向的东西
```

是一个思路。

### record 向 gpu 队列放标记；sync 是等待标记

对，关键区别是：

```cpp
cudaEventRecord(stop);
```

不是“停下来等 GPU”，而是：

> 往 GPU 队列里插一个 **stop 时间点标记**。

GPU 可能此时还没执行完 kernel。

而：

```cpp
cudaEventSynchronize(stop);
```

是：

> CPU 在这里等，直到 GPU 真正执行到 `stop` 这个标记。

所以顺序相当于：

```text
GPU 队列：
[start标记]
[kernel]
[stop标记]
```

然后：

```cpp
cudaEventSynchronize(stop);
```

就是 CPU 说：“等 GPU 跑到 stop 标记那里再继续。”

所以不能只 `Synchronize(stop)`，因为你得先：

```cpp
cudaEventRecord(stop);
```

告诉 CUDA **“stop 到底在哪儿”**。

一句话记住：

**Record = 放标记，Synchronize = 等标记被执行到。**

## 一份完整代码

```cpp
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(call) \
    do{\
        cudaError_t err=call;\
        if(err!=cudaSuccess) {\
            fprintf(stderr,"CUDA error at %s:%d - %s\n",__FILE__,__LINE__,cudaGetErrorString(err));\
            exit(EXIT_FAILURE);\
        }\
    } while(0)
__global__ void vector_add(const float* A,const float* B,float* C,int N) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N) {
        C[idx]=A[idx]+B[idx];
    }
}

int main() {
    int N=1<<20;
    size_t size=N*sizeof(float);

    // 分配 host 内存
    float* h_A=(float*)malloc(size);
    float* h_B=(float*)malloc(size);
    float* h_C=(float*)malloc(size);

    for(int i=0;i<N;i++) {
        h_A[i]=(float)i;
        h_B[i]=(float)(i*2);
    }
    // 分配 device 内存
    // 因为要修改 d_A 的值，所以我们要把 d_A 的地址传进去
    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    // 把 input 从 host 拷贝到 device
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    // 配置并且启动 kernel
    int block_size=256;
    int grid_size=(N+block_size-1)/block_size;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vector_add<<<grid_size,block_size>>>(d_A,d_B,d_C,N);

    // 先向 gpu 队列放 stop 这个标记
    cudaEventRecord(stop);
    // 再等待 gpu 跑到 stop 标记，再继续 cpu
    // 所以这里 cpu 会阻塞等待；等到 gpu 跑到 stop 再说
    cudaEventSynchronize(stop);
    
    float miliiseconds=0;
    cudaEventElapsedTime(&miliiseconds,start,stop);
    printf("GPU Kernel time: %.4f ms\n",miliiseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // 把结果从 device copy 回 host
    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

    // 在 host 中验证结果
    for(int i=0;i<N;i++) {
        if(fabsf(h_C[i]-(h_A[i]+h_B[i]))>1e-5f) {
            printf("failed\n");
            break ;
        }
    }
    printf("passed\n");

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

# 2.1 warp 与执行模型

## warp

GPU 硬件调度和执行的最小单位，固定包含 **32 个连续线程**。

### [1.2 Warp Scheduler](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/21-warp与执行模型/#12-warp-scheduler)

每个 SM 配备若干 Warp Scheduler（现代架构通常 4 个），每个时钟周期能从就绪的 Warp 中选择一个发射指令。这带来一个关键的延迟隐藏机制：

SM 去调度 warp

```cpp
Warp A 发射内存读取指令 → 等待数据返回（~400 cycles）
Warp Scheduler 切换到 Warp B → 执行计算指令
切换到 Warp C → 执行计算指令
...
Warp A 数据就绪 → 重新调度执行
Copy
```

💡 **提示**：Warp 之间的切换是**零开销的**（zero-overhead context switch），因为每个 Warp 的寄存器状态始终驻留在 SM 上，不需要保存/恢复。这与 CPU 的线程切换有本质区别。

## lane

```text
一个 warp 内部的编号。
一个 warp 通常有 32 个线程，所以：
laneId = 0 ~ 31

threadIdx.x = 0  -> lane 0
threadIdx.x = 1  -> lane 1
...
threadIdx.x = 31 -> lane 31

threadIdx.x = 32 -> 下一个 warp 的 lane 0
threadIdx.x = 33 -> 下一个 warp 的 lane 1
```



## 带 sync 是啥意思

这里的 `_sync` 表示：**mask 里指定的那些 lane，要先都执行到这个指令，再一起完成这次操作。**

比如：

```cpp
__all_sync(0xFFFFFFFF, pred);
```

就是 32 个 lane 都要到这里，CUDA 才能正确判断“是不是所有线程的 `pred` 都为 true”。

所以不是“等所有线程执行完程序”，而是：

> **等参与的线程都到达这个 `_sync` 指令。**

而且如果 `mask` 不是 `0xFFFFFFFF`，那就只等 mask 里指定的线程。

## Warp Shuffle

传统的线程间数据交换需要通过共享内存：

```text
线程A写入共享内存 → __syncthreads() → 线程B读取共享内存
Copy
```

这有两个问题：(1) 需要消耗宝贵的共享内存资源；(2) 需要同步屏障带来延迟。

Warp Shuffle 让同一 Warp 内的线程可以**直接读取彼此的寄存器值**，无需经过任何共享内存，也不需要显式同步——就像同学之间直接传纸条，不用先放到讲台上再取。

### 四个指令

```cpp
T __shfl_sync(unsigned mask, T var, int srcLane, int width=32);
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width=32);
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=32);
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=32);

```

![image-20261123230033910](cuda%20note.assets/image-20261123230033910.png)

```text
注意，怎么去区分 up,down，画数据传输箭头。
我们看 down。
比如 0~31，down 是让 31 的值 -> 20 的感觉。那比如你选中 delta=20，那你 19 其实就没有值，也就是 0 了。
```

### shuffle 是会先拍一个 snapshot，再进行这一轮

对，你问到关键了。**它不会出现你说的“lane 2 先更新，然后 lane 0 读到了 lane 2 更新后的值”这种情况。**

这一句：

```cpp
val += __shfl_down_sync(mask, val, offset);
```

可以理解成两步：

```cpp
tmp = __shfl_down_sync(mask, val, offset); // 先交换旧 val
val = val + tmp;                           // 再更新自己的 val
```

`__shfl_down_sync` 执行时，各个 lane 读取的是**这一轮开始时对方传进去的 `val`**。

**先让所有参与线程拿到这一轮需要的 RHS 值 →同步后， 再各自更新自己的 `val`。**

例如 offset=2 时，这一轮开始：

```text
lane0 = 6
lane1 = 8
lane2 = 10
lane3 = 12
```

Shuffle 先得到：

```text
lane0 读取 lane2 的 10
lane1 读取 lane3 的 12
```

然后才各自更新：

```text
lane0 = 6 + 10 = 16
lane1 = 8 + 12 = 20
```

即使 lane2 自己这轮也在算别的，**lane0 不会读到 lane2 更新之后的新值**。

所以这里没有“谁先加谁后加”的问题。

你可以把每一轮想象成：

> **先对所有 lane 当前的 `val` 拍一张快照 → 根据快照交换数据 → 各自计算新 `val`。**

这正是 shuffle 特别适合做这种 reduction 的原因。

### [4.5 实用示例：Warp 内求和](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/21-warp与执行模型/#45-实用示例warp-内求和)

```cpp
__device__ float warp_reduce_sum(float val) {
    // 蝶形归约：log2(32) = 5 步完成
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // 只有 lane 0 持有正确结果
}
```

归约过程可视化（以 8 线程为例）：

```cpp
初始: [1] [2] [3] [4] [5] [6] [7] [8]

Step 1 (offset=4): 每个线程加上偏移4处线程的值
       [1+5] [2+6] [3+7] [4+8] [5] [6] [7] [8]
       = [6]  [8]  [10]  [12]

Step 2 (offset=2): 每个线程加上偏移2处线程的值
       [6+10] [8+12] [10] [12] ...
       = [16]  [20]

Step 3 (offset=1): 每个线程加上偏移1处线程的值
       [16+20] [20] ...
       = [36]    ← lane 0 持有所有元素之和
Copy
```

## warp vote

```cpp
// 所有线程的 predicate 都为 true 时返回 true
int __all_sync(unsigned mask, int predicate);

// 任意线程的 predicate 为 true 时返回 true
int __any_sync(unsigned mask, int predicate);

// 返回一个 32-bit 位图，每个 bit 对应一个 lane 的 predicate 值
unsigned __ballot_sync(unsigned mask, int predicate);

```

实用场景——提前退出优化：

```cpp
__global__ void search_kernel(int* data, int target, int* found) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int pred = (data[tid] == target);

    // 如果 Warp 内所有线程都没找到，整个 Warp 可以跳过后续处理
    if (__any_sync(0xFFFFFFFF, pred)) {
        // 至少一个线程找到了目标
        if (pred) {
            atomicExch(found, tid);
        }
    }
}
```

### 这里为啥需要 atomic

对，你理解得对。

```cpp
atomicExch(found, tid);
```

意思是：**原子地把 `tid` 写进 `*found`**。

因为可能有很多 thread 同时发现：

```cpp
data[tid] == target
```

它们都会想写同一个：

```cpp
*found
```

如果普通写：

```cpp
*found = tid;
```

就会发生 data race。

`atomicExch` 保证这个写操作是原子的，不会几个线程把这个内存位置“同时写乱”。

而一般情况下：

```cpp
data[0]  // thread 0
data[1]  // thread 1
data[2]  // thread 2
```

每个线程访问不同地址，就没有 data race，所以不需要 atomic。

不过这里即使用了 `atomicExch`，如果多个线程都找到目标，**最终 `found` 是哪个 tid 并不确定**，只是保证写入过程安全。

## warp reduce

```cpp
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void block_reduce_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 每个线程加载一个元素
    float val = (tid < N) ? input[tid] : 0.0f;

    // 第一步：Warp 内归约
    val = warp_reduce_sum(val);

    // 第二步：每个 Warp 的 lane 0 将结果写入共享内存
    __shared__ float warp_sums[32];  // 最多 32 个 Warp（1024线程/32）
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    if (laneId == 0) {
        warp_sums[warpId] = val;
    }
    __syncthreads();

    // 第三步：第一个 Warp 对 warp_sums 做最终归约
    int numWarps = blockDim.x / 32;
    val = (threadIdx.x < numWarps) ? warp_sums[threadIdx.x] : 0.0f;

    if (warpId == 0) {
        val = warp_reduce_sum(val);
    }

    // lane 0 写出本 Block 的归约结果
    if (threadIdx.x == 0) {
        output[blockIdx.x] = val;
    }
}

```

```cpp
// 第三步：第一个 Warp 对 warp_sums 做最终归约
    int numWarps = blockDim.x / 32;
    val = (threadIdx.x < numWarps) ? warp_sums[threadIdx.x] : 0.0f;

我要让第一个 warp 对我得到的每组 warp sum 进行最后一次求和，所以我要先初始化他们的 value；这里默认 blockDim.x 是 32 的倍数。
    
然后我让第一个 warp 去做求和就好了。为啥要用第一个 warp 而不是分散开，因为 SM 的调度单位是 warp，而且我们用到了 shuffle
```

