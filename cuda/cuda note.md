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

# 2.2 内存访问优化

## K,M,G,T,P

很简单，按 1000 倍一级记：

- `K` = kilo = \(10^3\) = 千
- `M` = mega = \(10^6\) = 百万
- `G` = giga = \(10^9\) = 十亿
- `T` = tera = \(10^{12}\) = 万亿
- `P` = peta = \(10^{15}\)

所以：

```text
1 TFLOPS = 10^12 FLOPS
1 GFLOPS = 10^9 FLOPS
1 TB/s   = 10^12 Byte/s
1 GB/s   = 10^9 Byte/s
```

你这个例子：

```text
2 TB/s × 0.083 FLOP/Byte
= 2 × 10^12 × 0.083
≈ 1.66 × 10^11 FLOP/s
= 166 GFLOPS
```

因为：

```text
10^11 = 100 × 10^9
```

所以是 `166 G`。

最好记一句：

**K=3，M=6，G=9，T=12，P=15，每次 +3。**

## CUDA 反人类的编号！

对，CUDA 这里确实容易反直觉。

它的编号规则是：

```cpp
linear_tid =
    threadIdx.x
    + threadIdx.y * blockDim.x
    + threadIdx.z * blockDim.x * blockDim.y;
```

也就是说 **x 变化最快，其次 y，再其次 z**。

所以对于：

```cpp
blockDim = (32, 8)
```

编号是：

```text
(0,0) -> 0
(1,0) -> 1
(2,0) -> 2
...
(31,0) -> 31

(0,1) -> 32
(1,1) -> 33
...
```

不是：

```text
(0,0)
(0,1)
(0,2)
...
```

你可以把它想成 C/C++ 二维数组的 row-major：

```cpp
a[y][x]
```

也是 `x` 这一维在内存里连续变化。

所以对于 `block(32,8)`，**一个 warp 正好就是固定 y，然后 x=0~31。** 这点对理解 coalesced memory access 特别重要。

## matrix transpose

```cpp
#define TILE_DIM 32
#define ROW_STRIDE 8
extern "C" __global__ void solve(const float* A, const float* B, float* C, int N, int M, int K) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1];
    // 考虑块内的 (i,j) 取取得 input[j][i] input[j+8][i]...
    // 那么对于同一条指令，也就是固定的 offset，同一个 warp 对应 j 一样
    // 所以第一维一样，warp 内只有 i 不一样，但是连续，所以在内存上访问也是连续的
    // 符合我们 global memory 合并访问的需要
    // 需要注意，我们需要先得到这个 block 负责的 A 的区间
    // 对第一维，那么就是每 tile_dim 个对应一个 block
    int i=threadIdx.x,j=threadIdx.y;
    int input_row_offset=blockIdx.y*TILE_DIM;
    int input_col_offset=blockIdx.x*TILE_DIM;
    int real_row,real_col;
    for(int offset=0;offset<TILE_DIM;offset+=ROW_STRIDE) {
        real_row=j+offset+input_row_offset,real_col=i+input_col_offset;
        if(real_row<N&&real_col<M) {
            tile[j+offset][i]=A[real_row*M+real_col];
        }
    }
    __syncthreads();
    // 还是一样 我们负责把 output 的 (j+offset,i) 找到，并且往回填充
    // 比如原先负责的行是 32~63 列是 0~31
    // 那么比如 [33][1]->[1][33]
    // 那么现在我们要填充 output 的位置是行 0~31，列 32~63
    int output_row_offset=input_col_offset,output_col_offset=input_row_offset;
    for(int offset=0;offset<TILE_DIM;offset+=ROW_STRIDE) {
        real_row=j+offset+output_row_offset,real_col=i+output_col_offset;
        if(real_row<M&&real_col<N) {
            C[real_row*N+real_col]=tile[i][j+offset];
        }
    }
}
```

## blockIdx 编号规则

其实 `blockIdx` 的编号规则和 `threadIdx` **几乎完全一样**，只是一个给 block 编号，一个给 thread 编号。

比如你这样定义：

```
dim3 block(32, 8);
dim3 grid(3, 2);

kernel<<<grid, block>>>();
```

![image-20261124175327309](cuda%20note.assets/image-20261124175327309.png)

意味着有 3×2 = 6 个 block，每个 block 有 32×8 = 256 个线程。

每个格子是一个 block，内部都有独立的 256 个线程。

如果想计算 block 的一维编号，也是 x 变化最快：

```
int block_id = blockIdx.y * gridDim.x + blockIdx.x;
```

不过，**block 负责哪些数据，是由你自己写代码决定的，不是 CUDA 自动分配的。**

比如我们之前的矩阵转置，每个 block 负责一个 32×32 的数据区域：

```
int offset_row = blockIdx.y * 32;
int offset_col = blockIdx.x * 32;
```

那么：

| blockIdx (x,y) | 负责的行 | 负责的列 让 x 负责列 |
| -------------- | -------- | -------------------- |
| (0,0)          | 0\~31    | 0\~31                |
| (1,0)          | 0\~31    | 32\~63               |
| (2,0)          | 0\~31    | 64\~95               |
| (0,1)          | 32\~63   | 0\~31                |
| (1,1)          | 32\~63   | 32\~63               |
| (2,1)          | 32\~63   | 64\~95               |

记住三个层级：

- `gridDim`：整个 grid 有多少个 block。
- `blockIdx`：当前 block 在 grid 里的坐标。
- `threadIdx`：当前 thread 在 block 里的坐标。

另外，block 的编号顺序不代表实际执行顺序，GPU 可以自行调度。

## [4. 向量化加载](https://caomaolufei.github.io/AIInfraGuide/guides/模块二-cuda编程与算子优化/22-内存访问优化/#4-向量化加载)

### `reinterpret_cast`：换一种方式看同一块内存

假设：

```cpp
address.s_addr
```

是一个 4-byte 整数。原来的指针类型是：

```cpp
const std::uint32_t*
```

但现在想逐 byte 读取它，所以把地址重新解释成：

```cpp
const unsigned char*
```

内存没有发生复制或转换：

```text
原来：
[address.s_addr：4 bytes 整体]

reinterpret_cast 后：
bytes[0] bytes[1] bytes[2] bytes[3]
```

这正是 `reinterpret_cast` 的典型用途：

```text
同一块内存
-> 换一种底层类型观察
```

C++ 允许通过 `char*`、`unsigned char*` 或 `std::byte*` 查看任意对象的底层 byte representation，所以这里合法。

### [4.1 什么是向量化加载](https://caomaolufei.github.io/AIInfraGuide/guides/模块二-cuda编程与算子优化/22-内存访问优化/#41-什么是向量化加载)

向量化加载指的是用一条指令读取多个连续元素（如 `float4` 一次读 4 个 float，16 Bytes）。好处是：

1. **减少指令数**：4 次 `LDG.32` → 1 次 `LDG.128`
2. **提高带宽利用率**：单次事务传输更多有效数据
3. **减少调度开销**：更少的内存指令排队

### [4.2 常用向量类型](https://caomaolufei.github.io/AIInfraGuide/guides/模块二-cuda编程与算子优化/22-内存访问优化/#42-常用向量类型)

| 📊 类型    | 大小     | 等价        |
| :-------- | :------- | :---------- |
| `float2`  | 8 Bytes  | 2 个 float  |
| `float4`  | 16 Bytes | 4 个 float  |
| `int4`    | 16 Bytes | 4 个 int    |
| `double2` | 16 Bytes | 2 个 double |

💡 **提示**：GPU 一条 cache line 为 128 Bytes。一个 Warp 的 32 个线程每人加载一个 `float4`（16B）= 512B，需要请求 4 条 cache line。对比每人加载一个 `float`（4B）= 128B 只需 1 条 cache line。虽然 cache line 请求数增多了，但每条加载指令处理的数据量是 4 倍，总指令数减少 75%。

### [4.3 基本用法](https://caomaolufei.github.io/AIInfraGuide/guides/模块二-cuda编程与算子优化/22-内存访问优化/#43-基本用法)

```cpp
// 标量版本：每个线程加载 1 个 float
__global__ void scalar_copy(float* dst, float* src, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        dst[tid] = src[tid];
    }
}

// 向量化版本：每个线程加载 4 个 float
__global__ void vectorized_copy(float* dst, float* src, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid * 4;  // 每个线程负责4个元素

    if (idx + 3 < N) {
        // 就是换个视角看这片内存，我们把 src 指向的内存用 float4* 的视角去看，这样它走一步是移动了 float4 的
        // reinterpret_cast<float4*>(src) 也就是这个指针是 float4* 类型的
        float4 val = reinterpret_cast<float4*>(src)[tid];
        reinterpret_cast<float4*>(dst)[tid] = val;
    }
}
```

### [4.4 对齐要求](https://caomaolufei.github.io/AIInfraGuide/guides/模块二-cuda编程与算子优化/22-内存访问优化/#44-对齐要求)

向量化加载有严格的对齐要求：

```cpp
// float4 要求 16B 对齐
// cudaMalloc 分配的地址是 256B 对齐的 → 满足

// 但如果有偏移，需要特殊处理
float* base = ...; // 256B 对齐
float* offset_ptr = base + 3;  // 偏移 12B，不满足 16B 对齐

// ✅ 安全做法：先标量处理头部未对齐部分，再向量化处理主体
__global__ void safe_vectorized(float* data, int offset, int N) {
    // 处理头部未对齐的元素
    int align_offset = (4 - (offset % 4)) % 4;
    // ... 标量处理前 align_offset 个元素

    // 主体向量化处理
    float4* aligned_ptr = reinterpret_cast<float4*>(data + offset + align_offset);
    // ...
}
Copy
```

## memory bound 优化思路

```text
Kernel 慢
↓
它是不是 Memory-Bound？
│
├── 不是
│     ↓
│   看计算
│
└── 是
      ↓
检查 Global Memory
      ↓
是否 Coalesced？
│
├── 否
│     ↓
│   改 Thread → Data Mapping
│   改 Data Layout
│   AoS → SoA
│
└── 是
      ↓
有没有 Shared Memory？
      ↓
有没有 Bank Conflict？
│
├── 有
│     ↓
│   Padding / Swizzle
│
└── 没有
      ↓
能不能 Vectorized Load？
      ↓
float4 / int4
      ↓
是否已经接近 Bandwidth Peak？
│
├── 是
│     ↓
│   基本优化完成
│
└── 否
      ↓
减少冗余访问
增加 Data Reuse
```

# 2.3 Occupancy 与资源分配

## Occupancy

原文用了一个非常好理解的比喻：

```text
SM = 停车场

Warp = 车

SM 最大 Warp 数 = 总车位

当前 Active Warp = 当前停了多少辆车
```

假设一个 SM：

```text
最多能驻留 64 个 Warp
```

现在实际只能驻留：

```text
32 个 Warp
```

那么 Occupancy：

$$
\mathrm{Occupancy}
=
\frac{32}{64}
=
50\%
$$

正式定义就是：

$$
\mathrm{Occupancy}
=
\frac{\mathrm{每个\ SM\ 上活跃\ Warp\ 数}}
{\mathrm{每个\ SM\ 支持的最大\ Warp\ 数}}
$$

---

## 为啥它重要

```text
时间线：
Warp 0: [计算] [等待内存...400cycles...] [计算]
Warp 1:        [计算] [等待内存...400cycles...] [计算]
Warp 2:               [计算] [等待内存...400cycles...]
...

如果活跃 Warp 足够多，调度器总能找到就绪的 Warp 来填充等待期

那么在 warp0 访问内存需要等待的 400 cycles 期间，GPU 会调度其他就绪的 warp 在 SM 上进行计算。
假设一个 warp 能提供 4 cycles 的有效计算。
那么我们至少需要 100 个 warp 才能填满 400 周期的等待时间。

所以我们需要活跃 warp 足够多。
```

## SM 拥有的资源是有限的

比如按照原文的 A100 示例：

```text
每 SM 最大 Thread：
2048

每 SM 最大 Warp：
64

每 SM 最大 Block：
32

Register：
65536

Shared Memory：
164 KB

每 Block 最大 Thread：
1024
```

## 资源分配以 block 为单位

```text
一个 block 要么完整进入 SM，要么完全不进入。如果一个 SM 剩余资源很多，但是不够一个 block，那这部分就会浪费。
```

## 影响 Occupancy 的三大因素

### Register

寄存器以 **256 个为一组**（Allocation Granularity）分配给一个 Warp。也就是分配给每个 warp 的寄存器数量要对 256 对齐。

**示例**（A100，65536 寄存器/SM）：

| 每线程寄存器数 | 每 Warp 实际需要寄存器数  | SM 可容纳 Warp 数 | Occupancy |
| :------------- | :------------------------ | :---------------- | :-------- |
| 32             | 32×32=1024                | 65536/1024=64     | 100%      |
| 48             | 48×32=1536                | 65536/1536=42     | 42/64≈65% |
| 64             | 64×32=2048                | 65536/2048=32     | 50%       |
| 128            | 128×32=4096               | 65536/4096=16     | 25%       |
| 255            | 255×32=8160 → 对齐到 8192 | 65536/8192=8      | 12.5%     |

### shared memory

| 每 Block 共享内存 | SM 可容纳 Block 数              | 每 Block 256线程时总线程        | Occupancy |
| :---------------- | :------------------------------ | :------------------------------ | :-------- |
| 8 KB              | 20（但受最大 Block 数 32 限制） | min(20,32)×256=5120→cap at 2048 | 100%      |
| 32 KB             | 5                               | 5×256=1280                      | 62.5%     |
| 64 KB             | 2                               | 2×256=512                       | 25%       |
| 164 KB            | 1                               | 1×256=256                       | 12.5%     |

### block 大小

Block 大小同时受两个限制：

1. 每 Block 最大 1024 个线程
2. 每 SM 最大 Block 数量（A100 为 32）**就是限制你一个 SM 最多 32 个 block**

**一个容易被忽视的限制**：如果你的 Block 大小很小（如 32 个线程 = 1 个 Warp），即使每个 Block 资源消耗很低，也可能被”每 SM 最大 Block 数”限制：

| Block 大小      | 每 SM Block 数上限 | 总 Warp 数 | Occupancy |
| :-------------- | :----------------- | :--------- | :-------- |
| 32 (1 Warp)     | 32（Block 数限制） | 32         | 50%       |
| 64 (2 Warps)    | 32（Block 数限制） | 64         | 100%      |
| 128 (4 Warps)   | 16（2048/128）     | 64         | 100%      |
| 256 (8 Warps)   | 8（2048/256）      | 64         | 100%      |
| 512 (16 Warps)  | 4（2048/512）      | 64         | 100%      |
| 1024 (32 Warps) | 2（2048/1024）     | 64         | 100%      |

📌 **关键点**：Block 大小建议至少 128 或 256，既能避免被 Block 数限制，又能让编译器有更多优化空间。通常 **256** 是个不错的默认选择。

## [📝 总结](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/23-occupancy与资源分配/#-总结)

本质上还是一个 trade off，需要实际跑出来再去调。

| 核心概念       | 要点                                                         |
| :------------- | :----------------------------------------------------------- |
| Occupancy 定义 | 活跃 Warp 数 / SM 最大 Warp 数                               |
| 延迟隐藏       | Occupancy 越高，越能通过 Warp 切换掩盖内存延迟               |
| 寄存器限制     | 每线程寄存器越多 → Occupancy 越低；可用 `__launch_bounds__` 控制 |
| 共享内存限制   | Block 共享内存越大 → SM 容纳 Block 越少 → Occupancy 越低     |
| Block 大小限制 | block size 太小，导致 block 数量多，但是可能被”每 SM 最大 Block 数”限制 |
| 非越高越好     | Compute-bound Kernel 中低 Occupancy + 高数据复用可能性能更优 |
| 调优方法       | Nsight Compute 定位瓶颈 → 针对性调整 → 实测验证              |

# 2.4 同步与原子操作

## memory fence

**Memory Fence（内存栅栏）的核心作用就两个：定顺序、刷缓存。** 它保证在**“栅栏”之前执行的内存写入**，一定会在“栅栏”之后的代码执行前，真正**对目标范围内的其他线程可见**。

> 至于目标范围怎么定，看调用哪一个。
>
> ```cpp
> // Block 级：确保写入对同一 Block 内的线程可见
> __threadfence_block();
> 
> // Device 级：确保写入对同一 GPU 上所有线程可见
> __threadfence();
> 
> // System 级：确保写入对系统内所有设备（包括 CPU、其他 GPU）可见
> __threadfence_system();
> 
> ```
>
> 

### 为什么需要它？看一个经典场景

假设线程 A 算完一个数，放进 `data`，然后把 `flag` 改成 `1`，通知线程 B 来取：

C++

```cpp
// 线程 A（发数据）
data = 42;            // 1. 写入计算结果
__threadfence();      // <--- 内存栅栏
flag = 1;             // 2. 举旗子通知线程 B

// 线程 B（等数据）
while (flag == 0);    // 死等旗子举起来
int res = data;       // 读取结果
```

如果没有中间那行 `__threadfence()`，线程 B 很可能读到错的旧数据，原因有两个：

- **指令重排**：编译器或 GPU 硬件为了优化速度，可能会把 `flag = 1` 挪到 `data = 42` 前面执行。
- **缓存延迟**：线程 A 写的 `data = 42` 可能还停留在它的局部缓存或写缓冲区里没来得及同步出去，而 `flag = 1` 却先一步被线程 B 看到了。

加上 Fence 之后，就像在代码中间立了一道**单向闸门**：线程 A 执行到这里会停顿一下，确保闸门前的所有写入（`data = 42`）都彻底同步到了其他线程能看见的地方，才允许执行后面的 `flag = 1`。

## atomicCAS 实现其他的原子操作

```cpp
// atomicCAS 语义：
// old = *addr;
// if (old == compare) *addr = val;
// return old;

// 用 CAS 实现 atomicAdd for double（旧架构不直接支持 double atomicAdd）
__device__ double atomicAddDouble(double* addr, double val) {
    unsigned long long int* addr_as_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                       __double_as_longlong(
                           __longlong_as_double(assumed) + val));
    } while (assumed != old);
    // CAS 失败意味着其他线程修改了值，需要重试

    return __longlong_as_double(old);
}

```

这两张图分别讲了两件事：**原子操作为什么会慢（5.3）**，以及**怎么用最底层的 `atomicCAS` 自己造原子操作（5.4）**。

---

## 一、5.3 原子操作的性能代价：本质就是“排队”

原子操作的特点是“同一时间只允许一个线程改同一个内存地址”，你可以把它想象成**上只有一个坑位的厕所**。快不快，全看有多少人同时抢同一个坑位：

* **大家改不同地址**：每人一个坑位，不用排队，速度和普通读写差不多。
* **同一 Warp 内抢同一个地址**：一个 Warp 有 32 个线程，本来是齐头并进的，现在被迫排成一列一个个进（串行化），最差会慢 32 倍。
* **跨 Block 大量线程抢同一个地址**：成千上万个线程挤在同一个坑位门口排队，性能直接崩盘，变成严重瓶颈。

至于底下说的延迟数字，是因为“厕所离得远近不同”：
* **全局内存**在芯片外面或远端（L2 / DRAM），跑一趟要花 **400~600 个时钟周期**。
* **共享内存**就在计算单元（SM）内部，离得近，跑一趟只要 **20~100 个时钟周期**。

---

## 二、5.4 `atomicCAS`：万能原子操作

这段代码看着吓人，其实核心逻辑非常简单，拆成两部分看就通了。

### 1. `atomicCAS` 到底在干嘛？

CAS 是 **Compare-And-Swap（比较并交换）** 的缩写。它的语义翻译成人话就是：

> “帮我看一眼地址 `*addr` 里现在的值是不是我以为的 `compare`。**如果是**，说明没人动过它，帮我把它改成新值 `val`；**如果不是**，就别改。不管成没成功，都把刚才看到的旧值 `old` 返回给我。”

### 2. 下面那段 `atomicAddDouble` 代码为什么写得那么绕？

旧版 GPU 硬件没有直接提供 `double` 类型的原子加法，但提供了 `atomicCAS`，所以程序员得自己用 CAS 拼一个出来。代码里有两个让人眼花的地方：

**第一，为什么满屏都是 `unsigned long long int`？**
因为硬件自带的 `atomicCAS` 只认 64 位整数（`unsigned long long int`），不认 `double`。但它们俩占的内存大小一样（都是 64 位），所以代码里做了一堆类型转换（`__double_as_longlong` 等），纯属**“套个整数马甲骗过硬件”**，算加法时再转回 `double`。

**第二，`do ... while` 循环在干嘛？**
这就是 CAS 的经典套路——**“先猜、再试、被插队就重来”**：

1. **记下旧值**：先把内存里当前的值读出来，存在 `old` 里。
2. **存为假设值**：`assumed = old;` —— 我假设内存里现在就是这个值。
3. **算新值并尝试写入**：调用 `atomicCAS`，意思是“如果内存里还是 `assumed`，就帮我写成 `assumed + val`”，同时把内存里**实际**的旧值存回 `old`。
4. **检查有没有被插队（`while (assumed != old)`）**：
   * **如果 `assumed == old`**：说明我刚才算加法的时候，没有别的线程插队改过它！写入成功，循环结束。
   * **如果 `assumed != old`**：坏了，刚才有别的线程抢先改了这个地址！我的写入自动作废，而且 `old` 里现在装的正好是别人刚改完的最新值，于是回到第 2 步，拿这个最新值重新算、重新试，直到成功为止。

这就是为什么说 `atomicCAS` 是“万能”的——只要把第 3 步里的加法换成减法、乘法、取最大值，任何硬件不支持的原子操作都能用这个“试错循环”写出来。

## 几个原子操作优化性能的代码

### 分层规约

```text
就是先让一个 warp 的 sum 给记到 lane 0 的 val 上面。

然后对于一个 block，我们要求得其所有 warp 的 lane 0 的 val 的和。

那么对一个 block 我开一个 shared memory 上的 block_sum，然后让 val atomicAdd 即可。

然后对于每个 block 的 thread0，我要把这个 block 给 atomicAdd 到全局的 sum。
```

```cpp
__global__ void hierarchical_reduce(float* input, float* output, int N) {
    __shared__ float block_sum;
    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (gid < N) ? input[gid] : 0.0f;

    // 第一层：Warp 内归约（无需原子操作）
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 第二层：每个 Warp 的 lane 0 原子加到共享内存
    // 只有 blockDim.x/32 个线程竞争 → 冲突很小
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&block_sum, val);
    }
    __syncthreads();

    // 第三层：每个 Block 的 thread 0 原子加到全局结果
    // 只有 gridDim.x 个线程竞争（远少于总线程数）
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

```

## shared memory 当桶

```text
这份代码默认，grid 只有 1 维 block，比如 5 个 block；然后一个 block 也是只有 1 维 thread，比如 256 个 thread。

然后我们让每个 thread 负责 data[tid]。但是我们先计数到 block 的 shared_memory 上，再从 shared memory 计数到全局。
```



```cpp
// ❌ 每个线程直接对全局内存做 atomicAdd
__global__ void histogram_naive(int* data, int* hist, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        atomicAdd(&hist[data[tid]], 1);  // 全局内存原子，慢
    }
}

// ✅ 先在共享内存聚合，再写回全局
__global__ void histogram_shared(int* data, int* hist, int N, int numBins) {
    __shared__ int local_hist[256];  // 假设 bin 数 <= 256

    // 初始化局部直方图
    if (threadIdx.x < numBins) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // 在共享内存上做原子加（快得多）
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        atomicAdd(&local_hist[data[tid]], 1);
    }
    __syncthreads();

    // 将局部结果原子加回全局（只有 numBins 次全局原子操作）
    if (threadIdx.x < numBins) {
        atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
    }
}

```

## 每个 thread 维护私有计数器，register

```cpp
__global__ void privatized_histogram(int* data, int* hist, int N) {
    // 每个线程维护私有计数器（适用于 bin 数很少的情况）
    int private_hist[4] = {0, 0, 0, 0};  // 假设只有4个 bin

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        private_hist[data[i]]++;  // 纯寄存器操作，无冲突
    }

    // 最终写回（大幅减少原子操作次数）
    for (int bin = 0; bin < 4; bin++) {
        if (private_hist[bin] > 0) {
            atomicAdd(&hist[bin], private_hist[bin]);
        }
    }
}

```

## Cooperative Groups

**Cooperative Groups（协作组）本质上是 CUDA 的“自由拉群”工具。** 它把原本死板的线程同步机制包装成了灵活的“线程组对象”，让你既可以**往小了切**（比如只同步 8 个或 16 个线程），也可以**往大了扩**（直接同步整个 GPU 上的所有 Block）。

### 一、它有什么用？（解决了传统 CUDA 的三个痛点）

1. **突破固定粒度**：以前只有两个极端——要么全 Block 一起等（`__syncthreads()`），要么 32 线程的 Warp 一起等（`__syncwarp()`）。现在想按 4 人、8 人、16 人分组同步都行。
2. **实现官方“全局屏障”**：还记得前面第 4 节截图里说的“GPU 传统上没有全局屏障”吗？Cooperative Groups 直接补上了这块拼图，支持跨所有 Block 的 `grid.sync()`。
3. **代码更安全、更易复用**：以前调用一个 `__device__` 函数，你根本不知道里面藏没藏同步操作，一不小心放进 `if` 分支就死锁。现在必须把“线程组”当作参数传进去，从接口上就强制规定了哪些线程必须一起参与。

---

### 二、逐段拆解你贴的代码

#### 1. 7.2 常用线程组：怎么“拉群”
这段代码展示了三种最常用的拉群方式：
* `cg::this_thread_block()`：把当前整个 Block 拉成一个群，`block.sync()` 就完全等价于传统的 `__syncthreads()`。
* `cg::tiled_partition<16>(block)`：把当前 Block 均匀切成若干个 16 人的固定小群（Tile）。调用 `tile16.sync()` 时，只同步这 16 个人，完全不管隔壁小组在干嘛。
* `cg::coalesced_threads()`：只把当前**活跃**的线程临时拉群。比如代码进了某个 `if` 分支，原本 32 个线程只有 10 个满足条件，这个函数就把这 10 个活着的线程凑成一个临时群，避免等待那些没进分支的线程导致卡死。

#### 2. 7.3 协作组归约：对比 6.1 看爽点在哪里
这段代码做的事和前面 6.1 的 Warp 归约一模一样，只是换成了 Cooperative Groups 的写法：
* **写法更干净**：6.1 里要手写 `for (int offset = 16; ...)` 和难懂的掩码 `__shfl_down_sync(0xFFFFFFFF, val, offset)`；这里直接用 `warp.size() / 2` 和 `warp.shfl_down(val, offset)`，不用自己操心掩码。
* **通用性极强**：如果以后需求变了，你想改成每 16 个线程做一次归约，只需把 `<32>` 改成 `<16>`，归约函数内部的逻辑一行都不用动。
* **换皮属性**：`block.group_index().x`、`block.group_dim().x`、`block.thread_rank()` 其实就是以前 `blockIdx.x`、`blockDim.x`、`threadIdx.x` 的面向对象写法。

#### 3. 7.4 Grid 级同步：全 GPU 集合点名
这是 Cooperative Groups 最重磅的功能。代码分两阶段：
1. **第一阶段**：全 GPU 所有线程各自把 `data[gid]` 乘 2。
2. **`grid.sync()`**：全 GPU 的所有 Block 停下来互相等待，确保第一阶段彻底算完、结果全局可见。
3. **第二阶段**：安全地去读隔壁位置（可能属于另一个 Block 算出来）的数据。

---

### 三、为什么 7.4 特别警告“可能死锁”？

这是理解 Grid 级同步最关键的一个底层常识：

* **普通启动方式是“分批上场”**：假设你的 GPU 一次最多只能同时装下 100 个 Block，而你启动了 1000 个 Block。平时 GPU 会先跑 100 个，等它们彻底结束腾出位置，再换下一批 100 个上场。
* **加上 `grid.sync()` 后就会卡死**：如果前 100 个 Block 跑到 `grid.sync()` 停下来，死等后面 900 个 Block 过来集合；但后面 900 个 Block 因为 GPU 没空位根本上不了场——两边互相干等，直接**死锁**。

所以用 `grid.sync()` 必须满足两个硬条件：
1. 启动时必须换用专门的 `cudaLaunchCooperativeKernel`（不能用普通的 `<<<grid, block>>>`），它会在启动前检查硬件能不能一次性装下所有 Block，装不下直接报错，防止跑起来死锁。
2. **Grid 大小有上限**：总 Block 数绝对不能超过 GPU 能同时驻留的最大数量。因此如果数据量超大，通常要配合“网格跨步循环（Grid-Stride Loop）”，让有限的常驻 Block 自己循环处理完所有数据。

### 可以直接拿到一维编号

| **写法**              | **含义**                     | **一维情况下等价于**                               |
| --------------------- | ---------------------------- | -------------------------------------------------- |
| `tile.thread_rank()`  | 线程在当前 Tile 内的编号     | `threadIdx.x % tile_size`（32 线程时就是 lane ID） |
| `block.thread_rank()` | 线程在当前 Block 内的编号    | `threadIdx.x`                                      |
| `grid.thread_rank()`  | 线程在整个 Grid 里的全局编号 | `blockIdx.x * blockDim.x + threadIdx.x`            |

# 性能分析

## 带宽

* **有效带宽（Effective Bandwidth）**
  * 笔记里写“处理”，换成**“读写（或传输）”**会更准确——“处理”容易和计算吞吐量（FLOPS）混淆，带宽衡量的始终是数据搬运的速度。
  * **为什么叫“有效”**：它只计算你的算法**真正需要**从显存读入和写出的数据量，不包含因为访存不合并（uncoalesced）等原因被硬件浪费掉的额外传输。
  * **我要知道，读了多少数据，写了多少。**
  * 常用计算公式：
    $$\text{有效带宽} = \frac{\text{读取字节数} (B_r) + \text{写入字节数} (B_w)}{\text{Kernel 执行时间} (t)}$$
  
* **理论峰值带宽（Theoretical Peak Bandwidth）**
  * 你的理解非常到位，就是 GPU 显存硬件规格决定的**最高传输上限**（由显存位宽、显存频率等决定）。
  * 实际跑程序时几乎不可能达到 100%（一般优化得很好的访存密集型 Kernel 能跑到 80%~90% 就很优秀了）。

在性能分析时，通常就是把这两个数相除（$\frac{\text{有效带宽}}{\text{理论峰值带宽}}$），算出**带宽利用率**，从而判断你的 Kernel 在访存上还有多少优化空间。可以传输这么多数据。

## Nsight Compute

```bash
ncu \
    --kernel-name regex:block_reduce_kernel \
    --launch-skip 5 \
    --launch-count 1 \
    ./reduce
```

### bandwith 与 DRAM throughput 的区别

| **指标名称**                             | **单位** | **统计视角**                            | **核心公式**                                                 |
| ---------------------------------------- | -------- | --------------------------------------- | ------------------------------------------------------------ |
| **Effective Bandwidth** （有效带宽）     | `GB/s`   | **算法逻辑视角** （只算真正有用的数据） | `算法必须读写的字节数 ÷ 耗时`                                |
| **DRAM Throughput** （NCU 里的显存吞吐） | `%`      | **硬件物理视角** （总线实际跑过的全算） | `(显存总线实际流量 ÷ 耗时) ÷ 理论峰值`                       |
| **带宽利用率** （Bandwidth Utilization） | `%`      | **统称** （分物理和有效两种）           | • 物理利用率 = **DRAM Throughput** • 有效利用率 = `有效带宽 ÷ 理论峰值` |

#### 为什么要区分“物理”和“有效”？用运快递就能懂

假设你有一条公路，每天最多能跑 **100 吨** 货车（理论峰值带宽）。

- **情况一：完美装车（就像你刚才的归约代码）** 货车里装的全是实打实的货物，没有一点浪费。公路上跑了 60 吨货车，你实际收到 60 吨真货。此时 **`有效带宽 ÷ 峰值` ≈ `DRAM Throughput`**（都是 60%）。
- **情况二：DRAM Throughput 很高（80%），但 Effective Bandwidth 很低（15%）** 这说明货车里 **塞满了没用的泡沫**，公路堵得水泄不通，但你真正收到的货很少。 在 CUDA 里，这几乎一定是非合并访存（Uncoalesced Access）导致的：GPU 每次从显存搬数据有最小粒度（通常是 32 字节），如果你的线程跳着读数组，硬件被迫把一整块 32 字节全搬过来，结果你只用了其中 4 字节，剩下 28 字节全扔了。硬件总线累得半死（`DRAM Throughput` 极高），但算出来的有效带宽惨不忍睹。**GPU 每次从显存搬 32 bytes，有效地却只有 4 bytes；这样看上去 DRAM Throughput 很高，但是很多无用功。**
- **情况三：Effective Bandwidth 极高（甚至超过显存峰值），但 DRAM Throughput 很低（15%）** 这是**天大的好事**，说明你的数据在 **L1 / L2 缓存里被反复复用了**。 比如算法逻辑上要把同一块数据读 10 遍，结果第 1 遍从大仓库（DRAM）搬来后，后面 9 遍全直接在门卫室（L2 缓存）拿了。大仓库根本没怎么出货（`DRAM Throughput` 很低），但你的业务处理速度飞快（`Effective Bandwidth` 极高）。

### warp state/stall reason

```text
就是我的 SM 调度 warp，执行一条指令后，warp 进入等待。

NCU 可以统计 warp 等待地原因；这就是 warp stall reasons。
```

| Stall 类型       | 可以先怎么理解                                      |
| ---------------- | --------------------------------------------------- |
| Long Scoreboard  | 等待较长延迟的内存数据依赖                          |
| Short Scoreboard | 等待较短延迟的数据依赖，例如部分 Shared Memory 操作 |
| Barrier          | 等待线程同步                                        |
| Wait             | 等待固定延迟指令的依赖                              |
| Not Selected     | Warp 已经就绪，但 Scheduler 选择了其他 Warp         |

# reduce

## V0

```cpp
// V0: 朴素的树形规约（步长从小到大）
__global__ void reduce_v0(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局内存数据加载到共享内存
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 树形规约：步长从 1 开始逐步翻倍
    for (int step = 1; step < blockDim.x; step *= 2) {
        if (tid % (2 * step) == 0) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    // 每个 Block 的结果写回全局内存
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

```

![image-20261126235028599](cuda%20note.assets/image-20261126235028599.png)

```text
第一轮的时候，看成每两个一组。

第二轮，每 4 个一组。

注意，一组的和记在这一轮的左端点，相同编号的 thread id 上。

这样，其实我们每一轮利用到的 thread id 不连续，就会导致一个 warp 出现分化，有的工作有的不工作。

RTX4060 LAPTOP
Average time: 9.3235 ms
Effective bandwidth: 57.81 GB/s
Peak bandwidth: 256.03 GB/s
Bandwidth utilization: 22.58%
Result: 134217728
Expected: 134217728
Verification: PASSED
```

## V1

```cpp

```

![image-20261126235345844](cuda%20note.assets/image-20261126235345844.png)

```text
我们把，这一组的值，存到组编号对应的 thread 上。

比如说，第二轮，0，1，2，3 三组。
第 i 组让 thread i 去做。

这样，我们每次活跃的 thread 其实都是一段前缀，保证了 thread 编号的连续，自然降低了 warp divergence

那么，我们问题是让第 i 组的和存到 smem 的哪里？

1. 第 i 组存到 smem[i]
	那么，会有 smem[i]=smem[i*2]+smem[i*2+1]
	并且要求 RHS 都是旧值，所以需要先拍一个 snapshot 再进行赋值
2. 第 i 组存到第 i 组的左端点 smem[i*blocksize]
	那么我们直接合并就好 smem[i*blocksize]+=smem[i*blocksize+blocksize/2]
	
RTX4060 LAPTOP
Average time: 4.6823 ms
Effective bandwidth: 115.11 GB/s
Peak bandwidth: 256.03 GB/s
Bandwidth utilization: 44.96%
Result: 134217728
Expected: 134217728
Verification: PASSED
```

### V1 的一个性能卡点

```cpp
if(tid*block_size<blockDim.x)
不要写除法！
```

### code

```cpp
__global__ void block_reduce_kernel_v1_temp(float* input,float* output,int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();
    for(int block_size=2;block_size<=blockDim.x;block_size<<=1) {
        int temp;
        if(tid*block_size<blockDim.x) {
            temp=smem[tid*2]+smem[tid*2+1];
        }
        __syncthreads();
        if(tid*block_size<blockDim.x) {
            smem[tid]=temp;
        }
        __syncthreads();
    }
    if(tid==0) {
        output[blockIdx.x]=smem[0];
    }
}

__global__ void block_reduce_kernel_hw_v1_left_end_point(float* input,float* output,int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();
    for(int block_size=2;block_size<=blockDim.x;block_size<<=1) {
        if(tid*block_size<blockDim.x) {
            smem[tid*block_size]+=smem[tid*block_size+(block_size>>1)];
        }
        __syncthreads();
    }
    if(tid==0) {
        output[blockIdx.x]=smem[0];
    }
}
```

## V2 优化 shared memory 的 bank conflict

考虑对一个 block，我们一开始要求
$$
\sum_{i\in[0,step)} smem[i],step=blockDim.x
$$
如果，我们想让 step/2，需要干什么呢？

我们只要让 `smem[i]+=smem[i+step/2]`

这样我们比如初始 `step=256`，现在我们只要求 `[0,128)` 的 `smem[i]` 的和了。

同理下去，设一个 `step`，表示我们现在只要求得 `[0,step)` 的和。终点是 `step=1`。

```text
避免 bank conflict？
我们考虑一个 warp 访问的 smem
lane0: i,i+step
lane1: i+1,i+1+step

也就是 lane 连续，i 连续，那你的 bank 在 mod 32 意义下也是连续，不会有同值的情况，所以不会冲突。
```

```cpp
__global__ void block_reduce_kernel_v2_step(float* input,float* output,int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();
    for(int step=blockDim.x/2;step>=1;step>>=1) {
        if(tid<step) {
            smem[tid]+=smem[tid+step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

## V4 如果我现在需要的前缀和，已经在一个 warp 里，我直接 warp reduce

### warp reduce

```text
其实思想，跟 step 一样。
维护一个 offset，表示我现在只需要求得 lane[0,offset) 这些的值。
那么怎么从 offset=32 -> 16?
其实就是让 lane[0] 对应的 val +=lane[16]
lane[1]+=lane[17]
这样 lane[i]+=lane[i+16]。
我们用到 __shfl_down_sync，效果类似先拍一个 snapshot，带同步语义。
做完后只需要对 lane[0,16) 求和。

终点:offset=1
```

```cpp
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(
            0xFFFFFFFF, val, offset
        );
    }

    return val;
}
```

