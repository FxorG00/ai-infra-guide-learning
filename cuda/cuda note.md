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
