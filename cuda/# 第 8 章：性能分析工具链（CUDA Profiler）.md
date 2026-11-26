# 第 8 章：性能分析工具链（CUDA Profiler）

> 阅读说明：你发的这一章目前只有章节简介，没有展开具体操作教程。原文主要介绍 Nsight Systems、Nsight Compute 和 PyTorch Profiler 三个工具。因此，下面按照原文的三个工具展开，并结合 NVIDIA、PyTorch 官方文档补充实际操作方法。

这一章我们换一种学习方式。

因为你现在已经能够独立编写：

- Vector Add
- Warp Reduce
- Block Reduce
- Matrix Transpose

也理解了：

- Warp Divergence
- Coalesced Access
- Bank Conflict
- Occupancy
- Atomic Operation

但现在还有一个问题：

> 我知道这些理论上会影响性能，但我怎么知道自己的代码实际上有没有这些问题？

这就是性能分析工具存在的意义。

---

## 一、先建立整体认识：Benchmark 和 Profiler 是什么？

假设你现在写了三个版本的 Reduce：

```text
reduce_v0：朴素版本

reduce_v1：Shared Memory 优化版本

reduce_v2：Warp Shuffle 优化版本
```

你会有两个问题。

第一个问题：

> 哪个版本更快？快了多少？

第二个问题：

> 为什么这个版本更快？

这两个问题分别对应：

```text
Benchmark（性能基准测试）
    ↓
回答：有多快？

Profiler（性能分析工具）
    ↓
回答：为什么这么快，或者为什么这么慢？
```

例如你通过 Benchmark 得到：

| Kernel | 耗时 |
|---|---:|
| V0 | 2.0 ms |
| V1 | 1.0 ms |
| V2 | 0.5 ms |

现在你知道 V2 比 V0 快了 4 倍。

但是你还不知道：

```text
是不是因为 Bank Conflict 减少了？

是不是因为 Warp Divergence 减少了？

是不是因为 Shared Memory 访问减少了？

是不是因为同步次数减少了？
```

这些问题就需要 Profiler 来分析。

所以完整的性能优化流程是：

```text
写出正确的 Kernel
        ↓
Benchmark 测量性能
        ↓
Profiler 分析瓶颈
        ↓
修改 Kernel
        ↓
重新 Benchmark
        ↓
验证优化是否有效
```

以后写 CUDA 算子，基本就是不断重复这个流程。

---

## 二、这三个性能分析工具有什么区别？

原文主要介绍三个工具：

| 工具 | 命令或 API | 分析范围 |
|---|---|---|
| Nsight Systems | `nsys` | 整个 CPU/GPU 程序 |
| Nsight Compute | `ncu` | 单个 CUDA Kernel |
| PyTorch Profiler | `torch.profiler` | PyTorch 模型和算子 |

另外还有你之前已经使用过的：

```cpp
cudaEventElapsedTime()
```

它主要用于计时，不是完整的 Profiler。

这四个东西可以这样区分：

```text
整个 AI 程序
│
├── PyTorch Profiler
│   看哪些 PyTorch 算子耗时最多
│
├── Nsight Systems
│   看 CPU 和 GPU 整体执行时间线
│
└── 某个 CUDA Kernel
    │
    ├── CUDA Event
    │   测量这个 Kernel 跑了多久
    │
    └── Nsight Compute
        分析这个 Kernel 内部为什么慢
```

你目前正在学习 CUDA Reduce。

所以现阶段最重要的是：

```text
CUDA Event
+
Nsight Compute
```

另外两个先理解用途即可。

---

# 三、CUDA Event：最基本的性能测试

## 3.1 其实你已经学过 Benchmark 了

回忆你之前自己写的 Vector Add：

```cpp
cudaEvent_t start, stop;

cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

vector_add<<<grid_size, block_size>>>(
    d_A, d_B, d_C, N
);

cudaEventRecord(stop);

cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

printf("GPU Kernel time: %.4f ms\n", milliseconds);
```

这就是一个最简单的 Benchmark。

你之前已经理解了：

```text
cudaEventRecord
=
往 CUDA Stream 中放一个时间标记

cudaEventSynchronize
=
CPU 等待 GPU 执行到这个标记

cudaEventElapsedTime
=
计算两个标记之间的时间差
```

GPU 执行队列：

```text
[start]
   ↓
[vector_add kernel]
   ↓
[stop]
```

所以测到的就是这段 GPU 工作所花费的时间。

你并不需要为了学习性能测试，重新学习一套完全陌生的 API。

---

## 3.2 为什么不能直接用 CPU 计时？

假设你写：

```cpp
auto start = std::chrono::steady_clock::now();

vector_add<<<grid_size, block_size>>>(...);

auto stop = std::chrono::steady_clock::now();
```

问题是：

CUDA Kernel Launch 通常是异步的。

CPU 执行：

```cpp
vector_add<<<...>>>(...);
```

主要是向 GPU 提交任务，而不一定等 GPU 完成。

所以可能出现：

```text
CPU:
start
提交 Kernel
stop
继续执行

GPU:
      开始执行 Kernel
      ...
      Kernel 执行完成
```

CPU 测量出来的时间可能主要是提交任务的开销，而不是 GPU 执行 Kernel 的时间。

如果一定要用 CPU 时钟测量，就需要正确地同步 GPU。

但对于目前的 Kernel 测试，直接使用 CUDA Event 更方便。

---

## 3.3 为什么之前说要 Warmup？

Warmup 就是预热。

假设一个 Kernel 第一次执行：

```text
第一次：1.50 ms
第二次：0.42 ms
第三次：0.40 ms
第四次：0.41 ms
```

第一次比较慢，可能是因为：

- CUDA Context 或模块初始化。
- GPU 时钟还没有升上来。
- Cache 状态与后续执行不同。

所以正式测试之前，先运行几次：

```cpp
for (int i = 0; i < 10; ++i) {
    vector_add<<<grid_size, block_size>>>(...);
}

cudaDeviceSynchronize();
```

注意：

这里不计时。

只是让程序和 GPU 进入相对稳定的运行状态。

---

## 3.4 为什么要重复执行很多次？

因为单次测量可能存在波动。

例如：

```text
第一次：0.40 ms
第二次：0.42 ms
第三次：0.39 ms
第四次：0.41 ms
```

我们可以连续运行 100 次，再除以 100。

```cpp
int repeat = 100;

cudaEventRecord(start);

for (int i = 0; i < repeat; ++i) {
    vector_add<<<grid_size, block_size>>>(...);
}

cudaEventRecord(stop);
cudaEventSynchronize(stop);

float total_ms = 0;

cudaEventElapsedTime(&total_ms, start, stop);

float avg_ms = total_ms / repeat;
```

GPU 队列：

```text
[start]
   ↓
[Kernel 1]
[Kernel 2]
[Kernel 3]
...
[Kernel 100]
   ↓
[stop]
```

假设测得：

```text
100 次执行总耗时：40 ms
```

那么平均每次：

$$
t_{\mathrm{avg}}
=
\frac{40}{100}
=
0.4\ \mathrm{ms}
$$

注意，对于非常短的 Kernel，这种测量还可能包含连续提交任务时产生的空隙。

但现阶段拿它比较不同 Reduce 版本已经很实用了。

---

# 四、怎么通过 CUDA Event 测量带宽利用率？

这是你刚才最困惑的地方。

我们从最基本的单位开始。

## 4.1 带宽究竟是什么？

带宽表示：

> 单位时间内能够传输多少数据。

公式：

$$
\mathrm{Bandwidth}
=
\frac{\mathrm{Bytes}}{\mathrm{Time}}
$$

例如某个 Kernel 在 1 秒内处理了 100 GB 数据。

那么有效带宽就是：

$$
100\ \mathrm{GB/s}
$$

如果它只用了 0.1 秒：

$$
\frac{100}{0.1}
=
1000\ \mathrm{GB/s}
$$

就这么简单。

---

## 4.2 以 Reduce 为例

假设：

```cpp
int N = 1 << 27;
```

也就是：

$$
N=2^{27}=134217728
$$

个 float。

一个 float 是 4 Bytes。

所以输入数据量：

$$
B=N\times4
$$

得到：

$$
B=536870912\ \mathrm{Bytes}
$$

也就是：

```text
512 MiB
约 0.537 GB
```

Reduce 必须读取这些数据。

虽然还要写出结果，但最终结果的数据量相比输入通常很小。

所以我们可以暂时只计算输入读取量，用来比较不同 Reduce 版本。

---

## 4.3 如果执行时间是 0.5 ms 呢？

通过 CUDA Event 得到：

```text
Kernel time = 0.5 ms
```

注意：

ms 是毫秒，不是秒。

$$
0.5\ \mathrm{ms}
=
0.0005\ \mathrm{s}
$$

因此有效带宽：

$$
BW_{\mathrm{effective}}
=
\frac{536870912}{0.0005}
$$

得到：

$$
BW_{\mathrm{effective}}
\approx1073.74\ \mathrm{GB/s}
$$

也就是说：

这个 Kernel 以相当于每秒处理 1073.74 GB 有效输入数据的速度运行。

对应 C++ 代码：

```cpp
double bytes = (double)N * sizeof(float);

double seconds = avg_ms / 1000.0;

double bandwidth = bytes / seconds / 1e9;

printf("Bandwidth: %.2f GB/s\n", bandwidth);
```

为什么除以 1000？

```text
毫秒转换为秒
```

为什么除以 1e9？

```text
Bytes 转换为 GB
```

没有其他复杂的东西。

---

## 4.4 什么是理论峰值带宽？

假设你使用的 GPU 理论显存带宽是：

```text
2000 GB/s
```

这表示根据 GPU 的显存硬件规格，它理论上**每秒可以传输这么多数据**。

而你的 Reduce 实际有效带宽：

```text
1073.74 GB/s
```

那么相对于理论峰值的比例：

$$
\eta
=
\frac{1073.74}{2000}
\times100\%
$$

得到：

$$
\eta\approx53.69\%
$$

这就是教程里面类似：

```text
带宽利用率：53.7%
```

这种数字的一种计算方式。

对应代码：

```cpp
double peak_bandwidth = 2000.0;

double utilization =
    bandwidth / peak_bandwidth * 100.0;

printf("Bandwidth utilization: %.2f%%\n",
       utilization);
```

注意：

2000 GB/s 是这里假设的数字。

实际测试时，需要使用你自己的 GPU 对应的理论显存带宽，而不是照抄 A100 的参数。

---

## 4.5 不同 Kernel 的数据量怎么算？

你不能所有 Kernel 都直接使用：

```cpp
N * sizeof(float)
```

需要看算法读写了多少数据。

### Vector Add

```cpp
C[i] = A[i] + B[i];
```

每个元素：

```text
读取 A：4 Bytes
读取 B：4 Bytes
写入 C：4 Bytes

合计：12 Bytes
```

因此：

$$
B=12N
$$

### Matrix Transpose

```cpp
C[col * height + row] =
    A[row * width + col];
```

每个元素：

```text
读取 A：4 Bytes
写入 C：4 Bytes

合计：8 Bytes
```

因此：

$$
B=8NM
$$

这里假设输入矩阵形状为 N × M。

### Reduce

```text
读取 N 个 float

写出归约结果
```

如果只计算输入数据量：

$$
B\approx4N
$$

如果是单阶段 Block Reduce，每个 Block 写出一个 partial sum，那么更完整的有效数据量是：

$$
B=4N+4G
$$

其中 G 是 Block 数量。

总结：

| 算子 | FP32 有效数据量 |
|---|---:|
| Vector Add | 12N Bytes |
| Matrix Transpose | 8NM Bytes |
| Reduce（忽略输出） | 4N Bytes |
| Block Reduce | 4N + 4G Bytes |

这里计算的是算法有效读写量，不是硬件实际发生的全部内存流量。

---

## 4.6 为什么不使用很小的数据进行带宽测试？

你之前 Vector Add 使用：

```cpp
int N = 1 << 20;
```

一个数组只有 4 MiB。

而现代 GPU 的 L2 Cache 可能有几十 MB，甚至更多。

如果你重复执行 Kernel：

```text
第一次：
从显存读取数据
    ↓
部分数据进入 L2 Cache

第二次：
直接从 L2 Cache 读取数据
```

那么你测出来的有效带宽可能反映的是：

```text
L2 Cache + 显存的综合性能
```

而不是单纯的显存带宽。

甚至可能算出：

```text
带宽利用率 > 100%
```

这并不一定意味着 GPU 突破了显存物理极限。

可能只是大量数据根本没有经过显存。

所以如果目标是测试 DRAM / HBM 带宽：

> 输入数据规模应该明显大于 GPU 的 L2 Cache，并且还要考虑显存容量。

例如：

| FP32 元素数量 | 数据大小 |
|---|---:|
| 2^20 | 4 MiB |
| 2^24 | 64 MiB |
| 2^26 | 256 MiB |
| 2^27 | 512 MiB |

---

# 五、动手实验：写一个完整的 Reduce Benchmark

下面用你已经理解的 Warp Reduce 和 Block Reduce。

不是引入新的 Reduce 优化技术。

只是给原来的 Kernel 加上性能测试代码。

## 5.1 完整代码

文件名：

```text
reduce.cu
```

代码：

```cpp
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",     \
                    __FILE__, __LINE__,                       \
                    cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                              \
        }                                                     \
    } while (0)

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(
            0xFFFFFFFF, val, offset
        );
    }

    return val;
}

__global__ void block_reduce_kernel(
    const float* input,
    float* output,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (tid < N) ? input[tid] : 0.0f;

    // Warp Reduce
    val = warp_reduce_sum(val);

    __shared__ float warp_sums[32];

    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    if (laneId == 0) {
        warp_sums[warpId] = val;
    }

    __syncthreads();

    int numWarps = blockDim.x / 32;

    // 第一个 Warp 汇总所有 Warp 的结果
    if (warpId == 0) {
        val = (laneId < numWarps)
            ? warp_sums[laneId]
            : 0.0f;

        val = warp_reduce_sum(val);

        if (laneId == 0) {
            output[blockIdx.x] = val;
        }
    }
}

int main() {
    int N = 1 << 26;

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    size_t input_bytes = (size_t)N * sizeof(float);
    size_t output_bytes =
        (size_t)grid_size * sizeof(float);

    // 所有输入初始化为 1，方便验证
    std::vector<float> h_input(N, 1.0f);
    std::vector<float> h_output(grid_size);

    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        input_bytes,
        cudaMemcpyHostToDevice
    ));

    // 预热
    int warmup = 5;

    for (int i = 0; i < warmup; ++i) {
        block_reduce_kernel<<<grid_size, block_size>>>(
            d_input, d_output, N
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 创建计时 Event
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 正式测试
    int repeat = 20;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        block_reduce_kernel<<<grid_size, block_size>>>(
            d_input, d_output, N
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms, start, stop
    ));

    double avg_ms = total_ms / repeat;

    // 计算有效带宽
    double bytes =
        (double)input_bytes + output_bytes;

    double bandwidth =
        bytes / (avg_ms / 1000.0) / 1e9;

    printf("Average time: %.4f ms\n", avg_ms);

    printf("Effective bandwidth: %.2f GB/s\n",
           bandwidth);

    // 验证正确性
    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        output_bytes,
        cudaMemcpyDeviceToHost
    ));

    double sum = 0;

    for (float x : h_output) {
        sum += x;
    }

    printf("Result: %.0f\n", sum);
    printf("Expected: %d\n", N);

    printf("Verification: %s\n",
           sum == N ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
```

这份代码需要有足够的 GPU 显存和 Host 内存。

如果显存不足，可以先减小 N。

另外，这份代码测试的是 Block Reduce 阶段，而不是完整的 GPU 全局 Reduce，因为最后所有 Block 的 partial sum 是在 CPU 上汇总的。

我们暂时这样做，是为了单独分析这个 Kernel。

---

## 5.2 怎么运行这个程序？

假设你有一台配置好 CUDA 的 Linux 机器。

打开终端。

先检查：

```bash
nvidia-smi
```

查看 GPU 型号、显存大小等信息。

然后：

```bash
nvcc --version
```

确认 CUDA 编译器存在。

编译：

```bash
nvcc -O3 -lineinfo reduce.cu -o reduce
```

这里：

| 参数 | 含义 |
|---|---|
| `nvcc` | CUDA 编译器 |
| `-O3` | 启用编译优化 |
| `-lineinfo` | 保留源码行号信息，方便后续性能分析 |
| `reduce.cu` | 源代码文件 |
| `-o reduce` | 生成名为 reduce 的可执行文件 |

你已经学过 C++。

可以把这个理解成类似：

```bash
g++ main.cpp -o main
```

只不过现在编译 CUDA 程序需要使用 nvcc。

然后运行：

```bash
./reduce
```

假设输出：

```text
Average time: 0.4500 ms
Effective bandwidth: 598.85 GB/s
Result: 67108864
Expected: 67108864
Verification: PASSED
```

这里的时间和带宽只是示例，不是我实际在 GPU 上运行得到的结果。

你自己运行时会根据 GPU 型号得到不同数字。

到这里，你就已经完成了第一次正式的 CUDA Benchmark。

---

# 六、Nsight Compute：真正分析单个 Kernel

## 6.1 Nsight Compute 到底是什么？

Nsight Compute 是 NVIDIA 提供的 GPU Kernel 性能分析工具。

简称：

```text
NCU
```

终端命令：

```bash
ncu
```

它可以收集 GPU 的硬件性能计数器。

你可以把性能计数器理解成：

> GPU 硬件内部记录各种事件发生情况的统计装置。

例如：

```text
执行了多少指令？

发出了多少内存请求？

访问了多少 DRAM 数据？

Warp 有多少时间在等待？

Shared Memory 出现了多少 Bank Conflict？

一个 Thread 使用了多少 Register？

SM 上驻留了多少 Warp？
```

你之前学的这些概念，现在都可以通过工具观察。

CUDA Event 告诉你：

```text
这个 Kernel 花了 0.5 ms
```

NCU 则可以告诉你：

```text
这个 Kernel 的内存吞吐量怎么样？

计算单元忙不忙？

Occupancy 是否受 Register 限制？

有没有大量 Bank Conflict？
```

---

## 6.2 NCU 是写在 CUDA 代码里面的吗？

不是。

这是非常重要的一点。

你不需要在 Kernel 中加入：

```cpp
ncu(...);
```

NCU 是一个独立的程序。

你的工作流程：

```text
reduce.cu
    ↓ nvcc 编译
reduce 可执行文件
    ↓ ncu 启动并分析
性能报告
```

你原来运行程序：

```bash
./reduce
```

现在把它交给 NCU：

```bash
ncu ./reduce
```

就像使用调试器运行一个 C++ 程序。

例如：

```bash
gdb ./main
```

区别在于：

```text
GDB 主要用于调试程序正确性。

NCU 主要用于分析 GPU Kernel 性能。
```

---

## 6.3 怎么知道自己有没有安装 NCU？

终端执行：

```bash
ncu --version
```

如果能够输出版本信息，说明当前环境能够找到这个工具。

如果：

```text
command not found
```

可能是：

```text
没有安装 Nsight Compute

或者

安装了，但是没有加入 PATH
```

如果执行时出现：

```text
ERR_NVGPUCTRPERM
```

说明当前用户可能没有访问 GPU 性能计数器的权限。

这种情况需要检查机器的 NVIDIA 驱动配置，或者询问云 GPU 平台是否允许 Profiling。

具体安装和配置方式取决于你的 CUDA 环境。

---

## 6.4 第一次使用 NCU

我们刚才写的 reduce 程序已经编译完成。

可以执行：

```bash
ncu \
    --launch-count 1 \
    ./reduce
```

其中：

```bash
--launch-count 1
```

表示只收集一次符合条件的 Kernel Launch。

为什么需要这个参数？

因为我们的程序里有：

```text
5 次 Warmup

20 次正式测试
```

如果每次都详细收集性能指标，可能非常耗时。

我们目前只需要分析其中一次具有代表性的 Kernel。

---

## 6.5 怎么指定要分析的 Kernel？

假设一个 CUDA 程序里面有：

```cpp
init_kernel<<<...>>>(...);

reduce_v0<<<...>>>(...);

reduce_v1<<<...>>>(...);

reduce_v2<<<...>>>(...);
```

但你只想分析：

```text
reduce_v1
```

可以使用：

```bash
ncu \
    --kernel-name regex:reduce_v1 \
    --launch-count 1 \
    ./reduce
```

其中：

```bash
--kernel-name regex:reduce_v1
```

表示匹配名称包含 reduce_v1 的 Kernel。

这就是：

```text
只看自己感兴趣的 Kernel
```

---

## 6.6 怎么跳过 Warmup？

我们前面的代码：

```cpp
int warmup = 5;
```

所以可以使用：

```bash
ncu \
    --kernel-name regex:block_reduce_kernel \
    --launch-skip 5 \
    --launch-count 1 \
    ./reduce
```

逐行解释：

```text
ncu
    启动性能分析工具

--kernel-name regex:block_reduce_kernel
    指定要分析的 Kernel

--launch-skip 5
    跳过前 5 次匹配的 Kernel Launch

--launch-count 1
    只分析后面 1 次 Kernel Launch

./reduce
    要运行的可执行文件
```

这个就是你刚才看到的命令的实际含义。

不用背。

你之后每写一个 Kernel，都可以直接套用这个格式。

---

# 七、NCU 最重要的面板：Speed of Light（SOL）

原文特别强调：

```text
Speed of Light
```

简称：

```text
SOL
```

这个名字虽然有点奇怪，但实际上很好理解。

GPU 有理论上的性能上限。

例如：

```text
最大计算吞吐量

最大内存吞吐量
```

SOL 主要帮助你观察：

> 当前 Kernel 对这些硬件资源利用到了什么程度？

你现阶段可以先看：

```text
Memory Throughput

Compute Throughput

DRAM Throughput

Duration
```

使用命令：

```bash
ncu \
    --section SpeedOfLight \
    --kernel-name regex:block_reduce_kernel \
    --launch-skip 5 \
    --launch-count 1 \
    ./reduce
```

其中：

```bash
--section SpeedOfLight
```

表示收集 SOL 这一部分指标。

---

## 7.1 假设你看到下面的报告

注意：以下数字全部是为了讲解而构造的示例。

```text
GPU Speed Of Light Throughput
---------------------------------
Duration               0.450 ms
Memory Throughput      85.00 %
Compute Throughput      8.00 %
DRAM Throughput        80.00 %
```

这是什么意思？

```text
Memory Throughput = 85%

说明某些内存子系统的吞吐量
已经比较接近其峰值能力。
```

```text
Compute Throughput = 8%

说明整体计算吞吐量指标较低。
```

```text
DRAM Throughput = 80%

说明显存传输吞吐量
已经达到较高水平。
```

这时候我们可以初步判断：

> 这个 Kernel 很可能主要受内存系统限制，而不是计算吞吐量限制。

这就是你之前学的：

```text
Memory-Bound
```

---

## 7.2 如果反过来呢？

假设：

```text
Memory Throughput = 30%

Compute Throughput = 90%
```

那么很可能：

```text
数据搬运不是主要瓶颈

计算单元已经比较繁忙
```

这是：

```text
Compute-Bound
```

的一种典型表现。

例如高性能 GEMM。

但注意，这些只是诊断线索，不是绝对的分类公式。

---

## 7.3 如果两个都很低呢？

例如：

```text
Memory Throughput = 15%

Compute Throughput = 10%
```

这时候不能简单说：

```text
这个 Kernel 不存在瓶颈
```

恰恰相反，它可能存在：

- Occupancy 太低。
- Warp 大量等待内存延迟。
- 指令之间依赖太强。
- 同步次数太多。
- 数据规模太小。
- Warp Divergence 严重。
- 并行度不足。

也就是说：

```text
带宽没跑满

算力也没跑满

但就是跑不快
```

这时候就要继续查看其他性能指标。

---

# 八、自己计算的带宽和 NCU 的带宽有什么区别？

这是前面回答中比较容易让人混淆的地方。

你通过 CUDA Event 算的是：

```cpp
bandwidth = bytes / seconds;
```

这里的 bytes 是你根据算法推导出来的。

比如：

```text
Reduce 应该读取 N 个 float。
```

所以：

$$
B_{\mathrm{effective}}=4N
$$

这叫：

```text
Effective Bandwidth
有效带宽
```

但是 GPU 实际访问显存时，可能因为：

```text
非合并访问

地址未对齐

重复读取

Cache Miss

额外的内存请求
```

导致硬件实际传输的数据量更多。

也可能因为 Cache 命中，导致实际 DRAM 流量更少。

NCU 可以通过硬件计数器观察实际 DRAM 读写情况。

所以两者不一定一样。

---

## 8.1 用一个例子彻底区分

假设 Warp 需要读取 32 个 float。

有效数据量：

$$
32\times4=128\ \mathrm{Bytes}
$$

如果连续且对齐访问：

```text
实际需要读取的 Sector 数据
可能正好覆盖这 128 Bytes。
```

如果每个 Lane 都访问相隔很远的地址：

```text
每个 Lane 只需要 4 Bytes

却可能各自请求不同的 32B Sector
```

那么实际请求的数据量可能达到：

$$
32\times32=1024\ \mathrm{Bytes}
$$

这是以 32B Sector 为粒度的简化例子；是否真的形成同等大小的 DRAM 流量，还取决于缓存等因素。

于是：

```text
算法有效数据：128 Bytes

硬件请求数据：1024 Bytes
```

有效数据只占：

$$
\frac{128}{1024}=12.5\%
$$

所以完全可能出现：

```text
GPU 内存系统很忙

但是有效带宽很低
```

因为搬运了大量算法真正不需要的数据。

这就是你在 2.2 学 Coalescing 时的性能问题。

---

## 8.2 三个概念不要混淆

| 指标 | 含义 |
|---|---|
| Effective Bandwidth | 算法有效数据量 / 时间 |
| Effective BW / Peak BW | 有效带宽相对于理论显存峰值的比例 |
| NCU DRAM Throughput | 硬件 DRAM 吞吐量相对于工具定义峰值的比例 |

其中 NCU 使用的 peak sustained 口径不一定与产品规格表里的理论峰值完全一致。

所以不要要求这几个数字完全相等。

你以后比较 Reduce 的 V0～V7 时，可以先统一使用：

```text
CUDA Event 耗时

有效带宽

有效带宽 / GPU 理论峰值
```

再使用 NCU 解释性能差异。

---

# 九、NCU 里面其他重要指标

这一部分终于开始和你学过的 CUDA 基础连接起来了。

## 9.1 Occupancy

在 2.3 学过：

$$
\mathrm{Occupancy}
=
\frac{\mathrm{Active\ Warps}}
{\mathrm{Maximum\ Warps}}
$$

以前我们只能手算。

现在 NCU 可以提供：

```text
Theoretical Occupancy

Achieved Occupancy
```

Theoretical Occupancy：

```text
根据 Kernel 的资源需求，
理论上 SM 能同时驻留多少 Warp。
```

Achieved Occupancy：

```text
实际执行期间，
SM 平均驻留的 Warp 情况。
```

例如：

```text
Theoretical Occupancy = 75%

Achieved Occupancy = 60%
```

说明实际运行时没有完全达到理论驻留水平。

可能与 Grid 大小、执行尾部、不同 Block 的完成时间等有关。

但是：

> Occupancy 低不代表 Kernel 一定慢。

这就是你在 2.3 已经学过的 Trade-off。

---

## 9.2 Registers Per Thread

你之前学过：

```text
Register 用量越高

一个 SM 可能容纳的 Warp 越少
```

现在可以实际查看：

```text
Registers Per Thread = 64
```

这样就知道：

当前 Kernel 每个 Thread 使用了多少个 Register。

还可以在编译时查看：

```bash
nvcc -O3 -Xptxas -v reduce.cu -o reduce
```

编译器可能输出类似：

```text
Used 32 registers
```

这不是 Nsight Compute。

而是 CUDA 编译器提供的资源使用信息。

---

## 9.3 Shared Memory Per Block

例如你写：

```cpp
__shared__ float tile[32][33];
```

实际每个 Block 的这块数组大小：

$$
32\times33\times4=4224\ \mathrm{Bytes}
$$

NCU 可以提供 Kernel 的 Shared Memory 使用情况。

结合 Register 和 Block Size：

就可以分析到底是什么限制了 Occupancy。

---

## 9.4 Shared Memory Bank Conflicts

你之前自己写过 Matrix Transpose：

```cpp
__shared__ float tile[32][33];
```

你知道：

```cpp
__shared__ float tile[32][32];
```

按列读取时会产生 Bank Conflict。

那么现在可以做一个特别有意义的实验。

版本 A：

```cpp
__shared__ float tile[32][32];
```

版本 B：

```cpp
__shared__ float tile[32][33];
```

其他代码保持一致。

先测量：

```text
两个版本的 Kernel Time
```

然后使用 NCU 查看：

```text
Shared Memory Bank Conflicts
```

你预期会看到：

```text
版本 A：
Shared Memory 列读取存在严重 Bank Conflict

版本 B：
对应的 Bank Conflict 大幅减少
```

至于具体提升多少，需要实际测量。

这就是 Profiler 最有价值的地方：

> 让你之前学的理论，变成能够被实验验证的事实。

---

## 9.5 Warp State / Stall Reasons

在 2.1 学过：

```text
Warp Scheduler

Latency Hiding
```

Warp 可能处于：

```text
可以执行

等待内存

等待指令依赖

等待同步
```

NCU 可以统计 Warp 无法继续发射指令的原因。

这就是：

```text
Warp Stall Reasons
```

常见的几种情况：

| Stall 类型 | 可以先怎么理解 |
|---|---|
| Long Scoreboard | 等待较长延迟的内存数据依赖 |
| Short Scoreboard | 等待较短延迟的数据依赖，例如部分 Shared Memory 操作 |
| Barrier | 等待线程同步 |
| Wait | 等待固定延迟指令的依赖 |
| Not Selected | Warp 已经就绪，但 Scheduler 选择了其他 Warp |

例如：

```text
Barrier Stall 很高
```

那么你可以检查：

```cpp
__syncthreads();
```

是不是使用太多，或者不同 Warp 到达同步点的时间差太大。

例如：

```text
Long Scoreboard 很高
```

那么可以考虑：

```text
Global Memory Access

Cache Hit Rate

Occupancy

Memory-Level Parallelism
```

但是注意：

Stall 指标高不一定需要优化。

如果其他 Warp 一直能够填满执行资源，某个 Warp 的等待可能已经被成功隐藏了。

---

## 9.6 怎么收集更多指标？

最简单的方式：

```bash
ncu \
    --set full \
    --kernel-name regex:block_reduce_kernel \
    --launch-skip 5 \
    --launch-count 1 \
    ./reduce
```

其中：

```bash
--set full
```

表示收集较全面的性能分析信息。

如果只想收集某些部分：

```bash
ncu --list-sections
```

可以查看当前版本支持的 Section。

例如：

```text
SpeedOfLight

Occupancy

MemoryWorkloadAnalysis

WarpStateStats

SchedulerStats

LaunchStats
```

然后指定：

```bash
ncu \
    --section SpeedOfLight \
    --section Occupancy \
    --section MemoryWorkloadAnalysis \
    --launch-count 1 \
    ./reduce
```

现阶段不用死记所有 Metric 名称。

需要分析什么问题，再去找对应的 Section。

---

# 十、NCU 如何生成可视化报告？

前面我们使用的是命令行输出。

但如果指标很多，终端里查看比较麻烦。

可以让 NCU 将报告保存下来。

```bash
ncu \
    --set full \
    --kernel-name regex:block_reduce_kernel \
    --launch-skip 5 \
    --launch-count 1 \
    -o reduce_report \
    ./reduce
```

其中：

```bash
-o reduce_report
```

表示输出报告文件。

通常会生成：

```text
reduce_report.ncu-rep
```

然后使用 Nsight Compute 的 GUI 打开。

整个流程：

```text
远程 Linux GPU 服务器
        ↓
运行 ncu
        ↓
生成 .ncu-rep
        ↓
下载报告到自己的电脑
        ↓
通过 Nsight Compute GUI 查看
```

本地查看报告时不需要重新运行原来的 Kernel。

这对于租用云 GPU 或使用远程 GPU 服务器非常方便。

---

## 10.1 为什么不能直接使用 NCU 下的 CUDA Event 耗时？

因为 NCU 为了收集硬件性能计数器，可能：

```text
暂停程序

多次重放 Kernel

保存和恢复数据

控制 Cache 状态

收集不同类型的硬件计数器
```

这些操作会改变正常运行时的环境和开销。

所以应该分成两次测试。

第一次：

```bash
./reduce
```

获取正常 Benchmark 耗时。

第二次：

```bash
ncu --set full --launch-count 1 ./reduce
```

获取性能诊断信息。

不要把 NCU 运行期间你自己打印出来的 CUDA Event 时间直接当成正常性能结果。

记住：

```text
正常运行：测性能

Profiler 运行：分析瓶颈
```

---

# 十一、Nsight Systems：从整个程序的角度分析性能

现在进入原文介绍的另一个工具。

```text
Nsight Systems
```

简称：

```text
NSYS
```

命令：

```bash
nsys
```

它和 NCU 最大的区别是：

```text
NCU：观察某一个 Kernel 内部。

NSYS：观察整个程序的执行时间线。
```

可以把它们理解成：

```text
NSYS = 广角镜头

NCU = 显微镜
```

---

## 11.1 为什么只分析单个 Kernel 还不够？

假设你有一个 Transformer 推理程序。

GPU 需要执行：

```text
RMSNorm

QKV Projection

Attention

Output Projection

FFN

...
```

假设每个 Kernel 都已经优化得很快。

但是整个程序仍然很慢。

为什么？

可能 GPU 大量时间根本没有工作。

例如：

```text
CPU:
提交 A     处理 Python 逻辑     提交 B

GPU:
[Kernel A]------空闲------------[Kernel B]
```

虽然：

```text
Kernel A 很快

Kernel B 也很快
```

但是中间 GPU 一直在等 CPU 提交任务。

这段时间就叫：

```text
GPU Idle Gap
```

这时候你再优化 Kernel A 内部的 Bank Conflict，可能根本解决不了主要问题。

你应该优化的是：

```text
CPU 调度

Python 开销

Kernel Launch Overhead

CPU/GPU 同步
```

这就是 Nsight Systems 擅长分析的东西。

---

## 11.2 NSYS 能看到什么？

一个简化的时间线：

```text
时间 -------------------------------------->

CPU Thread:
[launch A][CPU work][launch B][launch C]

CUDA API:
[Launch]            [Launch] [Launch]

GPU Stream:
     [Kernel A]     [Kernel B][Kernel C]
               空闲
```

你可以观察：

- CPU 什么时候提交 Kernel。
- GPU 什么时候真正执行。
- 哪些 Kernel 耗时最长。
- Kernel 之间有没有空隙。
- Host-to-Device / Device-to-Host Copy。
- 不同 CUDA Stream 是否发生重叠。
- 是否存在不必要的同步。

以后学多 GPU 分布式训练，还可以分析部分通信操作与计算的重叠情况。

---

## 11.3 怎么运行 Nsight Systems？

检查安装：

```bash
nsys --version
```

假设你已经编译出：

```text
reduce
```

执行：

```bash
nsys profile -o reduce_timeline ./reduce
```

逐行理解：

```text
nsys
    Nsight Systems 工具

profile
    进行性能分析

-o reduce_timeline
    指定输出报告名称

./reduce
    要运行的程序
```

执行完成后通常会生成：

```text
reduce_timeline.nsys-rep
```

可以用 Nsight Systems GUI 打开，查看整个时间线。

如果只想在终端查看 Kernel 的统计信息：

```bash
nsys stats \
    --report cuda_gpu_kern_sum \
    reduce_timeline.nsys-rep
```

这样可以看到不同 Kernel 的执行次数、累计耗时等信息。

---

## 11.4 NSYS 和 NCU 的报告文件区别

```text
Nsight Systems
    ↓
.nsys-rep
    ↓
整个程序的时间线


Nsight Compute
    ↓
.ncu-rep
    ↓
单个 Kernel 的性能指标
```

不要搞混。

---

## 11.5 Kernel Launch Overhead 是什么？

每次执行：

```cpp
kernel<<<grid, block>>>(...);
```

都有一定的提交和启动成本。

对于一个很大的 GEMM：

```text
Kernel 执行时间很长
```

启动开销可能占比很小。

但是对于很多极小的 Kernel：

```text
Kernel 本身只执行几微秒
```

CPU 提交任务和 GPU 启动 Kernel 的成本，就可能非常明显。

例如：

```text
Kernel A
    ↓
Kernel B
    ↓
Kernel C
    ↓
Kernel D
```

如果这些操作可以融合：

```text
Fused Kernel ABCD
```

就可能：

```text
减少 Kernel Launch 次数

减少中间数据写回 Global Memory

减少重新读取中间数据
```

这就是后面学习：

```text
Operator Fusion

CUDA Graph

torch.compile
```

时非常重要的性能背景。

---

## 11.6 为什么 NSYS 对 AI Infra 特别重要？

以后你分析 LLM 推理引擎。

可能发现：

```text
Attention Kernel 本身很快

GEMM 本身也很快

但是 GPU 仍然大量空闲
```

这时候真正的问题可能是：

```text
Scheduler 调度太慢

CPU 开销太大

Batch Size 太小

Kernel Launch 太频繁

通信和计算没有重叠
```

这些都不是单独分析某个 Kernel 就能完全解决的。

所以实际 AI Infra 工程中，经常先使用：

```text
Nsight Systems
```

找到整个系统慢在哪里。

再使用：

```text
Nsight Compute
```

深入分析具体 Kernel。

---

# 十二、PyTorch Profiler

这是原文介绍的第三个工具。

它主要针对 PyTorch 程序。

你之前已经学过：

```python
import torch

x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")

z = x @ y
```

但是 PyTorch 底层实际上会调用 CUDA Kernel。

所以你可能想知道：

```text
哪个 PyTorch 算子最耗时？

MatMul？

Softmax？

LayerNorm？

Attention？

Memory Copy？
```

这时候可以使用：

```python
torch.profiler
```

---

## 12.1 最简单的使用方法

```python
import torch

from torch.profiler import (
    profile,
    ProfilerActivity
)

x = torch.randn(
    2048, 2048,
    device="cuda"
)

y = torch.randn(
    2048, 2048,
    device="cuda"
)

# Warmup
for _ in range(5):
    z = x @ y

torch.cuda.synchronize()

# 开始性能分析
with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA
    ]
) as prof:

    z = x @ y

    z = torch.relu(z)

    z = torch.softmax(z, dim=-1)

    torch.cuda.synchronize()

# 输出性能统计表
print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=10
    )
)
```

这里主要做了三个操作：

```python
z = x @ y
```

矩阵乘法。

```python
z = torch.relu(z)
```

激活函数。

```python
z = torch.softmax(z, dim=-1)
```

Softmax。

Profiler 会记录这个范围内的 CPU 和 CUDA 活动。

---

## 12.2 输出结果怎么看？

假设得到：

| Operator | Self CUDA Time |
|---|---:|
| aten::mm | 350 us |
| aten::_softmax | 100 us |
| aten::relu | 20 us |

这里也是简化的示例结果。

你不需要知道所有 PyTorch 内部函数名。

只需要知道：

```text
aten::mm
    矩阵乘相关算子

aten::_softmax
    Softmax 相关算子

aten::relu
    ReLU 相关算子
```

这样你就可以判断：

```text
目前最耗时的是 MatMul。

其次是 Softmax。

ReLU 占比很小。
```

这时候如果你花很多时间去优化 ReLU，整体收益可能非常有限。

---

## 12.3 Self Time 和 Total Time 有什么区别？

假设：

```text
Operator A
    ├── Operator B
    └── Operator C
```

那么：

```text
Self Time
```

表示这个 Operator 自己的归属耗时，不包含嵌套子操作的对应时间。

而：

```text
Total Time
```

通常包含它及其子操作的归属耗时。

实际看 CUDA 时间时，还要考虑异步执行和算子关联关系，不能把所有时间简单相加当成整个程序的 wall-clock time。

现阶段先使用：

```python
sort_by="self_cuda_time_total"
```

观察哪些算子自身关联的 CUDA 耗时较大。

---

## 12.4 怎么导出时间线？

可以使用：

```python
prof.export_chrome_trace("trace.json")
```

然后用兼容 Chrome Trace 格式的工具，例如 Perfetto，打开：

https://ui.perfetto.dev/

这样就可以看到：

```text
CPU Operator

CUDA Runtime API

GPU Kernel

执行时间线
```

原文还提到了 TensorBoard 集成。

TensorBoard 对应的 Profiler 插件需要额外环境配置，而且存在版本兼容问题。

现阶段你知道可以导出 Trace 就够了。

---

## 12.5 为什么 PyTorch Profiler 和 torch.compile 有关系？

假设原来代码：

```python
def f(x, y):
    z = x + y
    z = torch.relu(z)
    z = z * 2
    return z
```

原来的执行方式可能产生多个 Kernel：

```text
Add Kernel

ReLU Kernel

Multiply Kernel
```

使用：

```python
compiled_f = torch.compile(f)
```

之后，编译器有机会将它们融合。

例如：

```text
一个 Fused Kernel
```

这时候就可以使用 PyTorch Profiler 比较：

```text
torch.compile 前：

Kernel 数量

CUDA Time

CPU Overhead


torch.compile 后：

Kernel 数量

CUDA Time

CPU Overhead
```

注意，第一次调用 compiled_f 可能包含编译开销。

比较稳定运行性能时，需要先完成编译和预热。

这就是性能分析工具与后面 AI Compiler 的联系。

---

# 十三、三个工具如何配合使用？

假设你现在有一个 LLM 推理程序。

发现：

```text
推理速度太慢
```

第一步：

使用 PyTorch Profiler。

发现：

```text
Attention 占用了很多 CUDA 时间
```

第二步：

使用 Nsight Systems。

观察：

```text
GPU 是否一直在执行？

Attention 前后有没有 Idle Gap？

CPU 是否拖慢了 GPU？

数据复制是否太频繁？
```

第三步：

使用 Nsight Compute。

针对某个 Attention Kernel 分析：

```text
Memory Throughput

Compute Throughput

Occupancy

Bank Conflict

Warp Stall
```

第四步：

修改 CUDA Kernel 或上层调度逻辑。

第五步：

重新 Benchmark。

验证真实收益。

完整过程：

```text
PyTorch Profiler
        ↓
找到耗时算子
        ↓
Nsight Systems
        ↓
分析系统时间线
        ↓
Nsight Compute
        ↓
分析具体 Kernel 内部瓶颈
        ↓
修改代码
        ↓
Benchmark 验证
```

这不是强制固定的顺序。

如果你已经明确知道问题就在某个 CUDA Kernel 内部，可以直接使用 NCU。

---

# 十四、你现在怎么实际学习这一章？

结合你目前的进度，我不建议你现在就把这三个工具全部深入学习。

你正在学 Reduce。

最适合你的路线是：

## 第一阶段：只学 CUDA Event Benchmark

做到：

```text
会计时

会预热

会重复执行

会算平均时间

会算有效带宽

会算相对峰值带宽

会算 Speedup
```

其中 Speedup 的定义：

$$
\mathrm{Speedup}
=
\frac{t_{\mathrm{old}}}{t_{\mathrm{new}}}
$$

例如：

```text
V0 = 2 ms

V1 = 0.5 ms
```

那么：

$$
\mathrm{Speedup}
=
\frac{2}{0.5}
=
4
$$

也就是 V1 是 V0 的 4 倍速度。

---

## 第二阶段：学 NCU 的基本使用

只需要会这几条命令：

```bash
nvidia-smi
```

查看 GPU。

```bash
nvcc -O3 -lineinfo reduce.cu -o reduce
```

编译。

```bash
./reduce
```

测量正常性能。

```bash
ncu --launch-count 1 ./reduce
```

收集基本性能指标。

```bash
ncu --set full --launch-count 1 -o report ./reduce
```

生成完整性能报告。

这几条命令足够你开始实践。

---

## 第三阶段：做你已经写过的 Matrix Transpose 实验

我非常推荐你先做这个。

因为你之前已经独立写出了：

```cpp
__shared__ float tile[32][33];
```

并且真正理解了：

```text
为什么读取 Global Memory 是连续的

为什么写入 Global Memory 是连续的

为什么 Shared Memory 按列访问有 Bank Conflict

为什么 Padding +1 可以解决
```

现在可以实际测试三个版本。

版本 A：Naive Transpose

```text
直接读取 Global Memory

直接跨步写入 Global Memory
```

版本 B：Tiled Transpose

```cpp
__shared__ float tile[32][32];
```

版本 C：Padded Tiled Transpose

```cpp
__shared__ float tile[32][33];
```

其他条件尽量一致。

比较：

| 版本 | Kernel Time | Effective BW | Bank Conflict |
|---|---|---|---|
| Naive | 待测 | 待测 | 不适用 |
| Tiled | 待测 | 待测 | 待测 |
| Padded | 待测 | 待测 | 待测 |

这个实验可以让你第一次真正感受到：

```text
以前学到的 CUDA 硬件理论

↓

自己写出来的 CUDA Kernel

↓

Profiler 给出的真实性能数据
```

这三者是怎么连接的。

---

## 第四阶段：继续学习 Reduce V0～V7

学习每一个版本时，维护这样的表：

| 版本 | 优化方法 | 耗时 | 有效带宽 | 相对 V0 加速比 |
|---|---|---|---|---|
| V0 | Naive | 待测 | 待测 | 1x |
| V1 | 交错寻址优化 | 待测 | 待测 | 待测 |
| V2 | Sequential Addressing | 待测 | 待测 | 待测 |
| V3 | 双元素处理 | 待测 | 待测 | 待测 |
| V4 | Warp 阶段优化 | 待测 | 待测 | 待测 |
| V5 | Loop Unrolling | 待测 | 待测 | 待测 |
| V6 | Warp Shuffle | 待测 | 待测 | 待测 |
| V7 | Vectorized Load | 待测 | 待测 | 待测 |

注意所有版本应该统一：

```text
GPU 型号

输入数据规模

数据类型

计时范围

结果正确性要求
```

否则性能对比没有太大意义。

尤其要区分：

```text
仅生成 Block Partial Sum 的耗时

完整 Reduce 得到一个最终结果的耗时
```

这两种测试口径不能混在一起。

另外，老式 Warp Reduce 中依赖隐式 Warp 同步的实现，在现代 GPU 上需要重新检查正确性。

不要为了复现教程的版本而保留不安全的同步写法。

---

# 十五、如果我现在只使用 LeetGPU 怎么办？

如果你只在 LeetGPU 上提交 Kernel，那么可以先利用平台提供的运行时间进行比较。

例如：

```text
V0 Runtime = 1.2 ms

V1 Runtime = 0.8 ms

V2 Runtime = 0.4 ms
```

只要测试 GPU、数据规模和测量口径一致，就可以比较性能。

如果知道 GPU 理论带宽，还可以自己计算有效带宽利用率。

但是：

```text
ncu

nsys

nvcc -Xptxas -v
```

这些工具通常需要能够操作 CUDA 开发环境。

也就是说，你需要有：

```text
NVIDIA GPU

CUDA Toolkit

终端访问权限

GPU 性能计数器权限
```

不一定必须自己买显卡。

也可以使用允许 Profiling 的远程 GPU 服务器或云 GPU。

注意在租用之前确认平台是否支持 NCU 硬件性能计数器，否则可能只能运行 CUDA 代码，却不能完整分析性能。

---

# 十六、整个性能优化工具链的思维模型

以后你写一个 Kernel，先不要上来就想：

```text
我要提高 Occupancy

我要减少 Register

我要使用 float4

我要使用 Warp Shuffle
```

正确流程应该是：

```text
第一步：保证正确性
        ↓
第二步：CUDA Event 测量性能
        ↓
第三步：判断是否真的需要优化
        ↓
第四步：NCU 分析性能瓶颈
        ↓
第五步：根据瓶颈选择优化方法
        ↓
第六步：重新测量
```

例如：

```text
发现 Global Memory 访问效率低
        ↓
检查 Coalescing


发现 Shared Memory Bank Conflict 高
        ↓
Padding / Swizzle


发现 Occupancy 太低且影响指令发射
        ↓
检查 Register / Shared Memory


发现 Barrier Stall 严重
        ↓
检查同步和工作量分布


发现 Global Atomic Contention 严重
        ↓
Warp Reduce / Block Reduce
```

你会发现：

第二章学的所有 CUDA 优化知识，

实际上就是在给这里提供各种解决问题的方法。

---

# 十七、这一章目前必须掌握什么？

## 必须掌握

```text
1. Benchmark 和 Profiler 的区别。

2. CUDA Event 怎么测量 Kernel 耗时。

3. 为什么需要 Warmup 和 Repeat。

4. 怎么计算有效带宽。

5. 怎么计算相对峰值带宽和 Speedup。

6. 怎么使用 nvcc 编译 CUDA 程序。

7. 怎么使用 ncu 分析单个 Kernel。

8. 怎么看 Memory Throughput 和 Compute Throughput。

9. 怎么观察 Occupancy 和 Bank Conflict。

10. 为什么不能把 Profiler 下的耗时直接当作正常 Benchmark。
```

## 暂时了解即可

```text
Nsight Systems 的完整 Timeline 分析

PyTorch Profiler 的高级配置

TensorBoard 集成

NCU 所有底层 Metric 名称

GPU Performance Counter 的具体硬件实现

复杂的多 Stream 分析

分布式通信 Profiling
```

等你后面学完 GEMM、Softmax、Attention，再回来深入这些工具。

那时候你会有更多真实的优化场景。

---

# 十八、最后总结

这章最核心的四个工具：

| 工具 | 回答的问题 |
|---|---|
| CUDA Event | Kernel 跑了多久？ |
| Nsight Compute | Kernel 内部为什么慢？ |
| Nsight Systems | 整个 CPU/GPU 程序慢在哪里？ |
| PyTorch Profiler | 哪个 PyTorch 算子最耗时？ |

而你目前只需要重点掌握前两个。

把整篇内容压缩成一句话：

> Benchmark 负责量化性能，Profiler 负责解释性能，之前学过的 CUDA 优化知识负责改善性能。

---

# 十九、接下来的学习路线

我建议你现在这样走：

```text
CUDA 第 1、2 章
        ↓
已基本完成
        ↓
第 8 章基础工具
        ↓
CUDA Event + NCU 基本使用
        ↓
Matrix Transpose 性能对比实验
        ↓
第 3 章 Reduce V0～V7
        ↓
第 4 章 GEMM
        ↓
Softmax / Attention
        ↓
回头深入完整性能分析工具链
```

不需要因为 Profiler 有很多陌生的指标，就停下整个 CUDA 学习进度。

你现阶段最重要的，是建立一个习惯：

写出一个 Kernel，不仅要验证它算得对不对，还要测量它跑得快不快，并且尝试解释为什么。

---

# 参考资料

原文（目前为章节简介）：

https://caomaolufei.github.io/AIInfraGuide/guides/模块二-cuda编程与算子优化/第8章-性能分析工具链/

NVIDIA CUDA Best Practices Guide：

https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

NVIDIA Nsight Compute CLI 官方文档：

https://docs.nvidia.com/nsight-compute/NsightComputeCli/

NVIDIA Nsight Systems 官方文档：

https://docs.nvidia.com/nsight-systems/UserGuide/

PyTorch Profiler 官方文档：

https://docs.pytorch.org/docs/stable/profiler.html

本文中的操作命令、实验代码和示例指标属于结合官方文档补充的教学内容，并非原文章节中已经提供的完整实验。代码未在实际 GPU 环境中运行验证，示例性能数字也不是实测结果。