# ct-fbp-cuda

基于 CUDA 的 CT 图像**滤波反投影（FBP, Filtered BackProjection）**重建器：
从模拟 X 光扫描到 GPU 加速重建的完整流水线，附 CPU 参考实现、迭代重建对照
与逐项性能测量。

```text
 Shepp-Logan 体模 → 前向投影(Radon) → 正弦图 → Ramp 滤波 → 反投影 → 重建图像
        │               │                        │           │
      phantom.cu   projector_*.cu            filter_*.cu  backprojector_*.cu
                   v1 / warp / atomic / cdp     cuFFT      naive / texture
                   + 多流 / 多 GPU

 分支：SIRT 迭代重建（cuBLAS Tensor Core + Thrust + 统一内存）
       CUDA Graphs：Ramp → 反投影整链捕获重放
```

- **正确性可验证**：GPU 与 CPU 逐阶段比对（重建图逐点误差 ~1e-6），RMSE/PSNR 自动输出；
  9 组测试覆盖坐标约定、定标、变体一致性、图重放一致性
- **性能可量化**：512×512×360 下反投影加速 **~580×**（RTX 5070 vs 单线程 CPU）；
  少角度 SIRT 比 FBP 的 RMSE 低 35%
- **知识点全覆盖**：朴素 kernel / 纹理内存 / warp 归约 / 原子操作 / 动态并行 /
  cuFFT / cuBLAS Tensor Core / Thrust / 统一内存 / 多流 + 锁页 / 多 GPU / CUDA Graphs，
  每条都有实测数据（见 [`ct_tour`](#cuda-知识点全景)）
- **零第三方依赖**：除 CUDA Toolkit（cuFFT / cuBLAS / Thrust 随包提供）外无外部库；
  图像 IO 为自写 PGM 读写，CPU FFT 为自写 radix-2

## 原理速览

**前向投影**（模拟扫描，Radon 变换）：对每个角度 θ，沿一束束平行射线积分图像，
得到 1D 投影 p(θ,s)。所有角度按行堆叠即**正弦图**（离中心 (x₀,y₀) 的点划出
s = x₀cosθ + y₀sinθ 的正弦轨迹）。

**Ramp 滤波（Ram-Lak）**：反投影 inherently 对高频欠加权，会得到模糊图像；对正弦图
**每行**做 1D 高通滤波（频响 |ν|）预先补偿。实现走 FFT 域，可乘 Hamming 窗抑制振铃。

**反投影**：输出图像每个像素 (x,y) 对所有角度计算 s = x·cosθ + y·sinθ，去该角度
的投影行里**线性插值**查 s 处的值，累加后乘 Δθ。查值插值正是 GPU 优化的核心：
texture memory 的硬件双线性插值可以把两次全局读 + 插值指令压成一条 `tex2D`。

**SIRT 迭代重建**（阶段 6）：把成像写成线性方程组 b = A·x（A 的每行 = 一条射线
穿过图像的双线性权重），用最速下降型 SIRT 迭代求解。少角度时 FBP 的条纹伪影
被迭代正则化压下去，代价是几十到几百倍的计算量。

参考：Kak & Slaney, *Principles of Computerized Tomographic Imaging*, Ch.3（[官方免费 PDF](https://www.slaney.org/pct/pct-toc.html)）。

## 坐标与几何约定

全项目（CPU/GPU 共用，见 `include/ct/core/geometry.h`）：

```text
归一化视野 [-1,1]²，图像 N×N（row-major，index = i*N+j）
  第 i 行(自上而下)、第 j 列(自左向右)的像素中心：
    x = -1 + (j+0.5)·dx        （列 → +x 向右）
    y = +1 - (i+0.5)·dx        （行 0 在顶 = +y 向上，符合图像显示习惯）
探测器 n_det 单元：s_k = -1 + (k+0.5)·ds
角度 θ_i = i·π/n_angle 覆盖 [0°,180°)
射线参数化：x = s·cosθ − t·sinθ，y = s·sinθ + t·cosθ（t 沿射线方向）
```

前向/反向使用同一 s = x·cosθ + y·sinθ，自洽；单椭圆正弦轨迹测试
（`tests/test_pipeline.cpp`）锁定这套约定不被改坏。SIRT 的系统矩阵用同一套
采样公式构建，因此 `A·x` 与 GPU 前向投影逐点自洽。

## 构建（Windows）

要求：

- CUDA Toolkit ≥ 12.8（本机 13.1 与 13.5 均验证可编译；Blackwell 显卡需要 12.8+ 才有 sm_120）
- CMake ≥ 3.24
- Visual Studio 2022/2026（MSVC 14.2x~14.5x；见下方说明）
- Ninja（VS 自带：`Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/`）

一条命令（推荐，内部自动选 14.44 工具集 + Ninja；若 `build\` 已被 VS 生成器
占用则自动改用 `build-ninja\`）：

```bat
scripts\build.bat
```

或手动：

```bat
call "C:\...\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

产出三个可执行文件：`ct_fbp.exe`（主流水线）、`ct_tour.exe`（知识点巡礼）、
`test_pipeline.exe`（测试）。

> **注意**：CUDA ≤13.1 的 nvcc 只支持 MSVC 2019~2022（19.2x~19.4x）作为 host 编译器。
> 若默认工具集是 VS2026 的 19.51，而 CUDA 仍是 13.1，用 `-vcvars_ver=14.44` 固定到
> 已装的 VS2022 时代工具集即可；CUDA 13.5 已支持 VS2026 工具集，可直接用默认工具集。
> 若用 VS 生成器打开 `.sln`，还需在 VS 安装器里勾选 CUDA 的 "Visual Studio
> Integration" 组件（本仓库默认走 Ninja，无此依赖）。

VS 生成器（CMake Tools / `cmake --build build --config Debug|Release`）同样可用，
且 Debug/Release 都已验证能链接。

目标架构默认 `sm_120`（RTX 50 系）；其他显卡用
`-DCMAKE_CUDA_ARCHITECTURES=86`（30 系）/`89`（40 系）等覆盖。

CUDA 13.x 的 CCCL（thrust / libcu++）在 MSVC 传统预处理器下会直接 `#error`，
CMakeLists 已为所有目标统一加 `/utf-8` 与 `-Xcompiler=/Zc:preprocessor`，无需手工干预。

阶段 7 的动态并行（CDP2）需要 `-rdc=true` 与设备符号解析：`projector_cdp.cu` 单独编成
`ct_cdp` 静态库（`CUDA_SEPARABLE_COMPILATION` + `CUDA_RESOLVE_DEVICE_SYMBOLS`），
CMake 会自动插入 `nvcc -dlink` 并把设备链接产物并入 `ct_cdp.lib`，无需手工加参数。
链接该库会带出 `cudadevrt.lib`（NVIDIA 用静态 CRT 编译），因此 MSVC 会给出
`LNK4098: 默认库 LIBCMT 与其他库冲突` 的告警——这是 CDP 的正常现象，可忽略。

## 用法

```bat
build\ct_fbp.exe                              :: 默认 512×512, 512 探测器, 360 角度, CPU+GPU 对比
build\ct_fbp.exe --size 1024 --angles 720 --det 1024
build\ct_fbp.exe --mode cpu                   :: 只跑 CPU
build\ct_fbp.exe --filter hamming             :: Hamming 窗对比
build\ct_fbp.exe --out docs/results           :: 指定输出目录

:: 阶段 6：SIRT 迭代重建（少角度小配置；稠密矩阵受显存限制）
build\ct_fbp.exe --size 128 --det 128 --angles 16 --recon sirt --iters 60 --relax 1.0
build\ct_fbp.exe --size 128 --det 128 --angles 64 --recon sirt --iters 200 --matrix fp16
build\ct_fbp.exe --size 128 --det 128 --angles 16 --recon sirt --alloc managed

:: 阶段 7：前向投影变体（v1 / warp 归约 / 原子散射 / 动态并行）
build\ct_fbp.exe --mode gpu --forward warp
build\ct_fbp.exe --mode gpu --forward atomic
build\ct_fbp.exe --mode gpu --forward cdp

:: 阶段 8：多流异步流水线 / 多 GPU
build\ct_fbp.exe --mode gpu --streams 4
build\ct_fbp.exe --mode gpu --mgpu

:: 阶段 9：CUDA Graphs 重放基准
build\ct_fbp.exe --mode gpu --graph

:: 知识点巡礼：128×128×32 小配置一次跑完 11 组功能
build\ct_tour.exe
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--size N` | 512 | 图像边长 N×N |
| `--det D` | 512 | 探测器单元数 |
| `--angles A` | 360 | 投影角度数（覆盖 0~180°） |
| `--mode M` | both | `cpu` / `gpu` / `both`（both 时逐阶段比对） |
| `--filter F` | ram-lak | `ram-lak` / `hamming` |
| `--warmup W` | 3 | GPU 计时前预热次数 |
| `--runs R` | 5 | 计时重复次数取平均 |
| `--out DIR` | output | 输出目录 |
| `--recon R` | fbp | `fbp` / `sirt`（sirt 需 GPU，且 size/det/angles ≤ 128 量级） |
| `--iters N` | 60 | SIRT 迭代次数 |
| `--relax X` | 1.0 | SIRT 松弛因子 λ（须 < 2） |
| `--matrix M` | fp16 | SIRT 系统矩阵：`fp32`（TF32）/ `fp16`（Tensor Core） |
| `--alloc A` | device | SIRT 内存：`device` / `managed`（统一内存） |
| `--forward V` | v1 | 前向投影：`v1` / `warp` / `atomic` / `cdp` |
| `--streams K` | 0 | K > 0 时按角度分块跑 K 流异步流水 |
| `--mgpu` | 关 | 前向投影按角度切分到所有 GPU |
| `--graph` | 关 | 滤波→反投影链捕获为 CUDA Graph 重放 |

输出：`phantom.pgm`、`sinogram.pgm`、`sinogram_filtered.pgm`、
`recon_cpu.pgm`、`recon_gpu.pgm`、`recon_gpu_texture.pgm`，
SIRT 模式另有 `recon_sirt.pgm` 与 `sirt_convergence.csv`（迭代曲线），
以及控制台的质量/性能报告和 `timings.csv`。
PGM 可用 IrfanView/GIMP 等查看，或 `python tools/pgm2png.py xxx.pgm` 转 PNG。

### 输出编码

控制台输出会按宿主自动适配（实现见 `src/core/console_io.cpp`）：

- 控制台（cmd / Windows Terminal）：用 `WriteConsoleW` 输出，不受当前代码页影响；
- 管道 / 重定向文件（VS Code 调试控制台、CMake Tools 输出面板等）：把 UTF-8
  转成系统 ANSI 代码页（中文 Windows 即 GBK）再写出，避免被按 ANSI 解码时出现
  “閰嶇疆”式乱码；
- 需要 UTF-8 字节（CI 日志、按 UTF-8 解析的脚本）时设 `CT_OUTPUT_ENCODING=utf8`：

```bat
set CT_OUTPUT_ENCODING=utf8
build\ct_fbp.exe
```

## 测试

```bat
build\test_pipeline.exe       :: 或 cmake --build build --target test
```

9 组测试（无 CUDA 设备时 GPU 部分自动跳过）：

1. PGM 读写往返
2. 单椭圆正弦轨迹（坐标约定）
3. 中心圆盘 FBP 定标（重建值 ≈ 1）
4. Shepp-Logan CPU 全流程 RMSE
5. GPU vs CPU 逐阶段一致性
6. SIRT：少角度优于 FBP、fp16 vs fp32、统一内存、GPU 指标 vs CPU 指标
7. 前向投影变体（warp/atomic/cdp）与 v1 一致
8. 异步流水线 / 多 GPU 与串行 v1 一致
9. CUDA Graph 重放与手动链一致（naive 与 texture 两条链）

## 结果（RTX 5070 / CUDA 13.1）

完整数据、复现命令与负结果分析见 [docs/results/RESULTS.md](docs/results/RESULTS.md)。

### FBP 流水线（512×512×360）

| 阶段 | CPU (ms) | GPU (ms) | 加速比 |
|---|---|---|---|
| 前向投影 v1 | 599.28 | 1.03 | 581× |
| Ramp 滤波 | 33.94 | 0.05 | 637× |
| 反投影 naive | 188.16 | 0.32 | **583×** |
| 反投影 texture | — | 0.25 | −22%（相对 naive） |

| 配置 | RMSE | PSNR |
|---|---|---|
| 256×256×180 | 0.0458 | 26.95 dB |
| 512×512×360 | 0.0334 | 29.69 dB |
| 1024×1024×720 | 0.0250 | 32.21 dB |

### SIRT vs FBP（128×128，fp16 矩阵，λ=1.0）

| 角度 | 射线数 | FBP RMSE | SIRT 60 迭代 | SIRT 200 迭代 |
|---|---|---|---|---|
| 16 | 2048 | 0.2178 | **0.1425** | 0.1420 |
| 32 | 4096 | 0.1214 | **0.1080** | 0.1040 |
| 64 | 8192 | 0.0706 | 0.0888（未收敛） | **0.0664** |

角度越少 SIRT 优势越大（16 角度 −35%）；64 角度需 200 次迭代才反超 FBP。
单次迭代 0.5~1.3 ms，16 角度系统矩阵约 192 MiB（64 角度 768 MiB）。

| 体模 | 正弦图 | 重建（窄窗口） |
|---|---|---|
| ![phantom](docs/results/phantom.png) | ![sinogram](docs/results/sinogram.png) | ![recon](docs/results/recon_cpu_windowed.png) |

| 16 角度 FBP vs SIRT |
|---|
| ![sirt vs fbp](docs/results/sirt_vs_fbp_16ang.png) |

## CUDA 知识点全景

`ct_tour.exe` 在 128×128×32 配置上一键跑完全部功能，打印「知识点 → API → 实测」。
下表为 5 次运行的中位数（完整表见 [RESULTS.md §9](docs/results/RESULTS.md)）：

| # | 知识点 | 关键 API | 实测 |
|---|---|---|---|
| 1 | kernel 基础 / 线程映射 | `<<<grid, block>>>`、`__restrict__` | 0.042 ms / 前向投影 |
| 2 | 官方库调用 | `cufftPlanMany`、`cufftExecC2C` | 0.036 ms / 滤波 |
| 3 | 纹理内存 / 硬件双线性 | `cudaCreateTextureObject`、`tex2D` | 0.016 vs 0.009 ms（小图反而慢） |
| 4 | warp 原语 + 共享内存 | `__shfl_down_sync`、`__shared__`、`__syncthreads` | 0.022 ms，diff 1.8e-7 |
| 5 | 原子操作 | `atomicAdd` | 0.022 ms，diff 1.8e-7 |
| 6 | 动态并行（CDP2） | `cudaStreamTailLaunch`（`-rdc`） | 1.29 ms（31× 开销） |
| 7 | cuBLAS Tensor Core + Thrust | `cublasGemmEx`、`thrust::transform` | RMSE 0.106，0.99 ms/迭代 |
| 8 | 统一内存 | `cudaMallocManaged`、`cudaMemPrefetchAsync` | 与 device 结果一致 |
| 9 | 流 + 异步拷贝 + 锁页 | `cudaStream*`、`cudaMemcpyAsync`、`cudaMallocHost` | 0.24 vs 0.07 ms（小图开销） |
| 10 | 多 GPU | `cudaMemcpyPeerAsync`、`cudaDeviceCanAccessPeer` | 0.10 ms，1 设备退化 |
| 11 | CUDA Graphs | stream capture → instantiate → launch | 0.018 vs 0.058 ms（3.2×） |

收益随规模变化，报告保留"不赢"的配置：纹理内存在 512 以上 −22%，128 上反而慢；
warp/atomic 变体只在欠占用的小配置赢（128 上 1.9×，512 上慢 1.8~2.1×）；
CUDA Graphs 只在 launch-bound 的小配置赢（3~5×，512 上仅 5~9%）；多流流水线在
当前单 kernel 链上不划算（512 中位 +23%，1024 无稳定收益），是"重 kernel / 大传输"
场景的基础设施。这些负结果与正向加速比一起构成优化的边界说明。

## 实现说明

| Kernel / 模块 | 线程映射 | 要点 |
|---|---|---|
| 前向投影 v1 | 1 线程 = 1 (角度, 探测器) | 等距步进采样 + 双线性插值；积分范围按外接圆 \|t\| ≤ √(2−s²) 截断省 ~35% 空步 |
| 前向投影 warp | 1 block = 1 射线（256 线程） | 线程交错采样 → `__shfl_down_sync` + `__shared__` 两级归约 |
| 前向投影 atomic | 1 线程 = 1 (射线, 32 样本块) | `atomicAdd` 直接累加，同地址串行化是代价样本 |
| 前向投影 cdp | 父 kernel 按角度 launch 子 kernel | CDP2 + `cudaStreamTailLaunch`，演示嵌套 launch 开销 |
| Ramp 滤波 | 1 线程 = 1 补零元素 / 频域元素 | cuFFT 批量 1D C2C；plan/频响/缓冲按几何缓存 |
| 反投影 v1 naive | 1 线程 = 1 像素 (16×16 block) | cos/sin 预计算查表，手写线性插值，出界取 0 |
| 反投影 v2 texture | 同上 | 正弦图驻留 cudaArray，`tex2D` 硬件双线性插值 |
| SIRT | cuBLAS gemv（以 gemm n=1 实现） | `cublasGemmEx` + Tensor Core（TF32 / FP16）；Thrust 做向量步骤；行/列归一化预计算 |
| 多流流水线 | K 流 × 角度块 | `cudaMemcpyAsync` + `cudaMallocHost` 锁页缓冲，块间重叠 |
| 多 GPU | 按角度切分设备 | peer 可达时 `cudaMemcpyPeerAsync` 汇聚到 0 号卡，否则各卡 D2H |
| CUDA Graphs | 整链 1 次 launch | capture 前热身（cuFFT plan / 纹理对象不可在捕获区分配），exec 按键缓存 |

细节：

- **1/M 归一化折叠进频响**：`q = IFFT(FFT(p)·|ν_m|·w_m/M)`，CPU 自写 FFT 与
  cuFFT 均不带归一化，两侧数学完全一致（滤波后正弦图误差 ~1e-6·量级）。
- **texture 插值量化**：硬件线性过滤的小数权重按 1/512 步长量化，对重建图引入
  ~1e-4 的 RMSE，肉眼不可分辨；换来纹理缓存命中 + 免插值指令，512 配置快 ~22%。
- **缓存策略**：三角函数表、cudaArray/texture object、cuFFT plan、SIRT 系统矩阵、
  Graph exec 均按键（几何/窗口/精度/源指针）缓存，计时循环内无重复分配。
- **计时口径**：CUDA Events 只含设备端计算，H2D/D2H 拷贝在计时区外；CPU 为
  `std::chrono` 单线程。

## 目录结构

模块约定：公共 API 放 `include/ct/<模块>/`，实现在 `src/<模块>/`，两棵目录一一对应；
每个模块编译为同名静态库（`ct_projector`、`ct_filter`、`ct_backprojector`…），`recon`
依赖各功能模块，可执行文件只链接模块库。`core` 存放坐标约定、计时与内部 CUDA 工具，
`app` 只放可执行入口；CDP 因 `-rdc` 约束单独成 `ct_cdp` 库，由 projector 模块聚合。

```text
ct-fbp-cuda/
├── CMakeLists.txt
├── scripts/build.bat        # Windows 一键构建（vcvars + Ninja）
├── tools/pgm2png.py         # PGM→PNG 显示辅助（需 Pillow）
├── include/ct/              # 公共 API，按模块分目录
│   ├── core/                #   geometry.h（坐标约定）、timer.h（CUDA Events 计时）
│   ├── phantom/             #   phantom.h（Shepp-Logan 体模）
│   ├── projector/           #   projector.h（前向投影：CPU / GPU / warp / atomic / CDP）
│   ├── filter/              #   filter.h（Ramp 滤波：CPU / cuFFT）
│   ├── backprojector/       #   backprojector.h（反投影：CPU / naive / texture）
│   ├── metrics/             #   metrics.h（RMSE / PSNR）
│   ├── io/                  #   io.h（PGM 读写）
│   └── recon/               #   pipeline.h / sirt.h / graph.h（重建算法层）
├── src/                     # 实现，目录与 include/ct/ 一一对应
│   ├── app/                 #   main.cpp（CLI 主程序）、tour.cpp（ct_tour：11 组知识点巡礼）
│   ├── core/                #   cuda_check.h、cuda_utils.cuh（内部工具头）、console_io.cpp（输出编码适配）
│   ├── phantom/             #   phantom.cu
│   ├── projector/           #   projector_cpu.cpp / projector_gpu.cu / projector_cdp.cu
│   ├── filter/              #   filter_cpu.cpp / filter_gpu.cu
│   ├── backprojector/       #   backprojector_cpu.cpp / backprojector_gpu.cu
│   ├── metrics/             #   metrics.cpp / metrics_gpu.cu
│   ├── io/                  #   io.cpp
│   └── recon/               #   pipeline_gpu.cu（多流/多 GPU）、sirt_gpu.cu（阶段 6）、graph_gpu.cu（阶段 9）
├── tests/test_pipeline.cpp  # 零依赖测试（9 组）
├── plan/PLAN.md             # 阶段计划与验收标准
└── docs/results/            # 输出图像 + 质量性能报告（RESULTS.md）
```

## 扩展路线

- **扇束 FBP**：点光源几何 + cos³ 权重（等距重排或直接扇束公式）——阶段 5，本仓库未实现
- **Siddon 精确前向投影**：替代等距采样，消除采样步长依赖
- **稀疏 / on-the-fly SIRT**：不落盘稠密 A，用 kernel 现算 A·x 与 Aᵀ·r，突破 128×128 限制
- **真多卡**：≥2 卡上复测 peer 汇聚与负载切分（当前只验证了单卡退化路径）
- **Nsight Compute 剖析**：合并访问、占用率、纹理命中率的量化截图
