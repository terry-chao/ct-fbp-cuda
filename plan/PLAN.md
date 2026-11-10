# ct-fbp-cuda 项目方案

> 基于 CUDA 的 CT 图像滤波反投影（FBP）重建器
> 从模拟扫描到 GPU 加速重建的完整流水线

---

## 1. 项目概述

**一句话**：程序生成一张模拟人体头颅的测试图（Shepp-Logan 体模），模拟 X 光扫描得到正弦图（sinogram），再用 GPU 上的滤波反投影算法重建出断层图像，并给出 CPU/GPU 性能对比。

**目标读者/用途**：CUDA 学习项目 / 课程设计 / 简历作品。

**核心价值**：
- 覆盖 CUDA 三大基本功：朴素 kernel、texture memory 插值、内存合并访问优化
- 算法结果可视化（重建图像肉眼可验证），性能提升可量化（预期 50~200×）
- 有清晰的进阶路径（扇束 → 迭代重建），可以按需扩展

## 2. 交付物

| # | 交付物 | 说明 |
|---|--------|------|
| 1 | 可编译的 CMake 工程 | Windows + CUDA Toolkit，一条命令构建 |
| 2 | 完整流水线程序 | 体模 → 前向投影 → 滤波 → 反投影 → 输出图像 |
| 3 | CPU 参考实现 | 全套算法的 CPU 版，作为正确性基准和性能对照 |
| 4 | 图像输出 | 体模图、正弦图、重建图（PGM 格式，零依赖） |
| 5 | 质量与性能报告 | RMSE/PSNR 误差指标 + CUDA Events 计时的加速比表格 |
| 6 | README | 原理简介（带图）、构建方法、结果截图 |

## 3. 技术栈与环境

| 项 | 选择 | 备注 |
|----|------|------|
| 语言 | C++17 + CUDA C++ | |
| 构建 | CMake ≥ 3.24 | `find_package(CUDAToolkit)` |
| IDE | Visual Studio 2022 | Windows + nvcc |
| 图像 IO | 自写 PGM 读写（~50 行） | 零依赖；后期可换 OpenCV |
| FFT | cuFFT（GPU）/ 自写 DFT（CPU 参考） | 只用于滤波步骤 |
| 计时 | CUDA Events + `std::chrono` | GPU/CPU 分别计时 |
| 剖析 | Nsight Compute | 优化阶段使用 |

**默认参数**（可命令行覆盖）：

```text
图像尺寸 N×N        512×512
探测器单元数 n_det   512
投影角度数 n_angle   360（覆盖 0°~180°）
数据类型            float32
```

## 4. 理论基础（实现要点速查）

```text
                    ┌──────────────────────────────────────────┐
 Shepp-Logan 体模 → │ 前向投影 (Radon) → 正弦图 → Ramp滤波 → 反投影 │ → 重建图像
                    └──────────────────────────────────────────┘
```

1. **前向投影**：对每个角度 θ，沿一束平行射线积分图像，得到一条 1D 投影 p(θ, s)。所有角度的投影按行堆叠 = 正弦图。
2. **Ramp 滤波（Ram-Lak）**：对正弦图**每一行**做 1D 高通滤波，预先补偿反投影造成的模糊。实现走 FFT 域：`FFT → 乘 |f| 频响 → IFFT`。可加 Hamming 窗抑制振铃。
3. **反投影**：输出图像每个像素 (x, y)，对所有角度 θ 计算 `s = x·cosθ + y·sinθ`，去该角度的投影行里查 s 处的值（**线性插值**），累加后乘 Δθ。
4. **关键点**：反投影中"查 s 处的值"是插值操作 —— GPU 上用 texture memory 的硬件双线性插值，是本项目最核心的优化点。

## 5. 模块设计

```text
ct-fbp-cuda/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── phantom.h        # Shepp-Logan 体模生成接口
│   ├── projector.h      # 前向投影（CPU/GPU）
│   ├── filter.h         # Ramp 滤波（CPU/GPU）
│   ├── backprojector.h  # 反投影（CPU/GPU naive/GPU texture）
│   ├── io.h             # PGM 读写
│   ├── metrics.h        # RMSE / PSNR
│   └── timer.h          # CUDA Events 计时封装
├── src/
│   ├── phantom.cu
│   ├── projector_cpu.cpp / projector_gpu.cu
│   ├── filter_cpu.cpp   / filter_gpu.cu     # cuFFT
│   ├── backprojector_cpu.cpp / backprojector_gpu.cu
│   ├── io.cpp
│   ├── metrics.cpp
│   └── main.cpp         # CLI：--size --angles --det --mode cpu|gpu|both
├── tests/
│   └── test_pipeline.cpp # GPU vs CPU 输出一致性（容差判断）
└── docs/
    └── results/          # 输出图像与性能表格（跑出来后提交）
```

**kernel 设计一览**：

| Kernel | 线程映射 | 要点 |
|--------|----------|------|
| 前向投影 | 1 线程 = 1 (角度, 探测器) | 沿射线步进求和（Siddon 或等距采样） |
| 滤波 | 1 线程 = 1 探测器单元（逐行批处理） | cuFFT 批量 1D FFT |
| 反投影 v1（naive） | 1 线程 = 1 输出像素 | 手写线性插值 |
| 反投影 v2（texture） | 同上 | `tex2D` 硬件插值 + 合并访存 |

## 6. 阶段计划与验收标准

> 每阶段结束提交一次 git commit，附输出图像。

### 阶段 0：CPU 参考实现（约 1 周）
- [ ] 工程骨架、PGM 读写、Shepp-Logan 生成
- [ ] CPU 前向投影 → 保存正弦图，确认条纹形态正确
- [ ] CPU Ramp 滤波 + 反投影 → 重建出图像
- **验收**：重建图与体模肉眼一致，RMSE 可计算；用"单点/单椭圆"用例验证坐标约定无误

### 阶段 1：GPU 前向投影（约 3 天）
- [ ] `projector_gpu.cu` naive kernel
- **验收**：GPU 正弦图与 CPU 版最大误差 < 1e-4（float 累加顺序差异允许放宽）

### 阶段 2：GPU 滤波（约 2 天）
- [ ] cuFFT 批量实现 Ramp 滤波
- **验收**：滤波后正弦图与 CPU 版一致；对比加/不加 Hamming 窗的振铃差异

### 阶段 3：GPU 反投影（约 1 周）
- [ ] naive kernel（一像素一线程，手写插值）
- [ ] 串联完整 GPU 流水线，CLI 支持 `--mode both` 同屏对比
- **验收**：GPU 重建图与 CPU 参考一致；跑通 CPU vs GPU 计时表

### 阶段 4：性能优化（约 1 周）
- [ ] texture memory 版反投影
- [ ] Nsight Compute 剖析：检查合并访问、占用率
- [ ] 参数扫描：角度数/尺寸变化下的加速比曲线
- **验收**：512×512×360 配置下，反投影 GPU 相对单线程 CPU 加速 ≥ 50×；有剖析截图佐证优化效果

### 阶段 5：扇束扩展（约 1~2 周，可选）
- [ ] 扇束前向投影模拟（点光源几何）
- [ ] 扇束 FBP（cos³ 权重 + 等距重排 or 直接扇束公式）
- **验收**：扇束几何下重建成功，README 增加扇束小节

### 阶段 6：迭代重建对比（可选，长期）—— 已完成（cuBLAS/Thrust/统一内存）
- [x] SIRT 迭代重建：稠密系统矩阵 A（射线双线性权重，与前向投影采样自洽）
- [x] cuBLAS GemmEx 两种 Tensor Core 路径：fp32+TF32（`CUBLAS_COMPUTE_32F_FAST_TF32`）、
      fp16 矩阵（A/B/C 16F + 32F 累加，显存减半）
- [x] Thrust 设备端迭代步骤（transform/inner_product/for_each+zip）；`computeMetricsGPU`
- [x] 统一内存模式（`--alloc managed`，WDDM 下 prefetch 自动禁用）
- [x] CLI：`--recon sirt --iters N --relax X --matrix fp32|fp16 --alloc device|managed`
- **验收**：少角度（16 角度，128×128）SIRT RMSE 显著优于 FBP（实测 0.142 vs 0.218）；
  fp16 与 fp32 结果 RMSE 差 < 5e-3；managed 与 device 结果一致；GPU 指标 = CPU 指标

### 阶段 7：前向投影并行变体（warp/atomic/CDP）—— 已完成
- [x] warp 协作版：1 block = 1 射线，`__shfl_down_sync` + `__shared__` + `__syncthreads` 归约
- [x] 原子散射版：线程 = 射线分段，`atomicAdd` 直接累加（与归约版对比）
- [x] 动态并行版（CDP2）：父 kernel 按角度 launch 子 kernel（`cudaStreamTailLaunch`，-rdc）
- **验收**：三变体正弦图与 naive 版 max_abs < 1e-4；RESULTS 给出四版性能对比与结论

### 阶段 8：异步流水线与多 GPU（streams/pinned/mgpu）—— 已完成
- [x] 现有 GPU 阶段函数加 `cudaStream_t` 参数（默认 nullptr 保持原行为）
- [x] 前向投影分角度块多流流水线：K 流 + `cudaMemcpyAsync` + `cudaMallocHost` 双缓冲
- [x] 多 GPU 版：按角度切分设备，peer copy 或各自 D2H（单卡退化为分块，如实记录）
- **验收**：流水线正弦图 == 串行（< 1e-6）；RESULTS 记录重叠收益/开销

### 阶段 9：CUDA Graphs + 知识点巡礼 —— 已完成
- [x] 滤波→反投影链 stream capture → instantiate → replay，量化 launch 开销
- [x] `ct_tour` 二进制：小配置一键跑全部知识点，输出「知识点 → API → 实测」对照表
- **验收**：graph 重放结果 == 串行；README 增「CUDA 知识点全景」章节

## 7. 性能与质量评估方案

**正确性**（每次跑流水线自动输出）：
- RMSE、PSNR（重建图 vs 体模）
- GPU vs CPU 中间结果逐阶段比对，定位问题出在哪一步

**性能**（CUDA Events，预热 3 次取均值）：

| 配置 | CPU (ms) | GPU naive (ms) | GPU texture (ms) | 加速比 |
|------|----------|----------------|------------------|--------|
| 256×256×180 | | | | |
| 512×512×360 | | | | |
| 1024×1024×720 | | | | |

## 8. 风险与应对

| 风险 | 应对 |
|------|------|
| 坐标系/角度约定搞混（最常见 bug 来源） | 阶段 0 先用"单个点/单个椭圆"做单元测试，肉眼确认正弦图是正弦曲线 |
| Ramp 滤波出现振铃伪影 | FFT 前后补零；加 Hamming/Hann 窗 |
| GPU/CPU 结果对不上但差异很小 | float 求和顺序不同属正常，设合理容差（如 max_err < 1e-3） |
| Windows + nvcc 环境问题 | CMake 指定 CUDA arch（如 75/86/89 按显卡）；README 写清步骤 |
| 前向投影累加慢 | 先用等距步进采样（够用），Siddon 算法留作优化 |

## 9. 参考资料

- Kak & Slaney, *Principles of Computerized Tomographic Imaging* —— FBP 圣经，第 3 章直接对应本项目（官网有免费 PDF）
- NVIDIA CUDA C++ Programming Guide —— texture memory、CUDA Events 章节
- cuFFT 官方文档
- Shepp-Logan phantom 标准参数（维基百科即有）

## 10. 里程碑时间线（业余时间估算）

```text
第 1 周      ████ 阶段 0：CPU 参考（里程碑 M1：跑通全流程）
第 2 周      ████ 阶段 1+2：GPU 投影与滤波
第 3 周      ████ 阶段 3：GPU 反投影（里程碑 M2：全 GPU 流水线）
第 4 周      ████ 阶段 4：优化（里程碑 M3：达到加速目标，出报告）
第 5~6 周    ▒▒▒▒ 阶段 5/6：扇束 / 迭代（可选扩展）
```
