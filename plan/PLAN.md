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

### 阶段 6：迭代重建对比（可选，长期）
- [ ] SIRT 迭代重建（投影/反投影反复迭代）
- **验收**：少角度（如 45 个角度）场景下 SIRT 质量优于 FBP，展示"GPU 让迭代法变得可行"

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
