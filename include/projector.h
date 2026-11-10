#pragma once

#include "geometry.h"

// 前向投影（模拟扫描）：图像 → 正弦图。
// 正弦图存储：row-major，n_angle 行 x n_det 列，第 a 行是角度 theta_a 的投影。
//
// 实现：等距步进采样 + 双线性插值（CPU/GPU 使用完全相同的采样公式），
//       沿射线积分范围按内切圆截断：|t| <= sqrt(2 - s^2)。

// CPU 参考实现（单线程）
void forwardProjectCPU(const float* img, const Geometry& g, float* sino);

// GPU naive kernel：1 线程 = 1 (角度, 探测器单元)，沿射线步进求和。
// d_img / d_sino 均为设备指针，由调用方分配。
void forwardProjectGPU(const float* d_img, const Geometry& g, float* d_sino);

// ---- 阶段 7：同一积分的三种并行化变体（结果与 v1 在容差内一致）--------------
// 沿射线的积分是归约，三个变体展示不同的跨线程求和手段：

// v2 warp 协作：1 block = 1 射线，线程交错采样，__shfl_down_sync warp 归约 +
// __shared__ 树形合并（经典两级块内归约）。
void forwardProjectGPUWarp(const float* d_img, const Geometry& g, float* d_sino);

// v3 原子散射：1 线程 = 1 (射线, 样本块)，atomicAdd 直接累加进正弦图
// （直方图式累加；调用前内部清零 d_sino）。
void forwardProjectGPUAtomic(const float* d_img, const Geometry& g, float* d_sino);

// v4 动态并行（CDP2，教学演示）：父 kernel 每角度在设备端 launch 子 kernel
// （cudaStreamTailLaunch，-rdc 编译）。预期不快于 v1，用于实测嵌套 launch 开销。
void forwardProjectGPUCdp(const float* d_img, const Geometry& g, float* d_sino);
