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
