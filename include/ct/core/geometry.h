#pragma once

// ============================================================================
// 统一的几何约定（全项目共用，CPU/GPU 使用同一套公式）
//
//   归一化视野 FOV = 2.0：
//     - 图像覆盖 [-1,1] x [-1,1]，N x N 像素
//     - 探测器单元中心 s_k 覆盖 (-1,1)
//
//   图像存储：row-major，index = i*N + j
//     第 i 行（自上而下）、第 j 列（自左向右）的像素中心：
//       x_j = -1 + (j + 0.5) * dx        （列 → +x 向右）
//       y_i = +1 - (i + 0.5) * dx        （行 0 在图像顶部 = +y 向上）
//
//   探测器：n_det 个单元
//       s_k = -1 + (k + 0.5) * ds        （k → +s 向右）
//
//   角度：theta_i = i * pi / n_angle，覆盖 [0, 180)
//
//   平行束射线参数化（Radon 变换约定）：
//       p(theta, s) = ∫ f(s*cos - t*sin, s*sin + t*cos) dt
//     即射线方向 d = (-sin, cos)，s 是射线到原点的（带符号）距离，
//     反投影查值时同样用 s = x*cos + y*sin —— 前向/反向自洽。
// ============================================================================

#ifdef __CUDACC__
#define CT_HOSTDEV __host__ __device__
#else
#define CT_HOSTDEV
#endif

#include <cstddef>

constexpr float CT_PI = 3.14159265358979323846f;

struct Geometry {
    int n = 512;        // 图像边长（N x N 像素）
    int n_det = 512;    // 探测器单元数
    int n_angle = 360;  // 投影角度数（覆盖 [0, pi)）

    CT_HOSTDEV float dx() const { return 2.0f / n; }      // 像素宽度
    CT_HOSTDEV float ds() const { return 2.0f / n_det; }  // 探测器单元宽度
    CT_HOSTDEV float angle(int i) const { return i * (CT_PI / n_angle); }
    CT_HOSTDEV float pixelX(int j) const { return -1.0f + (j + 0.5f) * dx(); }  // 列 j
    CT_HOSTDEV float pixelY(int i) const { return 1.0f - (i + 0.5f) * dx(); }   // 行 i
    CT_HOSTDEV float detS(int k) const { return -1.0f + (k + 0.5f) * ds(); }

    size_t imagePixels() const { return (size_t)n * n; }
    size_t sinoPixels() const { return (size_t)n_angle * n_det; }
};
