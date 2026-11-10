#pragma once

#include "geometry.h"

// 反投影：滤波后正弦图 → 重建图像。
// 每个输出像素 (x, y) 对所有角度 θ 累加：
//   s = x*cosθ + y*sinθ → 在该角度投影行里线性插值查 s 处的值 → 乘 Δθ = π/n_angle

// CPU 参考实现（单线程）
void backprojectCPU(const float* sino, const Geometry& g, float* out);

// GPU naive：1 线程 = 1 输出像素，全局内存 + 手写线性插值（三角函数查表）
void backprojectGPU(const float* d_sino, const Geometry& g, float* d_out);

// GPU texture：正弦图绑定 2D texture（cudaArray + 硬件双线性插值 tex2D）
// 注：每次调用会把最新数据拷入缓存的 cudaArray（D2D），属于该实现的固有开销。
void backprojectGPUTexture(const float* d_sino, const Geometry& g, float* d_out);

// 释放反投影模块的静态缓存（三角函数表 / cudaArray / texture object）
void backprojectorGPUReleaseCache();
