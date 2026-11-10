#pragma once

#include "geometry.h"

// Ramp（Ram-Lak）滤波：对正弦图每行做 1D 高通滤波，频响 |ν|，可选窗抑制振铃。
//
// 实现走 FFT 域：行末尾补零到 M = pow2 >= 2*n_det，FFT → 乘 |ν|·窗/M → IFFT → 裁回。
// 公式（与 CPU/GPU 完全一致）：
//   ν_m = wrap(m) / (M * ds)，wrap(m) = m (m <= M/2) 或 m - M (m > M/2)
//   q   = IFFT( FFT(p) * |ν_m| * w_m / M )      （IFFT 含 1/M 归一化，折叠进频响）
//   Hamming 窗：w_m = 0.54 + 0.46 * cos(2*pi*m/M)

enum class FilterWindow {
    RamLak,   // 纯 ramp
    Hamming,  // ramp * Hamming 窗
};

// 滤波用补零长度
int paddedLength(int n_det);

// CPU 版：自写 radix-2 复数 FFT
void rampFilterCPU(const float* sino, const Geometry& g, FilterWindow win, float* out);

// GPU 版：cuFFT 批量 1D FFT（plan 与缓冲按几何缓存，重复调用零额外开销）
void rampFilterGPU(const float* d_sino, const Geometry& g, FilterWindow win, float* d_out);

// 释放 GPU 滤波的静态缓存（进程退出前调用，非必须）
void rampFilterGPUReleaseCache();
