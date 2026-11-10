#pragma once

// 跨编译单元共享的 device 辅助函数。
// （kernel 内的辅助函数若留在各 .cu 的匿名 namespace 里则无法复用，故提升到此处；
//   仅限 src/ 内部使用，与 cuda_check.h 同放 src/core/，不进 include/ct/ 公开 API。）

#include <cmath>

// 图像双线性采样：物理坐标 (x, y) → 像素值，出界取 0。
// 采样公式与 CPU 参考实现逐字一致（列 → +x 向右；行 0 在顶部 = +y 向上，
// 见 include/ct/core/geometry.h 的坐标约定），保证 GPU 各 kernel 间结果可对齐。
__device__ __forceinline__ float bilinearSample(const float* __restrict__ img, int n,
                                                float x, float y) {
    float dx = 2.0f / n;
    float fx = (x + 1.0f) / dx - 0.5f;  // 列方向（x → j）
    float fy = (1.0f - y) / dx - 0.5f;  // 行方向（y → i）
    int j0 = (int)floorf(fx), i0 = (int)floorf(fy);
    float wx = fx - j0, wy = fy - i0;
    auto at = [&](int jj, int ii) -> float {
        return (jj >= 0 && jj < n && ii >= 0 && ii < n) ? img[(size_t)ii * n + jj] : 0.0f;
    };
    return at(j0, i0) * (1 - wx) * (1 - wy) + at(j0 + 1, i0) * wx * (1 - wy) +
           at(j0, i0 + 1) * (1 - wx) * wy + at(j0 + 1, i0 + 1) * wx * wy;
}
