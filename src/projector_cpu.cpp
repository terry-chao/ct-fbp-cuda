// CPU 前向投影参考实现：等距步进采样 + 双线性插值
#include "projector.h"

#include <cmath>

// 双线性采样：把连续坐标 (x, y) 映射到像素网格（Geometry 约定）。
// 图像外一律返回 0（体模外为空）。
static float bilinearSample(const float* img, int n, float x, float y) {
    float dx = 2.0f / n;
    float fx = (x + 1.0f) / dx - 0.5f;  // 列方向（x → j）
    float fy = (1.0f - y) / dx - 0.5f;  // 行方向（y → i，行 0 在上）
    int j0 = (int)std::floor(fx), i0 = (int)std::floor(fy);
    float wx = fx - j0, wy = fy - i0;
    auto at = [&](int jj, int ii) -> float {
        return (jj >= 0 && jj < n && ii >= 0 && ii < n) ? img[(size_t)ii * n + jj] : 0.0f;
    };
    return at(j0, i0) * (1 - wx) * (1 - wy) + at(j0 + 1, i0) * wx * (1 - wy) +
           at(j0, i0 + 1) * (1 - wx) * wy + at(j0 + 1, i0 + 1) * wx * wy;
}

void forwardProjectCPU(const float* img, const Geometry& g, float* sino) {
    const float dx = g.dx(), ds = g.ds();
    const float dt = dx < ds ? dx : ds;  // 采样步长：不劣于图像/探测器网格
    for (int a = 0; a < g.n_angle; ++a) {
        float th = g.angle(a);
        float c = std::cos(th), s = std::sin(th);
        for (int k = 0; k < g.n_det; ++k) {
            float sk = g.detS(k);
            // 射线在半径 sqrt(2) 的外接圆内的长度：|t| <= sqrt(2 - s^2)
            float tmax = std::sqrt(std::fmax(0.f, 2.f - sk * sk));
            float sum = 0.f;
            for (float t = -tmax; t <= tmax; t += dt) {
                sum += bilinearSample(img, g.n, sk * c - t * s, sk * s + t * c);
            }
            sino[(size_t)a * g.n_det + k] = sum * dt;
        }
    }
}
