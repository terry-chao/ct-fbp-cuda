// CPU 反投影参考实现：每像素对所有角度查投影值（线性插值，出界为 0），乘 Δθ。
#include "ct/backprojector/backprojector.h"

#include <cmath>
#include <vector>

void backprojectCPU(const float* sino, const Geometry& g, float* out) {
    const float ds = g.ds();
    const float dtheta = CT_PI / g.n_angle;

    // 预计算三角函数表（每角度一次，避免内层重复求 cos/sin）
    std::vector<float> cosT(g.n_angle), sinT(g.n_angle);
    for (int a = 0; a < g.n_angle; ++a) {
        float th = g.angle(a);
        cosT[a] = std::cos(th);
        sinT[a] = std::sin(th);
    }

    for (int i = 0; i < g.n; ++i) {          // 行
        float y = g.pixelY(i);
        for (int j = 0; j < g.n; ++j) {      // 列
            float x = g.pixelX(j);
            float acc = 0.f;
            for (int a = 0; a < g.n_angle; ++a) {
                float s = x * cosT[a] + y * sinT[a];
                float f = (s + 1.0f) / ds - 0.5f;  // 探测器浮点索引
                int k0 = (int)std::floor(f);
                float w = f - k0;
                const float* row = sino + (size_t)a * g.n_det;
                float v0 = (k0 >= 0 && k0 < g.n_det) ? row[k0] : 0.f;
                float v1 = (k0 + 1 >= 0 && k0 + 1 < g.n_det) ? row[k0 + 1] : 0.f;
                acc += v0 + w * (v1 - v0);
            }
            out[(size_t)i * g.n + j] = acc * dtheta;
        }
    }
}
