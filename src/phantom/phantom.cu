// Shepp-Logan 体模生成（解析法，主机代码；编译为 .cu 与工程其余部分保持一致）
#include "ct/phantom/phantom.h"
#include "ct/core/geometry.h"

#include <cmath>

// MATLAB phantom('shepp-logan') 参数（对比度增强版）：
//   a       b       x0     y0      倾角   取值
static const Ellipse kSheppLogan[] = {
    {0.6900f, 0.9200f, 0.0f,    0.0f,    0.0f, 1.00f},   // 颅骨外圈
    {0.6624f, 0.8740f, 0.0f,    0.0f,    0.0f, -0.98f},  // 颅骨内圈（差值 0.02 形成壳层）
    {0.1100f, 0.3100f, 0.0f,    0.2200f, 0.0f, -0.02f},  // 脑内特征
    {0.1600f, 0.4100f, 0.0f,   -0.2200f, 0.0f, -0.02f},
    {0.2100f, 0.2500f, 0.3500f, 0.0f,    0.0f, 0.01f},   // 右侧小球
    {0.0460f, 0.0460f, 0.0f,    0.1000f, 0.0f, 0.01f},   // “眼”
    {0.0460f, 0.0460f, 0.0f,   -0.1000f, 0.0f, 0.01f},
    {0.0460f, 0.0230f, -0.0800f, -0.6050f, 0.0f, 0.01f}, // “脑干”三小椭
    {0.0230f, 0.0230f, 0.0f,   -0.6060f, 0.0f, 0.01f},
    {0.0230f, 0.0460f, 0.0600f, -0.6050f, 0.0f, 0.01f},
};

const Ellipse* sheppLoganTable(int& count) {
    count = (int)(sizeof(kSheppLogan) / sizeof(kSheppLogan[0]));
    return kSheppLogan;
}

std::vector<float> makeEllipses(int n, const Ellipse* ellipses, int count) {
    std::vector<float> img((size_t)n * n, 0.f);
    Geometry grid{};
    grid.n = n;
    for (int i = 0; i < n; ++i) {          // 行（自上而下 = y 从 +1 递减）
        float y = grid.pixelY(i);
        for (int j = 0; j < n; ++j) {      // 列（自左向右 = x 从 -1 递增）
            float x = grid.pixelX(j);
            float v = 0.f;
            for (int e = 0; e < count; ++e) {
                const Ellipse& el = ellipses[e];
                float dx = x - el.x0, dy = y - el.y0;
                float ca = std::cos(el.alpha), sa = std::sin(el.alpha);
                float u = (dx * ca + dy * sa) / el.a;   // 旋转后椭圆坐标
                float w = (-dx * sa + dy * ca) / el.b;
                if (u * u + w * w <= 1.f) v += el.val;
            }
            img[(size_t)i * n + j] = v;
        }
    }
    return img;
}

std::vector<float> makeSheppLogan(int n) {
    int count = 0;
    const Ellipse* table = sheppLoganTable(count);
    return makeEllipses(n, table, count);
}
