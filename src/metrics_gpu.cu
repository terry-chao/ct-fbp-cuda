// GPU 指标计算（阶段 6）：Thrust inner_product（带二元变换的归约）/ minmax_element，
// 与 CPU 版 computeMetrics 同一公式（rmse 平方差和、max_abs、ref 动态范围作 PSNR 峰值）。
#include "metrics.h"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

#include <cmath>
#include <limits>

namespace {

struct AbsDiff {
    __device__ float operator()(float a, float b) const { return fabsf(a - b); }
};

struct SqDiff {
    __device__ double operator()(float a, float b) const {
        double d = (double)b - (double)a;
        return d * d;
    }
};

}  // namespace

Metrics computeMetricsGPU(const float* d_ref, const float* d_test, size_t n) {
    Metrics m{0.0, 0.0, 0.0};
    if (n == 0) {
        m.psnr = std::numeric_limits<double>::infinity();
        return m;
    }

    m.max_abs = thrust::inner_product(thrust::device, d_ref, d_ref + n, d_test, 0.0f,
                                      thrust::maximum<float>(), AbsDiff());
    double se = thrust::inner_product(thrust::device, d_ref, d_ref + n, d_test, 0.0,
                                      thrust::plus<double>(), SqDiff());
    m.rmse = std::sqrt(se / (double)n);

    // 注：thrust::minmax_element 在本机 CCCL/MSVC 环境触发宿主侧崩溃，
    // 改用两次 reduce（与 inner_product 相同的代码路径，已验证稳定）。
    float vmin = thrust::reduce(thrust::device, d_ref, d_ref + n, 3.4e38f,
                                thrust::minimum<float>());
    float vmax = thrust::reduce(thrust::device, d_ref, d_ref + n, -3.4e38f,
                                thrust::maximum<float>());
    double range = (double)vmax - (double)vmin;
    m.psnr = m.rmse > 1e-12 ? 20.0 * std::log10(range / m.rmse)
                            : std::numeric_limits<double>::infinity();
    return m;
}
