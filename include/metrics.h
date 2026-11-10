#pragma once

#include <cstddef>

// 质量指标：test 相对 ref 的误差统计。
struct Metrics {
    double rmse;    // 均方根误差
    double psnr;    // 峰值信噪比（dB），ref 的动态范围作为峰值；rmse 为 0 时为 INFINITY
    double max_abs; // 最大绝对误差
};

Metrics computeMetrics(const float* ref, const float* test, size_t n);
