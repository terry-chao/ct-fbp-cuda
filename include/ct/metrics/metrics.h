#pragma once

#include <cstddef>

// 质量指标：test 相对 ref 的误差统计。
struct Metrics {
    double rmse;    // 均方根误差
    double psnr;    // 峰值信噪比（dB），ref 的动态范围作为峰值；rmse 为 0 时为 INFINITY
    double max_abs; // 最大绝对误差
};

Metrics computeMetrics(const float* ref, const float* test, size_t n);

// GPU 版（阶段 6，Thrust 实现）：d_ref/d_test 为设备指针，三次并行扫描
// （平方差和 / 最大绝对差 / ref 最值），迭代循环内调用零 D2H 同步。
Metrics computeMetricsGPU(const float* d_ref, const float* d_test, size_t n);
