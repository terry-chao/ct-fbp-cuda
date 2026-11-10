#pragma once

#include "filter.h"
#include "geometry.h"

// ============================================================================
// 阶段 9：CUDA Graphs —— 滤波 → 反投影链的捕获与重放。
//
// stream capture：先在目标流上热身一次（建立 cuFFT plan / 纹理对象 / 三角函数表
// 等缓存——分配类 API 不可在捕获区内执行），随后把 rampFilterGPU +
// backprojectGPU(Texture) 的全部 kernel/异步拷贝捕获为一张图，instantiate 后
// 单次 launch 即重放整条链。
//
// 价值：把 N 次 kernel launch 的 CPU 端提交开销压成 1 次图提交。本项目的链只有
// 5~6 个节点，收益主要在极小配置（kernel 微秒级、launch 开销占比高）时显现。
// ============================================================================

struct GraphReport {
    int nodes = 0;       // 图节点数（kernel/拷贝节点合计）
    int replays = 0;     // 计时用的重放次数
    double manualMs;     // 对照：手动逐次 launch 整条链的平均单次耗时
    double graphMs;      // 图重放的平均单次耗时
};

// d_sino 输入正弦图；d_recon 输出重建图；texture=true 用纹理版反投影。
// 结果写入 d_recon（链会真实执行一次以上），报告手动 vs 图的单次耗时。
GraphReport reconstructFBPGraph(const float* d_sino, const Geometry& g, FilterWindow win,
                                bool texture, float* d_recon, int replays);

// 释放缓存的 cudaGraphExec（按键缓存：几何/窗口/版本/源指针变化时自动重建）
void graphReleaseCache();
