#pragma once

#include "geometry.h"

#include <cstddef>
#include <vector>

// ============================================================================
// SIRT 迭代重建（阶段 6）—— cuBLAS(Tensor Core) + Thrust + 统一内存
//
// 系统矩阵观点：正弦图 b ≈ A·x
//   A：M×K 稠密矩阵，M = n_angle·n_det（射线数），K = n·n（像素数）
//   第 m 行 = 第 m 条射线穿过图像的双线性权重（与前向投影完全相同的采样公式
//   构出：dt 步进 + 双线性散射），因此 A 与 forwardProjectGPU 自洽。
//
// 迭代（全部在设备端完成，每步零 D2H 同步）：
//   p  = A·x                    cuBLAS gemv（row-major 按列主序视角）
//   r  = b − p                   Thrust transform
//   r̂  = r / R                  R = A·1（行和，预计算）
//   c  = Aᵀ·r̂                   cuBLAS gemv（转置）
//   x += λ · c / C              C = Aᵀ·1（列和，预计算），Thrust for_each
//
// Tensor Core 两种模式（cublasGemmEx，gemv 以 n=1 的 gemm 实现）：
//   Fp32Tf32：A 为 fp32，computeType = CUBLAS_COMPUTE_32F_FAST_TF32（TF32 张量核心）
//   Fp16    ：A 为 __half（显存减半），B/C 仍 fp32，computeType = 32F（FP16 张量核心）
//
// 内存模式：
//   Device  ：显式 cudaMalloc/cudaMemcpy（默认，与项目其余部分一致）
//   Managed ：cudaMallocManaged + cudaMemPrefetchAsync（统一内存，简化主机/设备共享）
// ============================================================================

enum class SirtMatrixPrecision {
    Fp32Tf32,  // fp32 矩阵 + TF32 Tensor Core
    Fp16,      // fp16 矩阵 + FP16 Tensor Core
};

enum class SirtAllocMode {
    Device,   // 显式设备内存
    Managed,  // 统一内存
};

struct SirtOptions {
    int iterations = 60;                              // 迭代次数
    float relax = 1.0f;                               // 松弛因子 λ
    SirtMatrixPrecision prec = SirtMatrixPrecision::Fp16;
    SirtAllocMode alloc = SirtAllocMode::Device;
    int recordEvery = 10;                             // 收敛曲线采样间隔（迭代数）
};

struct SirtReport {
    int iterations = 0;
    double buildMs = 0.0;        // 系统矩阵构建（含归一化向量）耗时
    double perIterMs = 0.0;      // 平均单次迭代耗时（CUDA Events）
    double finalResidual = 0.0;  // 末次 ||b − A·x||₂
    size_t matrixBytes = 0;      // 系统矩阵峰值显存（含构建临时缓冲）
    std::vector<int> iterIdx;          // 收敛曲线采样点的迭代序号
    std::vector<double> iterRmse;      // 收敛曲线：x vs 体模 RMSE（需传入 d_phantom）
    std::vector<double> iterResidual;  // 收敛曲线：||r||₂
};

// 系统矩阵峰值显存估算（字节）：fp16 模式 = M·K·(4+2)（fp32 临时 + fp16 存储），
// fp32 模式 = M·K·4。调用方据此做几何限制。
size_t sirtMatrixBytes(const Geometry& g, SirtMatrixPrecision prec);

// GPU SIRT 重建。
//   d_b       设备（或 managed）正弦图，M 元素 —— 由 forwardProjectGPU 产出
//   d_phantom 设备体模（K 元素），仅用于收敛曲线记录；传 nullptr 跳过 RMSE 记录
//   d_x       输出图像（K 元素），调用方分配，入口处被清零
SirtReport sirtReconstructGPU(const float* d_b, const float* d_phantom, const Geometry& g,
                              const SirtOptions& o, float* d_x);

// 释放 SIRT 模块静态缓存（系统矩阵/cublas 句柄等，按键缓存几何变化时自动失效）
void sirtReleaseCache();
