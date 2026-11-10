// SIRT 迭代重建（阶段 6）：cuBLAS Tensor Core + Thrust + 统一内存。
//
// 系统矩阵构建与前向投影自洽：buildMatrixFp32Kernel 与 forwardProjectKernel 使用
// 完全相同的射线参数化与 dt 步进，只是把"采样值求和"换成"双线性权重散射"。
// 稀疏角度（如 32 个角度）下 FBP 条纹伪影严重，SIRT 显著更优 —— 本模块的实验卖点。
//
// cuBLAS 列主序约定下的 row-major 技巧：A 按 row-major M×K 存储，视为列主序即
// K×M 矩阵 B（B(k,m)=A(m,k)）。于是：
//   p = A·x  (M 维) = Bᵀ·x → GemmEx(OP_T, m=M, k=K, lda=K)
//   c = Aᵀ·r  (K 维) = B·r  → GemmEx(OP_N, m=K, k=M, lda=K)
#include "ct/recon/sirt.h"
#include "core/cuda_check.h"

#include "ct/core/timer.h"
#include "ct/metrics/metrics.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

#define CT_CUBLAS_CHECK(call)                                                  \
    do {                                                                       \
        cublasStatus_t s_ = (call);                                            \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d  %s\n", (int)s_,    \
                         __FILE__, __LINE__, #call);                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---- kernel -----------------------------------------------------------------

// 1 线程 = 1 射线（行 m = a*n_det + k）：沿 t 步进，把双线性权重·dt 散射进 A 的
// 第 m 行。每行只被一个线程独占写入，无需原子操作。
__global__ void buildMatrixFp32Kernel(int n, int n_det, int n_angle, float ds, float dt,
                                      float* __restrict__ A, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_angle * n_det) return;
    int a = m / n_det, k = m % n_det;

    float th = a * (CT_PI / n_angle);
    float c, s;
    sincosf(th, &s, &c);
    float sk = -1.0f + (k + 0.5f) * ds;
    float tmax = sqrtf(fmaxf(0.f, 2.f - sk * sk));  // 同前向投影的外接圆截断

    float* row = A + (size_t)m * K;
    float ddx = 2.0f / n;
    auto scatter = [&](int j, int i, float w) {
        if (j >= 0 && j < n && i >= 0 && i < n) row[(size_t)i * n + j] += w;
    };
    for (float t = -tmax; t <= tmax; t += dt) {
        float x = sk * c - t * s, y = sk * s + t * c;
        float fx = (x + 1.0f) / ddx - 0.5f, fy = (1.0f - y) / ddx - 0.5f;
        int j0 = (int)floorf(fx), i0 = (int)floorf(fy);
        float wx = fx - j0, wy = fy - i0;
        scatter(j0, i0, (1 - wx) * (1 - wy) * dt);
        scatter(j0 + 1, i0, wx * (1 - wy) * dt);
        scatter(j0, i0 + 1, (1 - wx) * wy * dt);
        scatter(j0 + 1, i0 + 1, wx * wy * dt);
    }
}

// elementwise：fp32 → fp16（矩阵构建后压缩存储，显存减半）
__global__ void convertToHalfKernel(const float* __restrict__ in, __half* __restrict__ out,
                                    size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// ---- Thrust 仿函数（迭代内的 elementwise 步骤，全部设备端完成）---------------

struct ClampEps {
    __device__ float operator()(float v) const { return fmaxf(v, 1e-6f); }
};

struct FloatToHalf {
    __device__ __half operator()(float v) const { return __float2half(v); }
};

struct HalfToFloat {
    __device__ float operator()(__half v) const { return __half2float(v); }
};

// x += λ·c/C（zip 迭代器三元组：可写 x 引用 + 只读 c/C）
struct AxpyDivFunctor {
    float lam;
    __device__ void operator()(thrust::tuple<float&, const float&, const float&> t) const {
        thrust::get<0>(t) += lam * thrust::get<1>(t) / thrust::get<2>(t);
    }
};

// ---- 静态缓存（沿用项目 XxxCache 惯例：按键失效重建）------------------------

struct SirtCache {
    bool valid = false;
    int kn = 0, kdet = 0, kangle = 0, kprec = 0;
    bool managed = false;
    float* A32 = nullptr;               // fp32 矩阵；fp16 模式下构建后即释放
    __half* A16 = nullptr;              // fp16 矩阵
    float *R = nullptr, *C = nullptr;   // 行和/列和（clamp 后的归一化向量）
    float *p = nullptr, *r = nullptr, *rn = nullptr, *c = nullptr;  // 迭代工作区
    float *onesM = nullptr, *onesK = nullptr;                        // 求 R/C 用的全 1 向量
    __half *hx = nullptr, *hp = nullptr;  // fp16 模式的 gemm 输入/输出 staging
};
SirtCache g_cache;
cublasHandle_t g_handle = nullptr;

// 统一内存预取到设备 0（CUDA 13 签名需 cudaMemLocation）。best-effort：
// WDDM（Windows 默认驱动模型）不支持 prefetch，会返回 cudaErrorInvalidDevice——
// 首次失败后禁用后续预取并用 cudaGetLastError 清掉挂起的错误状态，
// 避免污染后续 CT_CUDA_CHECK(cudaGetLastError()) 检查。
bool g_prefetchOk = true;
void prefetchToDevice(const void* p, size_t bytes) {
    if (!g_prefetchOk || p == nullptr) return;
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id = 0;
    if (cudaMemPrefetchAsync(p, bytes, loc, 0, 0) != cudaSuccess) {
        g_prefetchOk = false;
        (void)cudaGetLastError();
    }
}

float* allocFloat(size_t n, bool managed) {
    float* p = nullptr;
    if (managed) CT_CUDA_CHECK(cudaMallocManaged(&p, sizeof(float) * n));
    else CT_CUDA_CHECK(cudaMalloc(&p, sizeof(float) * n));
    return p;
}

// 前向声明（ensureCache 内求 R/C 时使用，定义见下）
void gemvA(const float* x, float* p, int M, int K, SirtMatrixPrecision prec);
void gemvAT(const float* r, float* c, int M, int K, SirtMatrixPrecision prec);

// 构建（或复用缓存）系统矩阵与归一化向量。返回是否实际重建，重建耗时写入 buildMs。
bool ensureCache(const Geometry& g, const SirtOptions& o, double& buildMs) {
    const int prec = (int)o.prec;
    if (g_cache.valid && g_cache.kn == g.n && g_cache.kdet == g.n_det &&
        g_cache.kangle == g.n_angle && g_cache.kprec == prec &&
        g_cache.managed == (o.alloc == SirtAllocMode::Managed)) {
        buildMs = 0.0;
        return false;
    }

    // 失效旧缓存
    if (g_cache.A32) CT_CUDA_CHECK(cudaFree(g_cache.A32));
    if (g_cache.A16) CT_CUDA_CHECK(cudaFree(g_cache.A16));
    for (float* p : {g_cache.R, g_cache.C, g_cache.p, g_cache.r, g_cache.rn, g_cache.c,
                     g_cache.onesM, g_cache.onesK})
        if (p) CT_CUDA_CHECK(cudaFree(p));
    if (g_cache.hx) CT_CUDA_CHECK(cudaFree(g_cache.hx));
    if (g_cache.hp) CT_CUDA_CHECK(cudaFree(g_cache.hp));

    const bool managed = (o.alloc == SirtAllocMode::Managed);
    const int M = g.n_angle * g.n_det, K = g.n * g.n;
    const size_t MK = (size_t)M * K;
    const float dt = (g.dx() < g.ds()) ? g.dx() : g.ds();

    if (!g_handle) CT_CUBLAS_CHECK(cublasCreate(&g_handle));

    g_cache = SirtCache{};
    g_cache.kn = g.n;
    g_cache.kdet = g.n_det;
    g_cache.kangle = g.n_angle;
    g_cache.kprec = prec;
    g_cache.managed = managed;

    CpuTimer timer;
    timer.tick();

    // 1) fp32 稠密矩阵：清零 + 射线散射
    g_cache.A32 = allocFloat(MK, managed);
    CT_CUDA_CHECK(cudaMemset(g_cache.A32, 0, sizeof(float) * MK));
    buildMatrixFp32Kernel<<<(M + 255) / 256, 256>>>(g.n, g.n_det, g.n_angle, g.ds(), dt,
                                                    g_cache.A32, K);
    CT_CUDA_CHECK(cudaGetLastError());

    // 2) fp16 模式：压缩存储后释放 fp32 临时缓冲
    //    （A16 同样遵循内存模式：managed 下与 A/R/C 一致，后续 prefetch 才合法有效）
    if (o.prec == SirtMatrixPrecision::Fp16) {
        if (managed) CT_CUDA_CHECK(cudaMallocManaged((void**)&g_cache.A16, sizeof(__half) * MK));
        else CT_CUDA_CHECK(cudaMalloc((void**)&g_cache.A16, sizeof(__half) * MK));
        convertToHalfKernel<<<(int)((MK + 255) / 256), 256>>>(g_cache.A32, g_cache.A16, MK);
        CT_CUDA_CHECK(cudaGetLastError());
        CT_CUDA_CHECK(cudaFree(g_cache.A32));
        g_cache.A32 = nullptr;
    }

    // 3) 工作区与全 1 向量
    g_cache.R = allocFloat(M, managed);
    g_cache.C = allocFloat(K, managed);
    g_cache.p = allocFloat(M, managed);
    g_cache.r = allocFloat(M, managed);
    g_cache.rn = allocFloat(M, managed);
    g_cache.c = allocFloat(K, managed);
    g_cache.onesM = allocFloat(M, managed);
    g_cache.onesK = allocFloat(K, managed);
    thrust::fill(thrust::device, g_cache.onesM, g_cache.onesM + M, 1.0f);
    thrust::fill(thrust::device, g_cache.onesK, g_cache.onesK + K, 1.0f);
    if (o.prec == SirtMatrixPrecision::Fp16) {
        CT_CUDA_CHECK(cudaMalloc(&g_cache.hx, sizeof(__half) * K));
        CT_CUDA_CHECK(cudaMalloc(&g_cache.hp, sizeof(__half) * M));
    }

    // 4) 归一化向量：复用 gemv —— R = A·1（行和），C = Aᵀ·1（列和），再 clamp 防除零
    gemvA(g_cache.onesK, g_cache.R, M, K, o.prec);
    gemvAT(g_cache.onesM, g_cache.C, M, K, o.prec);
    thrust::transform(thrust::device, g_cache.R, g_cache.R + M, g_cache.R, ClampEps());
    thrust::transform(thrust::device, g_cache.C, g_cache.C + K, g_cache.C, ClampEps());

    CT_CUDA_CHECK(cudaDeviceSynchronize());

    // 5) 统一内存：把大块数据预取到设备（WDDM 下为 best-effort，教学演示）
    if (managed) {
        const void* A = (o.prec == SirtMatrixPrecision::Fp16) ? (const void*)g_cache.A16
                                                             : (const void*)g_cache.A32;
        size_t aBytes = (o.prec == SirtMatrixPrecision::Fp16) ? sizeof(__half) * MK
                                                             : sizeof(float) * MK;
        prefetchToDevice(A, aBytes);
        prefetchToDevice(g_cache.R, sizeof(float) * M);
        prefetchToDevice(g_cache.C, sizeof(float) * K);
    }

    g_cache.valid = true;
    buildMs = timer.tockMs();
    return true;
}

// p = A·x（M 维）。row-major M×K 的 A 按列主序视角为 K×M，故取 OP_T。
// fp32 模式走 FAST_TF32（TF32 Tensor Core）；fp16 模式 A/B/C 全 16F + 32F 累加
// （文档明确支持的组合），x/p 经 staging 缓冲做显式精度转换。
void gemvA(const float* x, float* p, int M, int K, SirtMatrixPrecision prec) {
    const float alpha = 1.0f, beta = 0.0f;
    if (prec == SirtMatrixPrecision::Fp32Tf32) {
        CT_CUBLAS_CHECK(cublasGemmEx(g_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, K, &alpha,
                                     g_cache.A32, CUDA_R_32F, K, x, CUDA_R_32F, K, &beta, p,
                                     CUDA_R_32F, M, CUBLAS_COMPUTE_32F_FAST_TF32,
                                     CUBLAS_GEMM_DEFAULT));
    } else {
        thrust::transform(thrust::device, x, x + K, g_cache.hx, FloatToHalf());
        CT_CUBLAS_CHECK(cublasGemmEx(g_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, K, &alpha,
                                     g_cache.A16, CUDA_R_16F, K, g_cache.hx, CUDA_R_16F, K,
                                     &beta, g_cache.hp, CUDA_R_16F, M, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT));
        thrust::transform(thrust::device, g_cache.hp, g_cache.hp + M, p, HalfToFloat());
    }
}

// c = Aᵀ·r（K 维）：列主序视角即 OP_N 的 B·r。
void gemvAT(const float* r, float* c, int M, int K, SirtMatrixPrecision prec) {
    const float alpha = 1.0f, beta = 0.0f;
    if (prec == SirtMatrixPrecision::Fp32Tf32) {
        CT_CUBLAS_CHECK(cublasGemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, K, 1, M, &alpha,
                                     g_cache.A32, CUDA_R_32F, K, r, CUDA_R_32F, M, &beta, c,
                                     CUDA_R_32F, K, CUBLAS_COMPUTE_32F_FAST_TF32,
                                     CUBLAS_GEMM_DEFAULT));
    } else {
        // 复用 hx 当输入 staging（此刻 x 不在用）、hp 当输出 staging
        thrust::transform(thrust::device, r, r + M, g_cache.hp, FloatToHalf());
        CT_CUBLAS_CHECK(cublasGemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, K, 1, M, &alpha,
                                     g_cache.A16, CUDA_R_16F, K, g_cache.hp, CUDA_R_16F, M,
                                     &beta, g_cache.hx, CUDA_R_16F, K, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT));
        thrust::transform(thrust::device, g_cache.hx, g_cache.hx + K, c, HalfToFloat());
    }
}

}  // namespace

size_t sirtMatrixBytes(const Geometry& g, SirtMatrixPrecision prec) {
    const size_t MK = (size_t)g.n_angle * g.n_det * g.n * g.n;
    return prec == SirtMatrixPrecision::Fp16 ? MK * 6 : MK * 4;
}

SirtReport sirtReconstructGPU(const float* d_b, const float* d_phantom, const Geometry& g,
                              const SirtOptions& o, float* d_x) {
    SirtReport rep;
    rep.iterations = o.iterations;
    rep.matrixBytes = sirtMatrixBytes(g, o.prec);
    const int M = g.n_angle * g.n_det, K = g.n * g.n;

    ensureCache(g, o, rep.buildMs);
    if (o.alloc == SirtAllocMode::Managed) prefetchToDevice(d_x, sizeof(float) * K);

    // x₀ = 0
    CT_CUDA_CHECK(cudaMemsetAsync(d_x, 0, sizeof(float) * K));

    GpuTimer timer;
    timer.start();
    double residual = 0.0;
    for (int it = 1; it <= o.iterations; ++it) {
        gemvA(d_x, g_cache.p, M, K, o.prec);                                // p = A·x
        thrust::transform(thrust::device, d_b, d_b + M, g_cache.p, g_cache.r,
                          thrust::minus<float>());                          // r = b − p
        residual = std::sqrt(
            thrust::inner_product(thrust::device, g_cache.r, g_cache.r + M, g_cache.r, 0.0));
        thrust::transform(thrust::device, g_cache.r, g_cache.r + M, g_cache.R, g_cache.rn,
                          thrust::divides<float>());                       // r̂ = r / R
        gemvAT(g_cache.rn, g_cache.c, M, K, o.prec);                        // c = Aᵀ·r̂
        AxpyDivFunctor upd{o.relax};
        thrust::for_each(thrust::device,
                         thrust::make_zip_iterator(thrust::make_tuple(d_x, g_cache.c, g_cache.C)),
                         thrust::make_zip_iterator(thrust::make_tuple(d_x + K, g_cache.c + K,
                                                                      g_cache.C + K)),
                         upd);                                              // x += λ·c/C
        const bool record = (it % o.recordEvery == 0) || (it == o.iterations);
        if (record) {
            rep.iterIdx.push_back(it);
            rep.iterResidual.push_back(residual);
            if (d_phantom) {
                Metrics m = computeMetricsGPU(d_phantom, d_x, (size_t)K);
                rep.iterRmse.push_back(m.rmse);
            }
        }
    }
    timer.stop();
    rep.perIterMs = timer.ms() / o.iterations;
    rep.finalResidual = residual;
    return rep;
}

void sirtReleaseCache() {
    if (g_cache.A32) CT_CUDA_CHECK(cudaFree(g_cache.A32));
    if (g_cache.A16) CT_CUDA_CHECK(cudaFree(g_cache.A16));
    for (float* p : {g_cache.R, g_cache.C, g_cache.p, g_cache.r, g_cache.rn, g_cache.c,
                     g_cache.onesM, g_cache.onesK})
        if (p) CT_CUDA_CHECK(cudaFree(p));
    if (g_cache.hx) CT_CUDA_CHECK(cudaFree(g_cache.hx));
    if (g_cache.hp) CT_CUDA_CHECK(cudaFree(g_cache.hp));
    g_cache = SirtCache{};
    if (g_handle) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
}
