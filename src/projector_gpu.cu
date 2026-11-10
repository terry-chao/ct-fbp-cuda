// GPU 前向投影（阶段 1）：naive kernel，1 线程 = 1 (角度, 探测器单元)。
// 采样公式与 CPU 参考实现逐字一致，保证结果可在容差内对齐。
#include "projector.h"
#include "cuda_check.h"
#include "cuda_utils.cuh"  // bilinearSample（阶段 6 起跨文件共享）

#include <cmath>

namespace {


__global__ void forwardProjectKernel(const float* __restrict__ img, int n, int n_det,
                                     int n_angle, float ds, float dt,
                                     float* __restrict__ sino) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // 探测器单元
    int a = blockIdx.y;                             // 角度
    if (k >= n_det) return;

    float th = a * (CT_PI / n_angle);
    float c, s;
    sincosf(th, &s, &c);
    float sk = -1.0f + (k + 0.5f) * ds;
    float tmax = sqrtf(fmaxf(0.f, 2.f - sk * sk));  // 外接圆截断，同 CPU

    float sum = 0.f;
    for (float t = -tmax; t <= tmax; t += dt) {
        sum += bilinearSample(img, n, sk * c - t * s, sk * s + t * c);
    }
    sino[(size_t)a * n_det + k] = sum * dt;
}

}  // namespace

// ---- 阶段 7：前向投影并行变体 ------------------------------------------------
// 射线积分本质是归约：v1 由单线程串行累加；下面两个变体把"沿 t 求和"并行化，
// 展示 CUDA 两种标准的跨线程求和手段（协作归约 / 原子散射）及其代价对比。
// 样本位置用双精度索引计算（t = -tmax + idx·dt），避免逐次 += 的浮点漂移，
// 与 v1 的一致性容差见 tests/testForwardVariantsMatch。

namespace {

constexpr int kRayBlock = 256;  // v2：1 block = 1 射线，256 线程交错采样

// v2 warp 协作版：线程交错取样本各算部分和 → warp 内 __shfl_down_sync 归约 →
// __shared__ 汇合 + __syncthreads → 首 warp 二级归约 → 线程 0 写回。
__global__ void forwardProjectWarpKernel(const float* __restrict__ img, int n, int n_det,
                                         int n_angle, float ds, float dt,
                                         float* __restrict__ sino) {
    const int m = blockIdx.x;  // 射线编号 = a*n_det + k
    const int a = m / n_det, k = m % n_det;
    const float th = a * (CT_PI / n_angle);
    float c, s;
    sincosf(th, &s, &c);
    const float sk = -1.0f + (k + 0.5f) * ds;
    const float tmax = sqrtf(fmaxf(0.f, 2.f - sk * sk));
    const int steps = (int)floorf(2.f * tmax / dt) + 1;

    float partial = 0.f;
    for (int idx = threadIdx.x; idx < steps; idx += kRayBlock) {
        float t = (float)((double)-tmax + (double)idx * dt);
        partial += bilinearSample(img, n, sk * c - t * s, sk * s + t * c);
    }

    // 两级归约：warp 内 shuffle（免共享内存、免同步），跨 warp 经共享内存汇合
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
    __shared__ float warpSum[kRayBlock / 32];
    if ((threadIdx.x & 31) == 0) warpSum[threadIdx.x >> 5] = partial;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < kRayBlock / 32) ? warpSum[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) sino[m] = v * dt;
    }
}

// v3 原子散射版：1 线程 = 1 (射线, 样本块)，部分和 atomicAdd 直接累加进正弦图。
// 免归约树与同步，但同地址原子操作串行化——直方图式累加的典型模式与代价样本。
constexpr int kChunk = 32;

__global__ void forwardProjectAtomicKernel(const float* __restrict__ img, int n, int n_det,
                                           int n_angle, float ds, float dt, int chunksPerRay,
                                           float* __restrict__ sino) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = id / chunksPerRay, ci = id % chunksPerRay;  // 连续线程覆盖同一射线
    if (m >= n_angle * n_det) return;
    const int a = m / n_det, k = m % n_det;
    const float th = a * (CT_PI / n_angle);
    float c, s;
    sincosf(th, &s, &c);
    const float sk = -1.0f + (k + 0.5f) * ds;
    const float tmax = sqrtf(fmaxf(0.f, 2.f - sk * sk));
    const int steps = (int)floorf(2.f * tmax / dt) + 1;

    float partial = 0.f;
    const int i0 = ci * kChunk, i1 = min(i0 + kChunk, steps);
    for (int idx = i0; idx < i1; ++idx) {
        float t = (float)((double)-tmax + (double)idx * dt);
        partial += bilinearSample(img, n, sk * c - t * s, sk * s + t * c);
    }
    if (partial != 0.f) atomicAdd(&sino[m], partial * dt);
}

}  // namespace

void forwardProjectGPU(const float* d_img, const Geometry& g, float* d_sino) {
    const float dt = (g.dx() < g.ds()) ? g.dx() : g.ds();
    dim3 block(256);
    dim3 grid((g.n_det + 255) / 256, g.n_angle);
    forwardProjectKernel<<<grid, block>>>(d_img, g.n, g.n_det, g.n_angle, g.ds(), dt, d_sino);
    CT_CUDA_CHECK(cudaGetLastError());
}

void forwardProjectGPUWarp(const float* d_img, const Geometry& g, float* d_sino) {
    const float dt = (g.dx() < g.ds()) ? g.dx() : g.ds();
    forwardProjectWarpKernel<<<g.n_angle * g.n_det, kRayBlock>>>(d_img, g.n, g.n_det, g.n_angle,
                                                                g.ds(), dt, d_sino);
    CT_CUDA_CHECK(cudaGetLastError());
}

void forwardProjectGPUAtomic(const float* d_img, const Geometry& g, float* d_sino) {
    const float dt = (g.dx() < g.ds()) ? g.dx() : g.ds();
    // 样本块数按最长射线计（tmax <= sqrt(2)），短射线的多余线程直接空转
    const int maxSteps = (int)floorf(2.f * 1.41421356f / dt) + 1;
    const int chunksPerRay = (maxSteps + kChunk - 1) / kChunk;
    CT_CUDA_CHECK(cudaMemsetAsync(d_sino, 0, sizeof(float) * g.sinoPixels()));
    const size_t total = (size_t)g.n_angle * g.n_det * chunksPerRay;
    dim3 block(256);
    dim3 grid((unsigned)((total + 255) / 256));
    forwardProjectAtomicKernel<<<grid, block>>>(d_img, g.n, g.n_det, g.n_angle, g.ds(), dt,
                                                chunksPerRay, d_sino);
    CT_CUDA_CHECK(cudaGetLastError());
}
