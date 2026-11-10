// 阶段 7：动态并行（CDP2）版前向投影 —— 教学演示。
// 父 kernel 每线程负责一个角度，在设备端 launch 子 kernel 完成该角度的全部射线
// （cudaStreamTailLaunch：子 kernel 在父 grid 全部结束时自动开始，无需显式同步）。
//
// 动态并行的适用场景是"网格形状运行时才可知"的嵌套任务；本例的形状静态已知，
// 因此预期不快于 v1（父 launch 开销 + tail-launch 串行屏障）——价值在于 API 用法
// 与开销的实测认知，结论记录于 docs/results/RESULTS.md。
// 编译要求：-rdc=true（CMake 中对本文件设 CUDA_RELOCATABLE_DEVICE_CODE）。
#include "projector.h"
#include "cuda_check.h"
#include "cuda_utils.cuh"

#include <cmath>

namespace {

// 子 kernel：与 v1 forwardProjectKernel 单角度部分逐字一致（角度 a 由父线程传入）
__global__ void forwardProjectAngleKernel(const float* __restrict__ img, int n, int n_det,
                                          int n_angle, int a, float ds, float dt,
                                          float* __restrict__ sino) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_det) return;

    float th = a * (CT_PI / n_angle);
    float c, s;
    sincosf(th, &s, &c);
    float sk = -1.0f + (k + 0.5f) * ds;
    float tmax = sqrtf(fmaxf(0.f, 2.f - sk * sk));

    float sum = 0.f;
    for (float t = -tmax; t <= tmax; t += dt) {
        sum += bilinearSample(img, n, sk * c - t * s, sk * s + t * c);
    }
    sino[(size_t)a * n_det + k] = sum * dt;
}

// 父 kernel：1 线程 = 1 角度，按 n_det 决定子网格形状（设备端动态决定 launch 参数）
__global__ void forwardProjectParentKernel(const float* __restrict__ img, int n, int n_det,
                                           int n_angle, float ds, float dt,
                                           float* __restrict__ sino) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_angle) return;
    dim3 block(256);
    dim3 grid((n_det + block.x - 1) / block.x);
    forwardProjectAngleKernel<<<grid, block, 0, cudaStreamTailLaunch>>>(img, n, n_det, n_angle, a,
                                                                       ds, dt, sino);
}

}  // namespace

void forwardProjectGPUCdp(const float* d_img, const Geometry& g, float* d_sino) {
    const float dt = (g.dx() < g.ds()) ? g.dx() : g.ds();
    dim3 block(64);
    dim3 grid((g.n_angle + block.x - 1) / block.x);
    forwardProjectParentKernel<<<grid, block>>>(d_img, g.n, g.n_det, g.n_angle, g.ds(), dt,
                                                d_sino);
    CT_CUDA_CHECK(cudaGetLastError());
    CT_CUDA_CHECK(cudaDeviceSynchronize());  // tail-launch 的子 kernel 在此汇合
}
