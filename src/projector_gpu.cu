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

void forwardProjectGPU(const float* d_img, const Geometry& g, float* d_sino) {
    const float dt = (g.dx() < g.ds()) ? g.dx() : g.ds();
    dim3 block(256);
    dim3 grid((g.n_det + 255) / 256, g.n_angle);
    forwardProjectKernel<<<grid, block>>>(d_img, g.n, g.n_det, g.n_angle, g.ds(), dt, d_sino);
    CT_CUDA_CHECK(cudaGetLastError());
}
