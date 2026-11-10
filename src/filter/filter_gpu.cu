// GPU Ramp 滤波（阶段 2）：cuFFT 批量 1D C2C FFT。
// 流水线：补零 pad → FFT → 乘 |ν|·窗/M → IFFT → 裁回 n_det。
// cuFFT plan / 频响 / 工作缓冲按 (M, n_det, n_angle, win) 缓存，重复调用零额外开销。
#include "ct/filter/filter.h"
#include "core/cuda_check.h"

#include <vector>

namespace {

struct FilterCache {
    bool valid = false;
    int M = 0, n_det = 0, n_angle = 0;
    int win = -1;
    cufftHandle plan = 0;
    float2* d_buf = nullptr;  // n_angle x M 复数工作区
    float* d_H = nullptr;     // M 点频响
};

FilterCache g_cache;

__global__ void padKernel(const float* __restrict__ sino, int n_det, int n_angle, int M,
                          float2* __restrict__ buf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_angle * M) return;
    int row = idx / M, m = idx % M;
    float v = (m < n_det) ? sino[(size_t)row * n_det + m] : 0.f;
    buf[idx] = make_float2(v, 0.f);
}

__global__ void applyFilterKernel(float2* __restrict__ buf, const float* __restrict__ H,
                                  int M, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float h = H[idx % M];
    buf[idx] = make_float2(buf[idx].x * h, buf[idx].y * h);
}

__global__ void cropKernel(const float2* __restrict__ buf, int n_det, int n_angle, int M,
                           float* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_angle * n_det) return;
    int row = idx / n_det, k = idx % n_det;
    out[idx] = buf[(size_t)row * M + k].x;  // 取实部
}

}  // namespace

void rampFilterGPU(const float* d_sino, const Geometry& g, FilterWindow win, float* d_out,
                   cudaStream_t stream) {
    const int M = paddedLength(g.n_det);
    const int winId = (int)win;

    // 缓存失效则重建 plan / 频响 / 工作区
    if (!g_cache.valid || g_cache.M != M || g_cache.n_det != g.n_det ||
        g_cache.n_angle != g.n_angle || g_cache.win != winId) {
        if (g_cache.plan) CT_CUFFT_CHECK(cufftDestroy(g_cache.plan));
        if (g_cache.d_buf) CT_CUDA_CHECK(cudaFree(g_cache.d_buf));
        if (g_cache.d_H) CT_CUDA_CHECK(cudaFree(g_cache.d_H));

        CT_CUDA_CHECK(cudaMalloc(&g_cache.d_buf, sizeof(float2) * (size_t)g.n_angle * M));
        // 频响在主机按与 CPU 相同的公式构造
        std::vector<float> H(M);
        for (int m = 0; m < M; ++m) {
            int wrapped = (m <= M / 2) ? m : m - M;
            float nu = fabsf((float)wrapped / ((float)M * g.ds()));
            float w = 1.f;
            if (win == FilterWindow::Hamming) {
                w = 0.54f + 0.46f * cosf(2.f * CT_PI * m / M);
            }
            H[m] = nu * w / M;
        }
        CT_CUDA_CHECK(cudaMalloc(&g_cache.d_H, sizeof(float) * M));
        CT_CUDA_CHECK(cudaMemcpy(g_cache.d_H, H.data(), sizeof(float) * M, cudaMemcpyHostToDevice));

        int n[1] = {M};
        int inembed[1] = {M}, istride[1] = {1}, idist = M;
        CT_CUFFT_CHECK(cufftPlanMany(&g_cache.plan, 1, n, inembed, istride[0], idist,
                                     inembed, istride[0], idist, CUFFT_C2C, g.n_angle));

        g_cache.valid = true;
        g_cache.M = M;
        g_cache.n_det = g.n_det;
        g_cache.n_angle = g.n_angle;
        g_cache.win = winId;
    }

    const int totalBuf = g.n_angle * M;
    const int totalOut = g.n_angle * g.n_det;
    // cuFFT plan 为共享缓存，每次调用把执行绑定到目标流（含 kernel 全部走该流）
    CT_CUFFT_CHECK(cufftSetStream(g_cache.plan, stream));
    padKernel<<<(totalBuf + 255) / 256, 256, 0, stream>>>(d_sino, g.n_det, g.n_angle, M,
                                                          g_cache.d_buf);
    CT_CUDA_CHECK(cudaGetLastError());
    CT_CUFFT_CHECK(cufftExecC2C(g_cache.plan, g_cache.d_buf, g_cache.d_buf, CUFFT_FORWARD));
    applyFilterKernel<<<(totalBuf + 255) / 256, 256, 0, stream>>>(g_cache.d_buf, g_cache.d_H, M,
                                                                  totalBuf);
    CT_CUDA_CHECK(cudaGetLastError());
    CT_CUFFT_CHECK(cufftExecC2C(g_cache.plan, g_cache.d_buf, g_cache.d_buf, CUFFT_INVERSE));
    cropKernel<<<(totalOut + 255) / 256, 256, 0, stream>>>(g_cache.d_buf, g.n_det, g.n_angle, M,
                                                           d_out);
    CT_CUDA_CHECK(cudaGetLastError());
}

void rampFilterGPUReleaseCache() {
    if (g_cache.plan) {
        cufftDestroy(g_cache.plan);
        g_cache.plan = 0;
    }
    if (g_cache.d_buf) {
        cudaFree(g_cache.d_buf);
        g_cache.d_buf = nullptr;
    }
    if (g_cache.d_H) {
        cudaFree(g_cache.d_H);
        g_cache.d_H = nullptr;
    }
    g_cache.valid = false;
}
