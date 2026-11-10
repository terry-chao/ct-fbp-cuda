// GPU 反投影（阶段 3/4）：两个版本，同一线程映射（1 线程 = 1 输出像素，16x16 block）。
//   naive  ：全局内存 + 手写线性插值
//   texture：正弦图驻留 cudaArray，tex2D 硬件双线性插值（水平方向即探测器方向）
//
// 硬件插值说明：线性过滤的小数权重被量化到 1/512 步长，对 n_det <= 512 的配置
// 引入的误差远小于重建本身的艺术差，换来的是免插值指令 + 缓存友好的访问路径。
//
// 三角函数表与 cudaArray/texture object 均静态缓存（按 n_angle / 源指针 失效重建）。
#include "backprojector.h"
#include "cuda_check.h"

#include <cmath>
#include <vector>

namespace {

// ---- 静态缓存 --------------------------------------------------------------
struct TrigCache {
    int n_angle = 0;
    float* d_cos = nullptr;
    float* d_sin = nullptr;
};
TrigCache g_trig;

struct TexCache {
    const float* src = nullptr;
    int n_det = 0, n_angle = 0;
    cudaArray_t arr = nullptr;
    cudaTextureObject_t tex = 0;
};
TexCache g_tex;

void ensureTrigTable(const Geometry& g) {
    if (g_trig.n_angle == g.n_angle && g_trig.d_cos) return;
    if (g_trig.d_cos) CT_CUDA_CHECK(cudaFree(g_trig.d_cos));
    if (g_trig.d_sin) CT_CUDA_CHECK(cudaFree(g_trig.d_sin));
    std::vector<float> cosT(g.n_angle), sinT(g.n_angle);
    for (int a = 0; a < g.n_angle; ++a) {
        float th = g.angle(a);
        cosT[a] = std::cos(th);
        sinT[a] = std::sin(th);
    }
    CT_CUDA_CHECK(cudaMalloc(&g_trig.d_cos, sizeof(float) * g.n_angle));
    CT_CUDA_CHECK(cudaMalloc(&g_trig.d_sin, sizeof(float) * g.n_angle));
    CT_CUDA_CHECK(cudaMemcpy(g_trig.d_cos, cosT.data(), sizeof(float) * g.n_angle, cudaMemcpyHostToDevice));
    CT_CUDA_CHECK(cudaMemcpy(g_trig.d_sin, sinT.data(), sizeof(float) * g.n_angle, cudaMemcpyHostToDevice));
    g_trig.n_angle = g.n_angle;
}

void destroyTexCache() {
    if (g_tex.tex) {
        cudaDestroyTextureObject(g_tex.tex);
        g_tex.tex = 0;
    }
    if (g_tex.arr) {
        cudaFreeArray(g_tex.arr);
        g_tex.arr = nullptr;
    }
    g_tex.src = nullptr;
}

// ---- kernels ---------------------------------------------------------------

__global__ void backprojectNaiveKernel(const float* __restrict__ sino, int n, int n_det,
                                       int n_angle, float ds, const float* __restrict__ cosT,
                                       const float* __restrict__ sinT,
                                       float* __restrict__ out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 列
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 行
    if (i >= n || j >= n) return;

    float inv = 2.0f / n;
    float x = -1.0f + (j + 0.5f) * inv;
    float y = 1.0f - (i + 0.5f) * inv;

    float acc = 0.f;
    for (int a = 0; a < n_angle; ++a) {
        float s = x * cosT[a] + y * sinT[a];
        float f = (s + 1.0f) / ds - 0.5f;
        int k0 = (int)floorf(f);
        float w = f - k0;
        const float* row = sino + (size_t)a * n_det;
        float v0 = (k0 >= 0 && k0 < n_det) ? row[k0] : 0.f;
        float v1 = (k0 + 1 >= 0 && k0 + 1 < n_det) ? row[k0 + 1] : 0.f;
        acc += v0 + w * (v1 - v0);  // 手写线性插值
    }
    out[(size_t)i * n + j] = acc * (CT_PI / n_angle);
}

__global__ void backprojectTextureKernel(cudaTextureObject_t tex, int n, int n_angle,
                                         const float* __restrict__ cosT,
                                         const float* __restrict__ sinT,
                                         float* __restrict__ out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 列
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 行
    if (i >= n || j >= n) return;

    float inv = 2.0f / n;
    float x = -1.0f + (j + 0.5f) * inv;
    float y = 1.0f - (i + 0.5f) * inv;

    float acc = 0.f;
    for (int a = 0; a < n_angle; ++a) {
        float s = x * cosT[a] + y * sinT[a];
        float u = (s + 1.0f) * 0.5f;         // 归一化探测器坐标 [0,1)
        float v = ((float)a + 0.5f) / n_angle;
        // u 出界说明射线在探测器范围之外（角落像素），与 CPU 版一致取 0
        acc += (u > 0.f && u < 1.f) ? tex2D<float>(tex, u, v) : 0.f;
    }
    out[(size_t)i * n + j] = acc * (CT_PI / n_angle);
}

}  // namespace

void backprojectGPU(const float* d_sino, const Geometry& g, float* d_out) {
    ensureTrigTable(g);
    dim3 block(16, 16);
    dim3 grid((g.n + 15) / 16, (g.n + 15) / 16);
    backprojectNaiveKernel<<<grid, block>>>(d_sino, g.n, g.n_det, g.n_angle, g.ds(),
                                            g_trig.d_cos, g_trig.d_sin, d_out);
    CT_CUDA_CHECK(cudaGetLastError());
}

void backprojectGPUTexture(const float* d_sino, const Geometry& g, float* d_out) {
    ensureTrigTable(g);

    // cudaArray / texture object 按源指针缓存，数据本体每次 D2D 刷新
    if (g_tex.src != d_sino || g_tex.n_det != g.n_det || g_tex.n_angle != g.n_angle) {
        destroyTexCache();
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        CT_CUDA_CHECK(cudaMallocArray(&g_tex.arr, &desc, g.n_det, g.n_angle));
        struct cudaResourceDesc rd;
        memset(&rd, 0, sizeof(rd));
        rd.resType = cudaResourceTypeArray;
        rd.res.array.array = g_tex.arr;
        struct cudaTextureDesc td;
        memset(&td, 0, sizeof(td));
        td.addressMode[0] = cudaAddressModeClamp;
        td.addressMode[1] = cudaAddressModeClamp;
        td.filterMode = cudaFilterModeLinear;   // 硬件线性插值（CUDA 13 起的枚举名）
        td.readMode = cudaReadModeElementType;
        td.normalizedCoords = 1;
        CT_CUDA_CHECK(cudaCreateTextureObject(&g_tex.tex, &rd, &td, nullptr));
        g_tex.src = d_sino;
        g_tex.n_det = g.n_det;
        g_tex.n_angle = g.n_angle;
    }
    CT_CUDA_CHECK(cudaMemcpy2DToArray(g_tex.arr, 0, 0, d_sino, sizeof(float) * g.n_det,
                                      sizeof(float) * g.n_det, g.n_angle,
                                      cudaMemcpyDeviceToDevice));

    dim3 block(16, 16);
    dim3 grid((g.n + 15) / 16, (g.n + 15) / 16);
    backprojectTextureKernel<<<grid, block>>>(g_tex.tex, g.n, g.n_angle, g_trig.d_cos,
                                              g_trig.d_sin, d_out);
    CT_CUDA_CHECK(cudaGetLastError());
}

void backprojectorGPUReleaseCache() {
    destroyTexCache();
    if (g_trig.d_cos) {
        cudaFree(g_trig.d_cos);
        g_trig.d_cos = nullptr;
    }
    if (g_trig.d_sin) {
        cudaFree(g_trig.d_sin);
        g_trig.d_sin = nullptr;
    }
    g_trig.n_angle = 0;
}
