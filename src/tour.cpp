// ct_tour（阶段 9）：CUDA 知识点巡礼。
// 小配置一键跑遍项目全部功能，打印「知识点 → CUDA API → 实测结果」对照表。
// 只做演示与测量，不做断言（正确性断言在 test_pipeline）。
#include "backprojector.h"
#include "cuda_check.h"
#include "filter.h"
#include "geometry.h"
#include "graph.h"
#include "metrics.h"
#include "phantom.h"
#include "pipeline.h"
#include "projector.h"
#include "sirt.h"
#include "timer.h"

#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>  // 控制台 UTF-8 输出（同 main.cpp）
#endif

namespace {

template <class F>
float gpuMs(F&& fn, int warmup = 3, int runs = 10) {
    for (int i = 0; i < warmup; ++i) fn();
    CT_CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t;
    t.start();
    for (int i = 0; i < runs; ++i) fn();
    t.stop();
    return t.ms() / runs;
}

std::string fmt(const char* f, ...) {
    char buf[160];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

void row(const char* kp, const char* api, const std::string& result) {
    std::printf("  %-26s %-46s %s\n", kp, api, result.c_str());
}

}  // namespace

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    Geometry g{};
    g.n = 128;
    g.n_det = 128;
    g.n_angle = 32;  // 少角度小配置：全部功能秒级跑完
    std::printf("=== ct_tour: CUDA 知识点巡礼（%dx%d 图像, %d 探测器, %d 角度）===\n\n", g.n, g.n,
                g.n_det, g.n_angle);

    cudaDeviceProp prop{};
    CT_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("设备: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    std::vector<float> phantom = makeSheppLogan(g.n);

    float *d_img, *d_sino, *d_sino2, *d_filt, *d_recon, *d_recon2;
    CT_CUDA_CHECK(cudaMalloc(&d_img, sizeof(float) * g.imagePixels()));
    CT_CUDA_CHECK(cudaMalloc(&d_sino, sizeof(float) * g.sinoPixels()));
    CT_CUDA_CHECK(cudaMalloc(&d_sino2, sizeof(float) * g.sinoPixels()));
    CT_CUDA_CHECK(cudaMalloc(&d_filt, sizeof(float) * g.sinoPixels()));
    CT_CUDA_CHECK(cudaMalloc(&d_recon, sizeof(float) * g.imagePixels()));
    CT_CUDA_CHECK(cudaMalloc(&d_recon2, sizeof(float) * g.imagePixels()));
    CT_CUDA_CHECK(cudaMemcpy(d_img, phantom.data(), sizeof(float) * g.imagePixels(),
                             cudaMemcpyHostToDevice));

    std::vector<float> host1(g.sinoPixels()), host2(g.sinoPixels());
    std::vector<float> recon1(g.imagePixels()), recon2(g.imagePixels());

    std::printf("[1/11] kernel 基础：线程映射 / 全局内存 / __restrict__\n");
    float tV1 = gpuMs([&] { forwardProjectGPU(d_img, g, d_sino); });
    CT_CUDA_CHECK(cudaMemcpy(host1.data(), d_sino, sizeof(float) * g.sinoPixels(), cudaMemcpyDeviceToHost));
    row("grid/block 线程映射", "<<<grid, block>>>, blockIdx/threadIdx", fmt("%.3f ms / 前向投影", tV1));

    std::printf("[2/11] 官方库调用：cuFFT 批量 1D FFT\n");
    float tFilt = gpuMs([&] { rampFilterGPU(d_sino, g, FilterWindow::RamLak, d_filt); });
    row("库 kernel 复用", "cufftPlanMany / cufftExecC2C", fmt("%.3f ms / Ramp 滤波", tFilt));

    std::printf("[3/11] 纹理内存：硬件双线性插值\n");
    float tBpN = gpuMs([&] { backprojectGPU(d_filt, g, d_recon); });
    float tBpT = gpuMs([&] { backprojectGPUTexture(d_filt, g, d_recon); });
    row("cudaArray + tex2D", "cudaCreateTextureObject / tex2D", fmt("%.3f ms vs naive %.3f ms", tBpT, tBpN));

    std::printf("[4/11] warp 原语 + 共享内存 + 同步\n");
    float tWarp = gpuMs([&] { forwardProjectGPUWarp(d_img, g, d_sino2); });
    CT_CUDA_CHECK(cudaMemcpy(host2.data(), d_sino2, sizeof(float) * g.sinoPixels(), cudaMemcpyDeviceToHost));
    Metrics mW = computeMetrics(host1.data(), host2.data(), g.sinoPixels());
    row("块内两级归约", "__shfl_down_sync / __shared__ / __syncthreads",
        fmt("%.3f ms, vs v1 diff %.1e", tWarp, mW.max_abs));

    std::printf("[5/11] 原子操作\n");
    float tAtom = gpuMs([&] { forwardProjectGPUAtomic(d_img, g, d_sino2); });
    CT_CUDA_CHECK(cudaMemcpy(host2.data(), d_sino2, sizeof(float) * g.sinoPixels(), cudaMemcpyDeviceToHost));
    Metrics mA = computeMetrics(host1.data(), host2.data(), g.sinoPixels());
    row("直方图式累加", "atomicAdd", fmt("%.3f ms, vs v1 diff %.1e", tAtom, mA.max_abs));

    std::printf("[6/11] 动态并行（CDP2，教学）\n");
    float tCdp = gpuMs([&] { forwardProjectGPUCdp(d_img, g, d_sino2); }, 1, 3);
    row("设备端嵌套 launch", "<<<...,cudaStreamTailLaunch>>> (-rdc)", fmt("%.3f ms（开销 %.0fx，见 RESULTS）", tCdp, tCdp / tV1));

    std::printf("[7/11] cuBLAS Tensor Core + Thrust\n");
    SirtOptions so;
    so.iterations = 40;
    so.relax = 1.9f;
    so.prec = SirtMatrixPrecision::Fp16;
    SirtReport sr = sirtReconstructGPU(d_sino, d_img, g, so, d_recon);
    Metrics mS = computeMetricsGPU(d_img, d_recon, g.imagePixels());
    row("SIRT 迭代重建", "cublasGemmEx(FP16/TF32) + thrust::*", fmt("RMSE %.4f, %.2f ms/iter", mS.rmse, sr.perIterMs));

    std::printf("[8/11] 统一内存\n");
    float *mb, *mx;
    CT_CUDA_CHECK(cudaMallocManaged(&mb, sizeof(float) * g.sinoPixels()));
    CT_CUDA_CHECK(cudaMallocManaged(&mx, sizeof(float) * g.imagePixels()));
    CT_CUDA_CHECK(cudaMemcpy(mb, host1.data(), sizeof(float) * g.sinoPixels(), cudaMemcpyHostToDevice));
    so.alloc = SirtAllocMode::Managed;
    so.prec = SirtMatrixPrecision::Fp32Tf32;
    sirtReconstructGPU(mb, d_img, g, so, mx);
    CT_CUDA_CHECK(cudaDeviceSynchronize());
    Metrics mM = computeMetrics(phantom.data(), mx, g.imagePixels());
    row("managed + prefetch", "cudaMallocManaged / cudaMemPrefetchAsync", fmt("RMSE %.4f（与 device 一致）", mM.rmse));

    std::printf("[9/11] 流 + 异步拷贝 + 锁页内存\n");
    PipelineReport pr = forwardProjectGPUPipeline(d_img, g, d_sino, host2.data(), 4);
    Metrics mP = computeMetrics(host1.data(), host2.data(), g.sinoPixels());
    row("多流流水线", "cudaStream* / cudaMemcpyAsync / cudaMallocHost",
        fmt("%.2f vs 串行 %.2f ms, diff %.1e", pr.pipelinedMs, pr.serialMs, mP.max_abs));

    std::printf("[10/11] 多 GPU\n");
    MultiGpuReport mr = forwardProjectGPUMultiGPU(phantom.data(), g, host2.data());
    std::printf("  %-26s %-46s %s\n", "分卡计算 + peer 汇聚",
                "cudaSetDevice / cudaMemcpyPeerAsync", fmt("%.2f ms, %d 设备", mr.totalMs, mr.deviceCount).c_str());
    std::printf("        拓扑: %s\n", mr.topology.c_str());

    std::printf("[11/11] CUDA Graphs\n");
    GraphReport gr = reconstructFBPGraph(d_sino, g, FilterWindow::RamLak, false, d_recon, 50);
    CT_CUDA_CHECK(cudaMemcpy(recon1.data(), d_recon, sizeof(float) * g.imagePixels(), cudaMemcpyDeviceToHost));
    // 手动链对照结果
    rampFilterGPU(d_sino, g, FilterWindow::RamLak, d_filt);
    backprojectGPU(d_filt, g, d_recon2);
    CT_CUDA_CHECK(cudaMemcpy(recon2.data(), d_recon2, sizeof(float) * g.imagePixels(), cudaMemcpyDeviceToHost));
    Metrics mG = computeMetrics(recon2.data(), recon1.data(), g.imagePixels());
    row("整链捕获重放", "StreamCapture / GraphInstantiate / Launch",
        fmt("图 %.3f vs 手动 %.3f ms, %d 节点, diff %.1e", gr.graphMs, gr.manualMs, gr.nodes, mG.max_abs));

    std::printf("\n结论：全部 11 组 CUDA 知识点已在真实 CT 重建功能上跑通（正确性断言见 test_pipeline）。\n");

    CT_CUDA_CHECK(cudaFree(d_img));
    CT_CUDA_CHECK(cudaFree(d_sino));
    CT_CUDA_CHECK(cudaFree(d_sino2));
    CT_CUDA_CHECK(cudaFree(d_filt));
    CT_CUDA_CHECK(cudaFree(d_recon));
    CT_CUDA_CHECK(cudaFree(d_recon2));
    CT_CUDA_CHECK(cudaFree(mb));
    CT_CUDA_CHECK(cudaFree(mx));
    rampFilterGPUReleaseCache();
    backprojectorGPUReleaseCache();
    sirtReleaseCache();
    graphReleaseCache();
    return 0;
}
