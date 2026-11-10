// 阶段 8：异步流水线与多 GPU 前向投影。
// 知识点：cudaStreamCreate / cudaMemcpyAsync(D2H) / cudaMallocHost（锁页）/
// cudaStreamSynchronize / cudaSetDevice / cudaDeviceCanAccessPeer /
// cudaDeviceEnablePeerAccess / cudaMemcpyPeerAsync。
#include "ct/recon/pipeline.h"
#include "core/cuda_check.h"
#include "ct/core/timer.h"
#include "ct/projector/projector.h"

#include <algorithm>
#include <cstdio>
#include <vector>

PipelineReport forwardProjectGPUPipeline(const float* d_img, const Geometry& g, float* d_sino,
                                         float* host_out, int nStreams) {
    PipelineReport rep;
    const int S = std::max(1, nStreams);
    rep.streams = S;
    const int chunkAngles = (g.n_angle + S - 1) / S;
    rep.chunks = (g.n_angle + chunkAngles - 1) / chunkAngles;
    rep.dataMB = (double)(sizeof(float) * g.sinoPixels()) / 1048576.0;

    std::vector<cudaStream_t> stream(S);
    for (int s = 0; s < S; ++s) CT_CUDA_CHECK(cudaStreamCreate(&stream[s]));

    // 锁页缓冲：DMA 直达，异步拷贝的前提（pageable 内存会退化为同步分段拷贝）。
    // 各分块写入互不相交的行区间，多流共用一块是安全的。
    float* pinned = nullptr;
    CT_CUDA_CHECK(cudaMallocHost(&pinned, sizeof(float) * g.sinoPixels()));

    // ---- 分块流水：chunk c 在 stream[c % S] 上执行（同流内 kernel→拷贝保序，
    //      跨流与其它 chunk 的计算/拷贝重叠）----
    CpuTimer t;
    t.tick();
    int c = 0;
    for (int a0 = 0; a0 < g.n_angle; a0 += chunkAngles, ++c) {
        const int s = c % S;
        const int cnt = std::min(chunkAngles, g.n_angle - a0);
        forwardProjectGPURange(d_img, g, d_sino, a0, cnt, stream[s]);
        CT_CUDA_CHECK(cudaMemcpyAsync(pinned + (size_t)a0 * g.n_det,
                                      d_sino + (size_t)a0 * g.n_det,
                                      sizeof(float) * (size_t)cnt * g.n_det,
                                      cudaMemcpyDeviceToHost, stream[s]));
    }
    for (int s = 0; s < S; ++s) CT_CUDA_CHECK(cudaStreamSynchronize(stream[s]));
    rep.pipelinedMs = t.tockMs();
    std::copy(pinned, pinned + g.sinoPixels(), host_out);

    // ---- 对照：串行形态（全量 kernel → 一次拷贝，同数据量）----
    t.tick();
    forwardProjectGPU(d_img, g, d_sino, stream[0]);
    CT_CUDA_CHECK(cudaMemcpyAsync(pinned, d_sino, sizeof(float) * g.sinoPixels(),
                                  cudaMemcpyDeviceToHost, stream[0]));
    CT_CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    rep.serialMs = t.tockMs();
    rep.gbps = rep.dataMB / (rep.pipelinedMs / 1000.0) / 1024.0;

    CT_CUDA_CHECK(cudaFreeHost(pinned));
    for (int s = 0; s < S; ++s) CT_CUDA_CHECK(cudaStreamDestroy(stream[s]));
    return rep;
}

MultiGpuReport forwardProjectGPUMultiGPU(const float* h_img, const Geometry& g, float* host_out) {
    MultiGpuReport rep;
    int devCount = 0;
    CT_CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 1) {
        std::fprintf(stderr, "错误: 无可用 CUDA 设备\n");
        std::exit(1);
    }
    rep.deviceCount = devCount;

    // 拓扑：各设备名 + 与主卡(0)的 peer 可达性
    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp prop{};
        CT_CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        char buf[160];
        if (i == 0) {
            std::snprintf(buf, sizeof(buf), "主卡 GPU0: %s", prop.name);
        } else {
            int canPeer = 0;
            CT_CUDA_CHECK(cudaDeviceCanAccessPeer(&canPeer, i, 0));
            std::snprintf(buf, sizeof(buf), " | GPU%d: %s (peer-to-0: %s)", i, prop.name,
                          canPeer ? "yes" : "no");
        }
        rep.topology += buf;
    }

    const int chunkAngles = (g.n_angle + devCount - 1) / devCount;

    // 各卡资源：图像/整幅正弦图缓冲 + 独立流（0 号卡的正弦图缓冲即汇聚缓冲 d_sino0）
    std::vector<float*> d_img_i(devCount, nullptr), d_sino_i(devCount, nullptr);
    std::vector<cudaStream_t> stream_i(devCount, nullptr);
    std::vector<bool> peerOn(devCount, false);  // i 号卡与 0 卡的 peer 是否已启用
    for (int i = 0; i < devCount; ++i) {
        CT_CUDA_CHECK(cudaSetDevice(i));
        CT_CUDA_CHECK(cudaStreamCreate(&stream_i[i]));
        CT_CUDA_CHECK(cudaMalloc(&d_img_i[i], sizeof(float) * g.imagePixels()));
        CT_CUDA_CHECK(cudaMalloc(&d_sino_i[i], sizeof(float) * g.sinoPixels()));
        CT_CUDA_CHECK(cudaMemcpyAsync(d_img_i[i], h_img, sizeof(float) * g.imagePixels(),
                                      cudaMemcpyHostToDevice, stream_i[i]));
    }
    float* d_sino0 = d_sino_i[0];

    CpuTimer t;
    t.tick();
    struct Chunk {
        int a0, cnt, dev;
        bool inSino0;  // 结果最终落在 d_sino0（0 卡自身或经 peer 汇聚）
    };
    std::vector<Chunk> chunks;
    int c = 0;
    for (int a0 = 0; a0 < g.n_angle; a0 += chunkAngles, ++c) {
        const int i = c % devCount;
        const int cnt = std::min(chunkAngles, g.n_angle - a0);
        CT_CUDA_CHECK(cudaSetDevice(i));
        forwardProjectGPURange(d_img_i[i], g, d_sino_i[i], a0, cnt, stream_i[i]);

        bool inSino0 = (i == 0);
        if (i != 0) {
            const size_t bytes = sizeof(float) * (size_t)cnt * g.n_det;
            const size_t off = (size_t)a0 * g.n_det;
            int canPeer = 0;
            CT_CUDA_CHECK(cudaDeviceCanAccessPeer(&canPeer, i, 0));
            if (canPeer) {
                if (!peerOn[i]) {
                    cudaError_t e = cudaDeviceEnablePeerAccess(0, 0);  // 当前卡 i 与 0 卡互通
                    peerOn[i] = (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled);
                    if (!peerOn[i]) (void)cudaGetLastError();
                }
                if (peerOn[i]) {
                    CT_CUDA_CHECK(cudaMemcpyPeerAsync(d_sino0 + off, 0, d_sino_i[i] + off, i,
                                                      bytes, stream_i[i]));
                    inSino0 = true;
                }
            }
        }
        chunks.push_back({a0, cnt, i, inSino0});
    }

    // 各卡计算/peer 拷贝完成（须在各自设备上下文中同步各自流）
    for (int i = 0; i < devCount; ++i) {
        CT_CUDA_CHECK(cudaSetDevice(i));
        CT_CUDA_CHECK(cudaStreamSynchronize(stream_i[i]));
    }

    // 汇聚到 host：落在 d_sino0 的块从 0 卡一次 D2H，其余块由所在卡直接 D2H
    CT_CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t gather = nullptr;
    CT_CUDA_CHECK(cudaStreamCreate(&gather));
    for (const Chunk& ck : chunks) {
        const size_t bytes = sizeof(float) * (size_t)ck.cnt * g.n_det;
        const size_t off = (size_t)ck.a0 * g.n_det;
        if (ck.inSino0)
            CT_CUDA_CHECK(cudaMemcpyAsync(host_out + off, d_sino0 + off, bytes,
                                          cudaMemcpyDeviceToHost, gather));
        else {
            CT_CUDA_CHECK(cudaSetDevice(ck.dev));
            CT_CUDA_CHECK(cudaMemcpyAsync(host_out + off, d_sino_i[ck.dev] + off, bytes,
                                          cudaMemcpyDeviceToHost, stream_i[ck.dev]));
        }
    }
    for (int i = 0; i < devCount; ++i) {
        CT_CUDA_CHECK(cudaSetDevice(i));
        CT_CUDA_CHECK(cudaStreamSynchronize(stream_i[i]));
    }
    CT_CUDA_CHECK(cudaSetDevice(0));
    CT_CUDA_CHECK(cudaStreamSynchronize(gather));
    rep.totalMs = t.tockMs();

    CT_CUDA_CHECK(cudaStreamDestroy(gather));
    for (int i = 0; i < devCount; ++i) {
        CT_CUDA_CHECK(cudaSetDevice(i));
        CT_CUDA_CHECK(cudaFree(d_img_i[i]));
        CT_CUDA_CHECK(cudaFree(d_sino_i[i]));
        CT_CUDA_CHECK(cudaStreamDestroy(stream_i[i]));
    }
    CT_CUDA_CHECK(cudaSetDevice(0));
    return rep;
}
