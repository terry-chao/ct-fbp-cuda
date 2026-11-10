// 阶段 9：CUDA Graphs —— 滤波→反投影链的 stream capture / instantiate / replay。
// 知识点：cudaStreamBeginCapture(ThreadLocal) / cudaStreamEndCapture /
// cudaGraphInstantiate / cudaGraphLaunch / cudaGraphGetNodes / cudaGraphExecDestroy。
#include "graph.h"
#include "backprojector.h"
#include "cuda_check.h"
#include "timer.h"

namespace {

struct GraphCache {
    bool valid = false;
    int kn = 0, kdet = 0, kangle = 0, kwin = -1;
    bool ktexture = false;
    const float* ksrc = nullptr;
    const float* kdst = nullptr;
    cudaGraphExec_t exec = nullptr;
    float* d_filt = nullptr;  // 图内引用的中间缓冲，须与 exec 同生命周期（不可释放）
    int nodes = 0;
};
GraphCache g_cache;

}  // namespace

GraphReport reconstructFBPGraph(const float* d_sino, const Geometry& g, FilterWindow win,
                                bool texture, float* d_recon, int replays) {
    GraphReport rep;
    rep.replays = replays;

    const bool keyMatch = g_cache.valid && g_cache.kn == g.n && g_cache.kdet == g.n_det &&
                          g_cache.kangle == g.n_angle && g_cache.kwin == (int)win &&
                          g_cache.ktexture == texture && g_cache.ksrc == d_sino &&
                          g_cache.kdst == d_recon;

    if (!keyMatch) {
        if (g_cache.exec) {
            CT_CUDA_CHECK(cudaGraphExecDestroy(g_cache.exec));
            CT_CUDA_CHECK(cudaFree(g_cache.d_filt));
            g_cache = GraphCache{};
        }

        cudaStream_t stream;
        CT_CUDA_CHECK(cudaStreamCreate(&stream));

        // 热身：在目标流上执行一次整条链，确保 cuFFT plan / 纹理对象 / 三角函数表
        // 等缓存在捕获区外建立（捕获区内禁止分配类 API）。
        // 注：滤波输出为正弦图尺寸，需独立缓冲（不能复用图像尺寸的 d_recon）。
        float* d_filt = nullptr;
        CT_CUDA_CHECK(cudaMalloc(&d_filt, sizeof(float) * g.sinoPixels()));
        rampFilterGPU(d_sino, g, win, d_filt, stream);
        texture ? backprojectGPUTexture(d_filt, g, d_recon, stream)
                : backprojectGPU(d_filt, g, d_recon, stream);
        CT_CUDA_CHECK(cudaStreamSynchronize(stream));

        // 捕获：同一序列在捕获模式下重放一遍，生成图
        CT_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        rampFilterGPU(d_sino, g, win, d_filt, stream);
        texture ? backprojectGPUTexture(d_filt, g, d_recon, stream)
                : backprojectGPU(d_filt, g, d_recon, stream);
        cudaGraph_t graph = nullptr;
        CT_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

        size_t nNodes = 0;
        CT_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nNodes));
        CT_CUDA_CHECK(cudaGraphInstantiate(&g_cache.exec, graph, 0));
        CT_CUDA_CHECK(cudaGraphDestroy(graph));
        // 注意：d_filt 被图节点引用，必须随 exec 常驻（释放后重放即非法访存）
        g_cache.d_filt = d_filt;
        CT_CUDA_CHECK(cudaStreamDestroy(stream));

        g_cache.valid = true;
        g_cache.kn = g.n;
        g_cache.kdet = g.n_det;
        g_cache.kangle = g.n_angle;
        g_cache.kwin = (int)win;
        g_cache.ktexture = texture;
        g_cache.ksrc = d_sino;
        g_cache.kdst = d_recon;
        g_cache.nodes = (int)nNodes;
    }
    rep.nodes = g_cache.nodes;

    // ---- 图重放计时 ----
    {
        cudaStream_t stream;
        CT_CUDA_CHECK(cudaStreamCreate(&stream));
        for (int i = 0; i < 3; ++i) CT_CUDA_CHECK(cudaGraphLaunch(g_cache.exec, stream));  // 预热
        CT_CUDA_CHECK(cudaStreamSynchronize(stream));
        GpuTimer t;
        t.start();
        for (int i = 0; i < replays; ++i) CT_CUDA_CHECK(cudaGraphLaunch(g_cache.exec, stream));
        t.stop();
        rep.graphMs = t.ms() / replays;
        CT_CUDA_CHECK(cudaStreamDestroy(stream));
    }

    // ---- 对照：手动逐次 launch 整条链（含每次的 5~6 个节点提交）----
    {
        cudaStream_t stream;
        CT_CUDA_CHECK(cudaStreamCreate(&stream));
        float* d_filt = nullptr;
        CT_CUDA_CHECK(cudaMalloc(&d_filt, sizeof(float) * g.sinoPixels()));
        for (int i = 0; i < 3; ++i) {
            rampFilterGPU(d_sino, g, win, d_filt, stream);
            texture ? backprojectGPUTexture(d_filt, g, d_recon, stream)
                    : backprojectGPU(d_filt, g, d_recon, stream);
        }
        CT_CUDA_CHECK(cudaStreamSynchronize(stream));
        GpuTimer t;
        t.start();
        for (int i = 0; i < replays; ++i) {
            rampFilterGPU(d_sino, g, win, d_filt, stream);
            texture ? backprojectGPUTexture(d_filt, g, d_recon, stream)
                    : backprojectGPU(d_filt, g, d_recon, stream);
        }
        t.stop();
        rep.manualMs = t.ms() / replays;
        CT_CUDA_CHECK(cudaFree(d_filt));
        CT_CUDA_CHECK(cudaStreamDestroy(stream));
    }
    return rep;
}

void graphReleaseCache() {
    if (g_cache.exec) {
        cudaGraphExecDestroy(g_cache.exec);
        cudaFree(g_cache.d_filt);
        g_cache = GraphCache{};
    }
}
