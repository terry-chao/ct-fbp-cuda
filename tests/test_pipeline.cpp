// 轻量级测试（零依赖，断言失败打印明细，返回失败数）：
//   1. PGM 读写往返
//   2. 单椭圆正弦图 → 正弦曲线轨迹（验证坐标/角度约定）
//   3. 中心圆盘 FBP 重建值 ≈ 1（验证整体定标）
//   4. Shepp-Logan CPU 全流程 RMSE
//   5. GPU vs CPU 逐阶段一致性（无 CUDA 设备时跳过）
//   6. SIRT 迭代重建 + GPU 指标（阶段 6：cuBLAS/Thrust/统一内存，无设备时跳过）
//   7. 前向投影变体一致性（阶段 7：warp/atomic/cdp vs v1，无设备时跳过）
//   8. 异步流水线/多GPU 前向投影一致性（阶段 8，无设备时跳过）
#include "backprojector.h"
#include "cuda_check.h"
#include "filter.h"
#include "geometry.h"
#include "io.h"
#include "metrics.h"
#include "phantom.h"
#include "pipeline.h"
#include "projector.h"
#include "sirt.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            ++g_failures;                                             \
            std::printf("    [FAIL] %s:%d  ", __FILE__, __LINE__);    \
            std::printf(__VA_ARGS__);                                 \
            std::printf("\n");                                        \
        }                                                             \
    } while (0)

static void testPGMRoundTrip() {
    std::printf("[test] PGM 读写往返\n");
    const int w = 33, h = 17;
    std::vector<float> img((size_t)w * h);
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j) img[(size_t)i * w + j] = (float)(i * w + j) / (w * h);

    CHECK(writePGM("test_roundtrip.pgm", img.data(), w, h, "roundtrip"), "写入失败");
    std::vector<float> back;
    int bw = 0, bh = 0;
    CHECK(readPGM("test_roundtrip.pgm", back, bw, bh), "读回失败");
    CHECK(bw == w && bh == h, "尺寸不符: %d x %d", bw, bh);
    double maxErr = 0;
    for (size_t i = 0; i < img.size(); ++i) maxErr = std::fmax(maxErr, std::fabs(img[i] - back[i]));
    CHECK(maxErr <= 1.0 / 255.0 + 1e-6, "8-bit 量化误差过大: %g", maxErr);
    std::remove("test_roundtrip.pgm");
}

static void testSingleEllipseSinogram() {
    std::printf("[test] 单椭圆正弦图轨迹（坐标约定）\n");
    Geometry g{};
    g.n = 64;
    g.n_det = 64;
    g.n_angle = 32;

    // 一个偏离中心的小圆盘：中心 (0.3, 0.2)，半径 0.05，值 1
    const float x0 = 0.3f, y0 = 0.2f, r = 0.05f;
    Ellipse e{r, r, x0, y0, 0.f, 1.f};
    std::vector<float> img = makeEllipses(g.n, &e, 1);

    std::vector<float> sino(g.sinoPixels());
    forwardProjectCPU(img.data(), g, sino.data());

    // 每个角度的投影峰值应落在 s* = x0*cos + y0*sin 处（±2 个探测器单元）
    float peakSum = 0.f;
    for (int a = 0; a < g.n_angle; ++a) {
        float th = g.angle(a);
        float sExpect = x0 * std::cos(th) + y0 * std::sin(th);
        int best = 0;
        for (int k = 1; k < g.n_det; ++k)
            if (sino[(size_t)a * g.n_det + k] > sino[(size_t)a * g.n_det + best]) best = k;
        float sGot = g.detS(best);
        CHECK(std::fabs(sGot - sExpect) <= 2.5f * g.ds(), "angle %d: 期望峰值 s=%.3f, 实测 %.3f",
              a, sExpect, sGot);
        peakSum += sino[(size_t)a * g.n_det + best];
    }
    // 峰值投影值 ≈ 2r（直径），定标合理性检查（放宽到 ±40%）
    float meanPeak = peakSum / g.n_angle;
    CHECK(meanPeak > 0.05f && meanPeak < 0.2f, "峰值量级异常: %.4f (期望 ~%.2f)", meanPeak, 2 * r);
}

static void testDiskReconstruction() {
    std::printf("[test] 中心圆盘 FBP 定标（重建值 ≈ 1）\n");
    Geometry g{};
    g.n = 128;
    g.n_det = 128;
    g.n_angle = 128;

    const float r = 0.5f;
    Ellipse e{r, r, 0.f, 0.f, 0.f, 1.f};
    std::vector<float> img = makeEllipses(g.n, &e, 1);

    std::vector<float> sino(g.sinoPixels()), filt(g.sinoPixels()), recon(g.imagePixels());
    forwardProjectCPU(img.data(), g, sino.data());
    rampFilterCPU(sino.data(), g, FilterWindow::RamLak, filt.data());
    backprojectCPU(filt.data(), g, recon.data());

    float center = recon[(size_t)(g.n / 2) * g.n + g.n / 2];
    CHECK(center > 0.9f && center < 1.1f, "中心重建值 %.4f，期望 ≈ 1", center);
    float corner = recon[(size_t)8 * g.n + 8];  // 左上角远离圆盘
    CHECK(std::fabs(corner) < 0.1f, "背景角落值 %.4f，期望 ≈ 0", corner);

    Metrics m = computeMetrics(img.data(), recon.data(), g.imagePixels());
    std::printf("    重建 vs 体模: RMSE=%.4f PSNR=%.2f dB 中心=%.3f\n", m.rmse, m.psnr, center);
    CHECK(m.rmse < 0.08, "RMSE %.4f 超阈值", m.rmse);
}

static void testSheppLoganCPU() {
    std::printf("[test] Shepp-Logan CPU 全流程\n");
    Geometry g{};
    g.n = 128;
    g.n_det = 128;
    g.n_angle = 90;

    std::vector<float> phantom = makeSheppLogan(g.n);
    std::vector<float> sino(g.sinoPixels()), filt(g.sinoPixels()), recon(g.imagePixels());
    forwardProjectCPU(phantom.data(), g, sino.data());
    rampFilterCPU(sino.data(), g, FilterWindow::RamLak, filt.data());
    backprojectCPU(filt.data(), g, recon.data());

    Metrics m = computeMetrics(phantom.data(), recon.data(), g.imagePixels());
    std::printf("    RMSE=%.4f PSNR=%.2f dB\n", m.rmse, m.psnr);
    // 128×128 / 90 角度 / Ram-Lak 的离散 FBP 参考水平：RMSE ~0.06，PSNR ~24 dB
    // （Shepp-Logan 边缘锐利，模糊+条纹伪影决定了该量级下限）
    CHECK(m.rmse < 0.08, "RMSE %.4f 超阈值", m.rmse);
    CHECK(m.psnr > 22.0, "PSNR %.2f 偏低", m.psnr);
}

static void testGpuMatchesCpu() {
    std::printf("[test] GPU vs CPU 逐阶段一致性\n");
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::printf("    (跳过: 无可用 CUDA 设备)\n");
        return;
    }

    Geometry g{};
    g.n = 128;
    g.n_det = 128;
    g.n_angle = 90;

    std::vector<float> phantom = makeSheppLogan(g.n);
    std::vector<float> sinoC(g.sinoPixels()), filtC(g.sinoPixels()), reconC(g.imagePixels());
    forwardProjectCPU(phantom.data(), g, sinoC.data());
    rampFilterCPU(sinoC.data(), g, FilterWindow::RamLak, filtC.data());
    backprojectCPU(filtC.data(), g, reconC.data());

    float *d_img, *d_sino, *d_filt, *d_recon;
    CT_CUDA_CHECK(cudaMalloc(&d_img, sizeof(float) * g.imagePixels()));
    CT_CUDA_CHECK(cudaMalloc(&d_sino, sizeof(float) * g.sinoPixels()));
    CT_CUDA_CHECK(cudaMalloc(&d_filt, sizeof(float) * g.sinoPixels()));
    CT_CUDA_CHECK(cudaMalloc(&d_recon, sizeof(float) * g.imagePixels()));
    CT_CUDA_CHECK(cudaMemcpy(d_img, phantom.data(), sizeof(float) * g.imagePixels(), cudaMemcpyHostToDevice));

    std::vector<float> sinoG(g.sinoPixels()), reconG(g.imagePixels()), reconT(g.imagePixels());

    // 阶段 1: 前向投影
    forwardProjectGPU(d_img, g, d_sino);
    CT_CUDA_CHECK(cudaMemcpy(sinoG.data(), d_sino, sizeof(float) * g.sinoPixels(), cudaMemcpyDeviceToHost));
    Metrics m1 = computeMetrics(sinoC.data(), sinoG.data(), g.sinoPixels());
    std::printf("    正弦图      max_abs=%.3e rmse=%.3e\n", m1.max_abs, m1.rmse);
    CHECK(m1.max_abs < 1e-3f, "正弦图 max_abs %.3e 超阈值", m1.max_abs);

    // 阶段 2: 滤波（注意 GPU 滤波输入为 GPU 正弦图，与 CPU 流水线各自独立端到端）
    rampFilterGPU(d_sino, g, FilterWindow::RamLak, d_filt);
    std::vector<float> filtG(g.sinoPixels());
    CT_CUDA_CHECK(cudaMemcpy(filtG.data(), d_filt, sizeof(float) * g.sinoPixels(), cudaMemcpyDeviceToHost));

    // 阶段 3: 反投影 naive / texture
    backprojectGPU(d_filt, g, d_recon);
    CT_CUDA_CHECK(cudaMemcpy(reconG.data(), d_recon, sizeof(float) * g.imagePixels(), cudaMemcpyDeviceToHost));
    Metrics m3 = computeMetrics(reconC.data(), reconG.data(), g.imagePixels());
    std::printf("    重建 naive  max_abs=%.3e rmse=%.3e\n", m3.max_abs, m3.rmse);
    CHECK(m3.rmse < 1e-3, "重建(naive) rmse %.3e 超阈值", m3.rmse);

    backprojectGPUTexture(d_filt, g, d_recon);
    CT_CUDA_CHECK(cudaMemcpy(reconT.data(), d_recon, sizeof(float) * g.imagePixels(), cudaMemcpyDeviceToHost));
    Metrics m4 = computeMetrics(reconC.data(), reconT.data(), g.imagePixels());
    std::printf("    重建 texture max_abs=%.3e rmse=%.3e (含 1/512 插值量化)\n", m4.max_abs, m4.rmse);
    CHECK(m4.rmse < 5e-3, "重建(texture) rmse %.3e 超阈值", m4.rmse);

    // GPU 端到端 vs 体模
    Metrics m5 = computeMetrics(phantom.data(), reconG.data(), g.imagePixels());
    std::printf("    GPU 端到端 vs 体模: RMSE=%.4f PSNR=%.2f dB\n", m5.rmse, m5.psnr);
    CHECK(m5.rmse < 0.08, "GPU 端到端 RMSE %.4f 超阈值", m5.rmse);

    CT_CUDA_CHECK(cudaFree(d_img));
    CT_CUDA_CHECK(cudaFree(d_sino));
    CT_CUDA_CHECK(cudaFree(d_filt));
    CT_CUDA_CHECK(cudaFree(d_recon));
    rampFilterGPUReleaseCache();
    backprojectorGPUReleaseCache();
}

// 阶段 6：SIRT（少角度优于 FBP）+ fp16 矩阵精度 + 统一内存 + GPU 指标
static void testSirtReconstruction() {
    std::printf("[test] SIRT 迭代重建（cuBLAS/Thrust/UM）+ GPU 指标\n");
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::printf("    (跳过: 无可用 CUDA 设备)\n");
        return;
    }

    Geometry g{};  // 极少角度：FBP 条纹伪影严重，SIRT 应显著更优
    g.n = 128;
    g.n_det = 128;
    g.n_angle = 16;

    std::vector<float> phantom = makeSheppLogan(g.n);
    std::vector<float> sino(g.sinoPixels()), filt(g.sinoPixels()), reconFbp(g.imagePixels());
    forwardProjectCPU(phantom.data(), g, sino.data());
    rampFilterCPU(sino.data(), g, FilterWindow::RamLak, filt.data());
    backprojectCPU(filt.data(), g, reconFbp.data());
    Metrics mf = computeMetrics(phantom.data(), reconFbp.data(), g.imagePixels());
    std::printf("    FBP  参照: RMSE=%.4f PSNR=%.2f dB\n", mf.rmse, mf.psnr);

    float *d_b, *d_phantom, *d_x;
    CT_CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * g.sinoPixels()));
    CT_CUDA_CHECK(cudaMalloc(&d_phantom, sizeof(float) * g.imagePixels()));
    CT_CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * g.imagePixels()));
    CT_CUDA_CHECK(cudaMemcpy(d_b, sino.data(), sizeof(float) * g.sinoPixels(), cudaMemcpyHostToDevice));
    CT_CUDA_CHECK(cudaMemcpy(d_phantom, phantom.data(), sizeof(float) * g.imagePixels(), cudaMemcpyHostToDevice));

    SirtOptions o;
    o.iterations = 300;
    o.relax = 1.9f;  // λ 接近 2 收敛更快（SIRT 稳定性要求 λ < 2）

    // fp32(TF32) + 显式设备内存
    o.prec = SirtMatrixPrecision::Fp32Tf32;
    o.alloc = SirtAllocMode::Device;
    SirtReport rep = sirtReconstructGPU(d_b, d_phantom, g, o, d_x);
    std::vector<float> x32(g.imagePixels());
    CT_CUDA_CHECK(cudaMemcpy(x32.data(), d_x, sizeof(float) * g.imagePixels(), cudaMemcpyDeviceToHost));
    Metrics m32 = computeMetrics(phantom.data(), x32.data(), g.imagePixels());
    std::printf("    SIRT fp32: RMSE=%.4f PSNR=%.2f dB (%.2f ms/iter, 残差 %.3e)\n", m32.rmse,
                m32.psnr, rep.perIterMs, rep.finalResidual);
    CHECK(m32.rmse < mf.rmse * 0.8, "SIRT RMSE %.4f 未显著优于 FBP %.4f", m32.rmse, mf.rmse);
    CHECK(rep.finalResidual > 0.0 && rep.iterRmse.size() >= 2, "收敛曲线记录异常");

    // fp16 矩阵：FP16 Tensor Core，结果应与 fp32 接近
    o.prec = SirtMatrixPrecision::Fp16;
    sirtReconstructGPU(d_b, d_phantom, g, o, d_x);
    std::vector<float> x16(g.imagePixels());
    CT_CUDA_CHECK(cudaMemcpy(x16.data(), d_x, sizeof(float) * g.imagePixels(), cudaMemcpyDeviceToHost));
    Metrics m16 = computeMetrics(phantom.data(), x16.data(), g.imagePixels());
    std::printf("    SIRT fp16: RMSE=%.4f PSNR=%.2f dB\n", m16.rmse, m16.psnr);
    CHECK(std::fabs(m16.rmse - m32.rmse) < 5e-3, "fp16 RMSE %.4f 偏离 fp32 %.4f", m16.rmse, m32.rmse);
    CHECK(m16.rmse < mf.rmse * 0.8, "SIRT fp16 RMSE %.4f 未优于 FBP", m16.rmse);

    // 统一内存 + prefetch：迭代路径与显式设备内存相同，结果应一致
    float *mb, *mx;
    CT_CUDA_CHECK(cudaMallocManaged(&mb, sizeof(float) * g.sinoPixels()));
    CT_CUDA_CHECK(cudaMallocManaged(&mx, sizeof(float) * g.imagePixels()));
    CT_CUDA_CHECK(cudaMemcpy(mb, sino.data(), sizeof(float) * g.sinoPixels(), cudaMemcpyHostToDevice));
    o.prec = SirtMatrixPrecision::Fp32Tf32;
    o.alloc = SirtAllocMode::Managed;
    sirtReconstructGPU(mb, d_phantom, g, o, mx);
    CT_CUDA_CHECK(cudaDeviceSynchronize());
    Metrics mm = computeMetrics(phantom.data(), mx, g.imagePixels());
    std::printf("    SIRT managed: RMSE=%.4f（与 device 差 %.2e）\n", mm.rmse,
                std::fabs(mm.rmse - m32.rmse));
    CHECK(std::fabs(mm.rmse - m32.rmse) < 1e-6, "managed RMSE %.6f 偏离 device %.6f", mm.rmse,
          m32.rmse);

    // GPU 指标 vs CPU 指标（同一份数据：mx 既可主机读又可设备读——统一内存）
    Metrics mg = computeMetricsGPU(d_phantom, mx, g.imagePixels());
    std::printf("    指标 GPU vs CPU: rmse 差 %.2e, max_abs 差 %.2e\n",
                std::fabs(mg.rmse - mm.rmse), std::fabs(mg.max_abs - mm.max_abs));
    CHECK(std::fabs(mg.rmse - mm.rmse) < 1e-6, "GPU 指标 rmse %.8f ≠ CPU %.8f", mg.rmse, mm.rmse);
    CHECK(std::fabs(mg.max_abs - mm.max_abs) < 1e-6, "GPU 指标 max_abs 不一致");

    CT_CUDA_CHECK(cudaFree(d_b));
    CT_CUDA_CHECK(cudaFree(d_phantom));
    CT_CUDA_CHECK(cudaFree(d_x));
    CT_CUDA_CHECK(cudaFree(mb));
    CT_CUDA_CHECK(cudaFree(mx));
    sirtReleaseCache();
}

// 阶段 7：前向投影三种并行变体（warp 协作归约 / 原子散射 / 动态并行）与 v1 一致
static void testForwardVariantsMatch() {
    std::printf("[test] 前向投影变体一致性（warp/atomic/cdp vs v1）\n");
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::printf("    (跳过: 无可用 CUDA 设备)\n");
        return;
    }

    Geometry g{};
    g.n = 128;
    g.n_det = 128;
    g.n_angle = 90;
    std::vector<float> phantom = makeSheppLogan(g.n);

    float *d_img, *d_sino[4];
    CT_CUDA_CHECK(cudaMalloc(&d_img, sizeof(float) * g.imagePixels()));
    CT_CUDA_CHECK(cudaMemcpy(d_img, phantom.data(), sizeof(float) * g.imagePixels(), cudaMemcpyHostToDevice));
    for (auto& p : d_sino) CT_CUDA_CHECK(cudaMalloc(&p, sizeof(float) * g.sinoPixels()));

    forwardProjectGPU(d_img, g, d_sino[0]);
    forwardProjectGPUWarp(d_img, g, d_sino[1]);
    forwardProjectGPUAtomic(d_img, g, d_sino[2]);
    forwardProjectGPUCdp(d_img, g, d_sino[3]);

    std::vector<float> ref(g.sinoPixels()), got(g.sinoPixels());
    CT_CUDA_CHECK(cudaMemcpy(ref.data(), d_sino[0], sizeof(float) * g.sinoPixels(), cudaMemcpyDeviceToHost));
    const char* names[3] = {"warp", "atomic", "cdp"};
    for (int v = 1; v < 4; ++v) {
        CT_CUDA_CHECK(cudaMemcpy(got.data(), d_sino[v], sizeof(float) * g.sinoPixels(), cudaMemcpyDeviceToHost));
        Metrics m = computeMetrics(ref.data(), got.data(), g.sinoPixels());
        std::printf("    %-8s max_abs=%.3e rmse=%.3e\n", names[v - 1], m.max_abs, m.rmse);
        // 求和顺序/样本定位（双精度索引 vs 逐次 += 漂移）差异的量级上限
        CHECK(m.max_abs < 1e-3f, "%s 变体 max_abs %.3e 超阈值", names[v - 1], m.max_abs);
    }

    CT_CUDA_CHECK(cudaFree(d_img));
    for (auto& p : d_sino) CT_CUDA_CHECK(cudaFree(p));
}

// 阶段 8：多流异步流水线与多 GPU 前向投影 == 串行 v1
static void testPipelineAndMultiGpu() {
    std::printf("[test] 异步流水线/多GPU 前向投影一致性\n");
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::printf("    (跳过: 无可用 CUDA 设备)\n");
        return;
    }

    Geometry g{};
    g.n = 128;
    g.n_det = 128;
    g.n_angle = 90;
    std::vector<float> phantom = makeSheppLogan(g.n);

    float *d_img, *d_sino;
    CT_CUDA_CHECK(cudaMalloc(&d_img, sizeof(float) * g.imagePixels()));
    CT_CUDA_CHECK(cudaMalloc(&d_sino, sizeof(float) * g.sinoPixels()));
    CT_CUDA_CHECK(cudaMemcpy(d_img, phantom.data(), sizeof(float) * g.imagePixels(), cudaMemcpyHostToDevice));

    std::vector<float> ref(g.sinoPixels());
    forwardProjectGPU(d_img, g, d_sino);
    CT_CUDA_CHECK(cudaMemcpy(ref.data(), d_sino, sizeof(float) * g.sinoPixels(), cudaMemcpyDeviceToHost));

    // 多流流水线（3 流 30 块）：同一 kernel 按角度分块，结果应与全量一致
    std::vector<float> got(g.sinoPixels());
    PipelineReport pr = forwardProjectGPUPipeline(d_img, g, d_sino, got.data(), 3);
    Metrics m1 = computeMetrics(ref.data(), got.data(), g.sinoPixels());
    std::printf("    pipeline 3 流/%d 块: max_abs=%.3e (%.2f vs %.2f ms)\n", pr.chunks,
                m1.max_abs, pr.pipelinedMs, pr.serialMs);
    CHECK(m1.max_abs < 1e-6f, "流水线 max_abs %.3e 超阈值", m1.max_abs);

    // 多 GPU（单卡机器上为 1 设备退化路径，代码路径相同）
    MultiGpuReport mr = forwardProjectGPUMultiGPU(phantom.data(), g, got.data());
    Metrics m2 = computeMetrics(ref.data(), got.data(), g.sinoPixels());
    std::printf("    多GPU %d 设备: max_abs=%.3e (%.2f ms)\n", mr.deviceCount, m2.max_abs, mr.totalMs);
    CHECK(m2.max_abs < 1e-6f, "多GPU max_abs %.3e 超阈值", m2.max_abs);

    CT_CUDA_CHECK(cudaFree(d_img));
    CT_CUDA_CHECK(cudaFree(d_sino));
}

int main() {
    testPGMRoundTrip();
    testSingleEllipseSinogram();
    testDiskReconstruction();
    testSheppLoganCPU();
    testGpuMatchesCpu();
    testSirtReconstruction();
    testForwardVariantsMatch();
    testPipelineAndMultiGpu();

    if (g_failures == 0) {
        std::printf("\n全部测试通过 ✔\n");
        return 0;
    }
    std::printf("\n%d 项断言失败 ✘\n", g_failures);
    return 1;
}
