// CLI 与完整流水线编排：
//   体模 → 前向投影 → Ramp 滤波 → 反投影 → 输出图像/指标/计时表
// 用法见 usage()。--mode both 时逐阶段比对 GPU 与 CPU 结果。
#include "backprojector.h"
#include "cuda_check.h"
#include "filter.h"
#include "geometry.h"
#include "io.h"
#include "metrics.h"
#include "phantom.h"
#include "projector.h"
#include "sirt.h"
#include "timer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX  // windows.h 的 min/max 宏会与下方 std::min/std::max 冲突
#include <windows.h>  // SetConsoleOutputCP：中文 Windows 控制台默认 GBK，需切 UTF-8 才不乱码
#endif

namespace fs = std::filesystem;

struct Options {
    Geometry g{};
    std::string mode = "both";   // cpu | gpu | both
    std::string filterName = "ram-lak";
    FilterWindow win = FilterWindow::RamLak;
    std::string out = "output";
    int warmup = 3;
    int runs = 5;
    // 阶段 6：SIRT 迭代重建
    std::string recon = "fbp";     // fbp | sirt
    int iters = 60;
    double relax = 1.0;            // SIRT 松弛因子 λ（收敛稳定性要求 λ < 2）
    std::string alloc = "device";  // device | managed（统一内存）
    std::string matrix = "fp16";   // fp32(TF32) | fp16(Tensor Core)
};

static void usage(const char* prog) {
    std::printf(
        "基于 CUDA 的 CT 滤波反投影（FBP）重建器\n"
        "用法: %s [选项]\n"
        "  --size N      图像边长 N×N（默认 512）\n"
        "  --det D       探测器单元数（默认 512）\n"
        "  --angles A    投影角度数，覆盖 [0,180)（默认 360）\n"
        "  --mode M      cpu | gpu | both（默认 both）\n"
        "  --filter F    ram-lak | hamming（默认 ram-lak）\n"
        "  --warmup W    GPU 计时前预热次数（默认 3）\n"
        "  --runs R      计时重复次数取平均（默认 5，CPU 预热固定 1 次）\n"
        "  --out DIR     输出目录（默认 output）\n"
        "  --recon R     fbp | sirt（默认 fbp；sirt 为迭代重建，少角度下质量更优，\n"
        "                需 GPU，稠密矩阵限制 --size/--det/--angles ≤ 128）\n"
        "  --iters N     SIRT 迭代次数（默认 60）\n"
        "  --matrix M    fp32 | fp16（SIRT 系统矩阵精度：TF32 / FP16 Tensor Core，默认 fp16）\n"
        "  --alloc A     device | managed（SIRT 内存模式：显式 / 统一内存，默认 device）\n",
        prog);
}

static bool parseArgs(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "错误: %s 缺少参数\n", what);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--size") o.g.n = std::atoi(next("--size").c_str());
        else if (a == "--det") o.g.n_det = std::atoi(next("--det").c_str());
        else if (a == "--angles") o.g.n_angle = std::atoi(next("--angles").c_str());
        else if (a == "--mode") o.mode = next("--mode");
        else if (a == "--filter") o.filterName = next("--filter");
        else if (a == "--warmup") o.warmup = std::atoi(next("--warmup").c_str());
        else if (a == "--runs") o.runs = std::atoi(next("--runs").c_str());
        else if (a == "--out") o.out = next("--out");
        else if (a == "--recon") o.recon = next("--recon");
        else if (a == "--iters") o.iters = std::atoi(next("--iters").c_str());
        else if (a == "--relax") o.relax = std::atof(next("--relax").c_str());
        else if (a == "--matrix") o.matrix = next("--matrix");
        else if (a == "--alloc") o.alloc = next("--alloc");
        else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "错误: 未知选项 '%s'\n\n", a.c_str());
            usage(argv[0]);
            return false;
        }
    }
    if (o.g.n <= 0 || o.g.n_det <= 0 || o.g.n_angle <= 0 || o.runs <= 0 || o.warmup < 0 ||
        o.iters <= 0 || o.relax <= 0.0 || o.relax >= 2.0) {
        std::fprintf(stderr, "错误: 参数必须为正整数\n");
        return false;
    }
    if (o.mode != "cpu" && o.mode != "gpu" && o.mode != "both") {
        std::fprintf(stderr, "错误: --mode 只支持 cpu | gpu | both\n");
        return false;
    }
    if (o.recon != "fbp" && o.recon != "sirt") {
        std::fprintf(stderr, "错误: --recon 只支持 fbp | sirt\n");
        return false;
    }
    if (o.matrix != "fp32" && o.matrix != "fp16") {
        std::fprintf(stderr, "错误: --matrix 只支持 fp32 | fp16\n");
        return false;
    }
    if (o.alloc != "device" && o.alloc != "managed") {
        std::fprintf(stderr, "错误: --alloc 只支持 device | managed\n");
        return false;
    }
    if (o.filterName == "ram-lak") o.win = FilterWindow::RamLak;
    else if (o.filterName == "hamming") o.win = FilterWindow::Hamming;
    else {
        std::fprintf(stderr, "错误: --filter 只支持 ram-lak | hamming\n");
        return false;
    }
    return true;
}

// ---- 计时辅助 ---------------------------------------------------------------
template <class F>
double timeCpuMs(F&& fn, int runs) {
    fn();  // 预热 1 次（缓存热度），再正式计时
    double total = 0.0;
    CpuTimer t;
    for (int i = 0; i < runs; ++i) {
        t.tick();
        fn();
        total += t.tockMs();
    }
    return total / runs;
}

template <class F>
float timeGpuMs(F&& fn, int warmup, int runs) {
    for (int i = 0; i < warmup; ++i) fn();
    CT_CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t;
    double total = 0.0;
    for (int i = 0; i < runs; ++i) {
        t.start();
        fn();
        t.stop();
        total += t.ms();
    }
    return (float)(total / runs);
}

struct StageTiming {
    const char* name;
    double cpu_ms = -1.0;
    double gpu_ms = -1.0;
};

static void reportDiff(const char* what, const float* cpu, const float* gpu, size_t n) {
    Metrics m = computeMetrics(cpu, gpu, n);
    std::printf("  %-24s max_abs = %-10.3e rmse = %.3e\n", what, m.max_abs, m.rmse);
}

static void reportQuality(const char* what, const float* phantom, const float* recon, size_t n) {
    Metrics m = computeMetrics(phantom, recon, n);
    std::printf("  %-24s RMSE = %-8.4f PSNR = %6.2f dB  (max_abs = %.3e)\n", what, m.rmse,
                m.psnr, m.max_abs);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // 源码以 /utf-8 编译，输出即 UTF-8 字节；控制台默认按系统 ANSI 页（GBK）解码会乱码。
    // 重定向到文件/管道时该调用无副作用，字节流仍为 UTF-8。
    SetConsoleOutputCP(CP_UTF8);
#endif
    Options o;
    if (!parseArgs(argc, argv, o)) return 1;
    const bool runCpu = (o.mode == "cpu" || o.mode == "both");
    const bool runGpu = (o.mode == "gpu" || o.mode == "both");

    if (o.recon == "sirt" && !runGpu) {
        std::fprintf(stderr, "错误: --recon sirt 需要 GPU（--mode gpu 或 both）\n");
        return 1;
    }

    std::printf("=== ct-fbp-cuda ===\n");
    std::printf("配置: %dx%d 图像, %d 探测器, %d 角度, 滤波 %s, 模式 %s\n\n", o.g.n, o.g.n,
                o.g.n_det, o.g.n_angle, o.filterName.c_str(), o.mode.c_str());

    if (runGpu) {
        int count = 0;
        CT_CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count <= 0) {
            std::fprintf(stderr, "错误: 未检测到 CUDA 设备\n");
            return 1;
        }
        cudaDeviceProp prop{};
        CT_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::printf("GPU: %s (sm_%d%d, %zu MiB)\n\n", prop.name, prop.major, prop.minor,
                    (size_t)prop.totalGlobalMem >> 20);
    }

    fs::create_directories(o.out);

    // ---- 体模 --------------------------------------------------------------
    std::vector<float> phantom = makeSheppLogan(o.g.n);
    writePGMRanged(o.out + "/phantom.pgm", phantom.data(), o.g.n, o.g.n, 0.f, 1.f,
                   "Shepp-Logan phantom");

    std::vector<float> sinoC, filtC, reconC;       // CPU 结果
    std::vector<float> sinoG, filtG, reconGN, reconGT;  // GPU 结果
    std::vector<StageTiming> stages = {{"前向投影"}, {"Ramp 滤波"}, {"反投影 naive"}, {"反投影 texture"}};

    // ---- CPU 参考流水线 ------------------------------------------------------
    if (runCpu) {
        sinoC.resize(o.g.sinoPixels());
        reconC.resize(o.g.imagePixels());
        filtC.resize(o.g.sinoPixels());

        stages[0].cpu_ms = timeCpuMs([&] { forwardProjectCPU(phantom.data(), o.g, sinoC.data()); }, o.runs);
        stages[1].cpu_ms = timeCpuMs([&] { rampFilterCPU(sinoC.data(), o.g, o.win, filtC.data()); }, o.runs);
        stages[2].cpu_ms = timeCpuMs([&] { backprojectCPU(filtC.data(), o.g, reconC.data()); }, o.runs);

        writePGM(o.out + "/sinogram.pgm", sinoC.data(), o.g.n_det, o.g.n_angle, "sinogram (CPU forward)");
        writePGM(o.out + "/sinogram_filtered.pgm", filtC.data(), o.g.n_det, o.g.n_angle,
                 "filtered sinogram (CPU, " + o.filterName + ")");
    }

    // ---- GPU 流水线 ----------------------------------------------------------
    if (runGpu) {
        float* d_img = nullptr;
        float* d_sino = nullptr;
        float* d_filt = nullptr;
        float* d_recon = nullptr;
        CT_CUDA_CHECK(cudaMalloc(&d_img, sizeof(float) * o.g.imagePixels()));
        CT_CUDA_CHECK(cudaMalloc(&d_sino, sizeof(float) * o.g.sinoPixels()));
        CT_CUDA_CHECK(cudaMalloc(&d_filt, sizeof(float) * o.g.sinoPixels()));
        CT_CUDA_CHECK(cudaMalloc(&d_recon, sizeof(float) * o.g.imagePixels()));
        CT_CUDA_CHECK(cudaMemcpy(d_img, phantom.data(), sizeof(float) * o.g.imagePixels(),
                                  cudaMemcpyHostToDevice));

        sinoG.resize(o.g.sinoPixels());
        filtG.resize(o.g.sinoPixels());
        reconGN.resize(o.g.imagePixels());
        reconGT.resize(o.g.imagePixels());

        stages[0].gpu_ms = timeGpuMs([&] { forwardProjectGPU(d_img, o.g, d_sino); }, o.warmup, o.runs);
        CT_CUDA_CHECK(cudaMemcpy(sinoG.data(), d_sino, sizeof(float) * o.g.sinoPixels(),
                                  cudaMemcpyDeviceToHost));
        stages[1].gpu_ms = timeGpuMs([&] { rampFilterGPU(d_sino, o.g, o.win, d_filt); }, o.warmup, o.runs);
        CT_CUDA_CHECK(cudaMemcpy(filtG.data(), d_filt, sizeof(float) * o.g.sinoPixels(),
                                  cudaMemcpyDeviceToHost));
        stages[2].gpu_ms = timeGpuMs([&] { backprojectGPU(d_filt, o.g, d_recon); }, o.warmup, o.runs);
        CT_CUDA_CHECK(cudaMemcpy(reconGN.data(), d_recon, sizeof(float) * o.g.imagePixels(),
                                  cudaMemcpyDeviceToHost));
        stages[3].gpu_ms = timeGpuMs([&] { backprojectGPUTexture(d_filt, o.g, d_recon); }, o.warmup, o.runs);
        CT_CUDA_CHECK(cudaMemcpy(reconGT.data(), d_recon, sizeof(float) * o.g.imagePixels(),
                                  cudaMemcpyDeviceToHost));

        if (!runCpu) {
            writePGM(o.out + "/sinogram.pgm", sinoG.data(), o.g.n_det, o.g.n_angle, "sinogram (GPU forward)");
        }

        // ---- 阶段 6：SIRT 迭代重建（少角度场景质量优于 FBP，可与上方 FBP 结果对比）----
        if (o.recon == "sirt") {
            SirtOptions so;
            so.iterations = o.iters;
            so.relax = (float)o.relax;
            so.prec = (o.matrix == "fp16") ? SirtMatrixPrecision::Fp16
                                           : SirtMatrixPrecision::Fp32Tf32;
            so.alloc = (o.alloc == "managed") ? SirtAllocMode::Managed
                                              : SirtAllocMode::Device;

            size_t need = sirtMatrixBytes(o.g, so.prec);
            size_t freeB = 0, totalB = 0;
            CT_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
            if (need > freeB / 2) {  // 留裕量（迭代工作区 + 上下文）
                std::fprintf(stderr,
                             "错误: SIRT 系统矩阵峰值 %.1f MiB，超出可用显存 %.1f MiB 安全线\n"
                             "      请减小 --size/--det/--angles（如 --size 128 --det 128 --angles 32）\n",
                             need / 1048576.0, freeB / 1048576.0);
                return 1;
            }

            float* d_x = nullptr;
            if (so.alloc == SirtAllocMode::Managed)
                CT_CUDA_CHECK(cudaMallocManaged(&d_x, sizeof(float) * o.g.imagePixels()));
            else
                CT_CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * o.g.imagePixels()));

            SirtReport rep = sirtReconstructGPU(d_sino, d_img, o.g, so, d_x);
            std::vector<float> reconSIRT(o.g.imagePixels());
            if (so.alloc == SirtAllocMode::Managed) {
                CT_CUDA_CHECK(cudaDeviceSynchronize());  // 统一内存：主机读取前同步
                std::copy(d_x, d_x + o.g.imagePixels(), reconSIRT.begin());
            } else {
                CT_CUDA_CHECK(cudaMemcpy(reconSIRT.data(), d_x,
                                         sizeof(float) * o.g.imagePixels(),
                                         cudaMemcpyDeviceToHost));
            }
            writePGM(o.out + "/recon_sirt.pgm", reconSIRT.data(), o.g.n, o.g.n,
                     "SIRT recon (" + o.matrix + ", " + std::to_string(o.iters) + " iters)");

            Metrics ms = computeMetricsGPU(d_img, d_x, o.g.imagePixels());
            std::printf("\n[SIRT] 系统矩阵 %dx%d，%s 矩阵 + %s 内存，峰值 %.1f MiB\n",
                        o.g.n_angle * o.g.n_det, o.g.n * o.g.n, o.matrix.c_str(),
                        o.alloc.c_str(), need / 1048576.0);
            std::printf("  构建 %.1f ms（缓存命中时为 0）| 迭代 %d 次，平均 %.2f ms/次\n",
                        rep.buildMs, o.iters, rep.perIterMs);
            std::printf("  末次残差 ||b-Ax||₂ = %.4e\n", rep.finalResidual);
            std::printf("  重建 vs 体模: RMSE = %.4f  PSNR = %.2f dB（FBP 对比见 [质量] 段）\n",
                        ms.rmse, ms.psnr);

            // 收敛曲线落盘
            FILE* f = std::fopen((o.out + "/sirt_convergence.csv").c_str(), "w");
            if (f) {
                std::fprintf(f, "iter,rmse,residual\n");
                for (size_t i = 0; i < rep.iterResidual.size(); ++i) {
                    double rmse = i < rep.iterRmse.size() ? rep.iterRmse[i] : -1.0;
                    std::fprintf(f, "%d,%.6f,%.6e\n", rep.iterIdx[i], rmse, rep.iterResidual[i]);
                }
                std::fclose(f);
            }
            CT_CUDA_CHECK(cudaFree(d_x));
        }

        CT_CUDA_CHECK(cudaFree(d_img));
        CT_CUDA_CHECK(cudaFree(d_sino));
        CT_CUDA_CHECK(cudaFree(d_filt));
        CT_CUDA_CHECK(cudaFree(d_recon));
        rampFilterGPUReleaseCache();
        backprojectorGPUReleaseCache();
        sirtReleaseCache();
    }

    // ---- 图像输出（各重建图统一灰阶，便于肉眼对比）-----------------------------
    {
        float vmin = 1e30f, vmax = -1e30f;
        auto extend = [&](const std::vector<float>& v) {
            if (v.empty()) return;
            for (float x : v) {
                vmin = std::min(vmin, x);
                vmax = std::max(vmax, x);
            }
        };
        extend(reconC);
        extend(reconGN);
        extend(reconGT);
        if (!reconC.empty())
            writePGMRanged(o.out + "/recon_cpu.pgm", reconC.data(), o.g.n, o.g.n, vmin, vmax, "FBP recon (CPU)");
        if (!reconGN.empty())
            writePGMRanged(o.out + "/recon_gpu.pgm", reconGN.data(), o.g.n, o.g.n, vmin, vmax, "FBP recon (GPU naive)");
        if (!reconGT.empty())
            writePGMRanged(o.out + "/recon_gpu_texture.pgm", reconGT.data(), o.g.n, o.g.n, vmin, vmax,
                           "FBP recon (GPU texture)");
    }

    // ---- 报告 ----------------------------------------------------------------
    if (runCpu && runGpu) {
        std::printf("[正确性] GPU vs CPU 逐阶段比对（float 累加顺序差异属正常）\n");
        reportDiff("正弦图", sinoC.data(), sinoG.data(), o.g.sinoPixels());
        reportDiff("滤波后正弦图", filtC.data(), filtG.data(), o.g.sinoPixels());
        reportDiff("重建图 naive", reconC.data(), reconGN.data(), o.g.imagePixels());
        reportDiff("重建图 texture", reconC.data(), reconGT.data(), o.g.imagePixels());
        std::printf("\n");
    }

    std::printf("[质量] 重建图 vs 体模\n");
    if (!reconC.empty()) reportQuality("CPU", phantom.data(), reconC.data(), o.g.imagePixels());
    if (!reconGN.empty()) reportQuality("GPU naive", phantom.data(), reconGN.data(), o.g.imagePixels());
    if (!reconGT.empty()) reportQuality("GPU texture", phantom.data(), reconGT.data(), o.g.imagePixels());

    std::printf("\n[性能] CPU 预热 1 次后取 %d 次均值；GPU 预热 %d 次后取 %d 次均值\n", o.runs,
                o.warmup, o.runs);
    std::printf("%-16s %12s %12s %10s\n", "阶段", "CPU (ms)", "GPU (ms)", "加速比");
    for (const StageTiming& s : stages) {
        std::string cpu = s.cpu_ms >= 0 ? std::to_string(s.cpu_ms) : "-";
        std::string gpu = s.gpu_ms >= 0 ? std::to_string(s.gpu_ms) : "-";
        double speedup = (s.cpu_ms > 0 && s.gpu_ms > 0) ? s.cpu_ms / s.gpu_ms : 0.0;
        std::string sp = speedup > 0 ? std::to_string(speedup) + "x" : "-";
        if (s.cpu_ms >= 0 && s.gpu_ms >= 0)
            std::printf("%-16s %12.2f %12.2f %9.1fx\n", s.name, s.cpu_ms, s.gpu_ms, speedup);
        else if (s.cpu_ms >= 0)
            std::printf("%-16s %12.2f %12s %10s\n", s.name, s.cpu_ms, "-", "-");
        else
            std::printf("%-16s %12s %12.2f %10s\n", s.name, "-", s.gpu_ms, "-");
    }

    // ---- 落盘：计时 CSV -------------------------------------------------------
    {
        FILE* f = std::fopen((o.out + "/timings.csv").c_str(), "w");
        if (f) {
            std::fprintf(f, "stage,cpu_ms,gpu_ms,speedup\n");
            for (const StageTiming& s : stages) {
                double sp = (s.cpu_ms > 0 && s.gpu_ms > 0) ? s.cpu_ms / s.gpu_ms : 0.0;
                std::fprintf(f, "%s,%.4f,%.4f,%.2f\n", s.name,
                             s.cpu_ms >= 0 ? s.cpu_ms : -1.0, s.gpu_ms >= 0 ? s.gpu_ms : -1.0, sp);
            }
            std::fclose(f);
        }
    }
    std::printf("\n图像与计时已写入 '%s'\n", o.out.c_str());
    return 0;
}
