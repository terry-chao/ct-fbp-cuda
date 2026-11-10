#pragma once

#include <chrono>
#include <cuda_runtime.h>

// CPU 计时：std::chrono（毫秒）
class CpuTimer {
public:
    void tick() { t0_ = std::chrono::steady_clock::now(); }
    double tockMs() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0_)
            .count();
    }

private:
    std::chrono::steady_clock::time_point t0_{};
};

// GPU 计时：CUDA Events（毫秒），只计量设备端执行，不受异步 launch 影响。
class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start() { cudaEventRecord(start_); }
    void stop() { cudaEventRecord(stop_); }
    // 阻塞直到 stop 事件完成，返回 start→stop 的毫秒数
    float ms() {
        cudaEventSynchronize(stop_);
        float m = 0.f;
        cudaEventElapsedTime(&m, start_, stop_);
        return m;
    }

private:
    cudaEvent_t start_{}, stop_{};
};
