#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cufft.h>

// CUDA / cuFFT 调用错误检查：失败即打印并退出（学习项目，fail-fast 策略）
#define CT_CUDA_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d  %s\n  -> %s\n",          \
                         __FILE__, __LINE__, #call,                            \
                         cudaGetErrorString(e_));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CT_CUFFT_CHECK(call)                                                   \
    do {                                                                       \
        cufftResult e_ = (call);                                               \
        if (e_ != CUFFT_SUCCESS) {                                             \
            std::fprintf(stderr, "cuFFT error at %s:%d  %s\n  -> %s\n",         \
                         __FILE__, __LINE__, #call,                            \
                         _cufftGetErrorString(e_));                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

inline const char* _cufftGetErrorString(cufftResult r) {
    switch (r) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        case CUFFT_MISSING_DEPENDENCY: return "CUFFT_MISSING_DEPENDENCY";
        case CUFFT_NVRTC_FAILURE: return "CUFFT_NVRTC_FAILURE";
        case CUFFT_NVJITLINK_FAILURE: return "CUFFT_NVJITLINK_FAILURE";
        case CUFFT_NVSHMEM_FAILURE: return "CUFFT_NVSHMEM_FAILURE";
        default: return "unknown cufft error";
    }
}
