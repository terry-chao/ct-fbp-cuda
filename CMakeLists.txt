cmake_minimum_required(VERSION 3.24)

# 目标 GPU 架构：默认 sm_120 (Blackwell / RTX 50 系列)，可 -DCMAKE_CUDA_ARCHITECTURES=86 等覆盖
# 必须在 project() 之前设置才会生效
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(ct_fbp_cuda
    VERSION 1.0.0
    DESCRIPTION "CUDA-accelerated CT filtered backprojection (FBP) reconstruction"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# 核心库：CPU 参考实现 + GPU kernels
add_library(ct_core STATIC
    src/phantom.cu
    src/projector_cpu.cpp
    src/projector_gpu.cu
    src/filter_cpu.cpp
    src/filter_gpu.cu
    src/backprojector_cpu.cpp
    src/backprojector_gpu.cu
    src/io.cpp
    src/metrics.cpp
    src/metrics_gpu.cu
    src/sirt_gpu.cu)
target_include_directories(ct_core PUBLIC include)
# thrust 为纯头文件库（位于 CUDA 工具包 include 目录，经 cudart 的接口路径引入）
target_link_libraries(ct_core PUBLIC CUDA::cudart CUDA::cufft CUDA::cublas)

# MSVC 源文件默认编码（含中文注释的文件需要 /utf-8，避免 C4819 告警）
target_compile_options(ct_core PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/utf-8>)

# 主程序
add_executable(ct_fbp src/main.cpp)
target_link_libraries(ct_fbp PRIVATE ct_core)
target_compile_options(ct_fbp PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>)

# 测试（src/ 下有内部工具头 cuda_check.h）
enable_testing()
add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE ct_core)
target_include_directories(test_pipeline PRIVATE src)
target_compile_options(test_pipeline PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>)
add_test(NAME pipeline COMMAND test_pipeline)
