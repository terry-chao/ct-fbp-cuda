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

# ---------------------------------------------------------------------------
# 模块划分：公共 API 在 include/ct/<模块>/，实现在 src/<模块>/，每个模块一个静态库。
#   core           坐标/几何约定、计时封装、CUDA 错误检查、输出编码适配（内部头 src/core/）
#   phantom        Shepp-Logan 体模
#   projector      前向投影（CPU / GPU v1 / warp / atomic / CDP）
#   filter         Ramp 滤波（CPU / cuFFT）
#   backprojector  反投影（CPU / GPU naive / GPU texture）
#   metrics        RMSE / PSNR（CPU / GPU）
#   io             PGM 读写
#   recon          重建算法层：FBP 流水线、SIRT 迭代、CUDA Graphs
# ---------------------------------------------------------------------------

# MSVC 编译选项：/utf-8 让含中文注释的源文件不再触发 C4819；
# CUDA 13.x 的 CCCL（thrust / libcu++）在 MSVC 传统预处理器下会直接 #error，
# 必须显式让 cl.exe 使用标准符合预处理器（/Zc:preprocessor）。
set(CT_MSVC_COMPILE_OPTIONS
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:preprocessor>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/Zc:preprocessor>)

# 模块库统一设置：公共头在 include/，内部工具头在 src/
function(ct_add_module name)
    add_library(${name} STATIC ${ARGN})
    target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_options(${name} PRIVATE ${CT_MSVC_COMPILE_OPTIONS})
endfunction()

# --- 阶段 7：动态并行（CDP2）需要可重定位设备代码（-rdc=true）----------------
# 单独编成小静态库，让 RDC 只作用于这一个文件（全项目开 RDC 会拖慢其他 kernel 的
# 编译与代码生成）：
#   CUDA_SEPARABLE_COMPILATION 让 CMake 生成 nvcc -dlink 设备链接步骤；
#   CUDA_RESOLVE_DEVICE_SYMBOLS 把 dlink 产物并入 ct_cdp.lib，
#   最终可执行文件才能解析 projector_cdp.obj 引用的 __cudaRegisterLinkedBinary_* 符号。
add_library(ct_cdp STATIC src/projector/projector_cdp.cu)
set_target_properties(ct_cdp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(ct_cdp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(ct_cdp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ct_cdp PUBLIC CUDA::cudart)
target_compile_options(ct_cdp PRIVATE ${CT_MSVC_COMPILE_OPTIONS})

# --- 各功能模块 -------------------------------------------------------------
# 共享运行时工具：控制台/管道输出编码适配（Windows 中文乱码）
ct_add_module(ct_core src/core/console_io.cpp)

ct_add_module(ct_io src/io/io.cpp)

ct_add_module(ct_phantom src/phantom/phantom.cu)
target_link_libraries(ct_phantom PUBLIC CUDA::cudart)

ct_add_module(ct_projector
    src/projector/projector_cpu.cpp
    src/projector/projector_gpu.cu)
# CDP 版前向投影是 projector 模块的组成实现（独立静态库仅为满足 -rdc 约束）
target_link_libraries(ct_projector PUBLIC ct_cdp CUDA::cudart)

ct_add_module(ct_filter
    src/filter/filter_cpu.cpp
    src/filter/filter_gpu.cu)
target_link_libraries(ct_filter PUBLIC CUDA::cudart CUDA::cufft)

ct_add_module(ct_backprojector
    src/backprojector/backprojector_cpu.cpp
    src/backprojector/backprojector_gpu.cu)
target_link_libraries(ct_backprojector PUBLIC CUDA::cudart)

ct_add_module(ct_metrics
    src/metrics/metrics.cpp
    src/metrics/metrics_gpu.cu)
target_link_libraries(ct_metrics PUBLIC CUDA::cudart)

# 重建算法层：串联体模之外的全部阶段
ct_add_module(ct_recon
    src/recon/pipeline_gpu.cu
    src/recon/sirt_gpu.cu
    src/recon/graph_gpu.cu)
target_link_libraries(ct_recon PUBLIC
    ct_projector
    ct_filter
    ct_backprojector
    ct_metrics
    CUDA::cudart
    CUDA::cublas)

# 可执行文件统一链接的模块集合
set(CT_APP_MODULES
    ct_core
    ct_recon
    ct_projector
    ct_filter
    ct_backprojector
    ct_metrics
    ct_phantom
    ct_io)

# --- 可执行文件 -------------------------------------------------------------
# 主程序
add_executable(ct_fbp src/app/main.cpp)
target_link_libraries(ct_fbp PRIVATE ${CT_APP_MODULES})
target_include_directories(ct_fbp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(ct_fbp PRIVATE ${CT_MSVC_COMPILE_OPTIONS})

# 知识点巡礼（阶段 9）：小配置一键跑全部功能
add_executable(ct_tour src/app/tour.cpp)
target_link_libraries(ct_tour PRIVATE ${CT_APP_MODULES})
target_include_directories(ct_tour PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(ct_tour PRIVATE ${CT_MSVC_COMPILE_OPTIONS})

# 测试（内部工具头在 src/core/ 下）
enable_testing()
add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE ${CT_APP_MODULES})
target_include_directories(test_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(test_pipeline PRIVATE ${CT_MSVC_COMPILE_OPTIONS})
add_test(NAME pipeline COMMAND test_pipeline)
