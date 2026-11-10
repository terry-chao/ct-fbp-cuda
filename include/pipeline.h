#pragma once

#include "geometry.h"

#include <string>

// ============================================================================
// 阶段 8：异步流水线与多 GPU（cudaStream / cudaMemcpyAsync / 锁页内存 / peer copy）
//
// forwardProjectGPUPipeline：把前向投影按角度分块，K 个流轮转执行
//   （kernel → D2H 异步拷贝在同流内保序），不同流的计算与拷贝互相重叠；
//   host 侧一块整幅锁页缓冲接收（各块写入互不相交的行区间，无需额外同步）。
//
// forwardProjectGPUMultiGPU：按角度把前向投影切分到所有设备，各卡独立流；
//   与主卡可 peer 时用 cudaMemcpyPeerAsync 汇聚到主卡再一次 D2H，
//   否则各卡直接 D2H。单卡机器上退化为 1 块（框架照常走通，如实记录拓扑）。
// ============================================================================

struct PipelineReport {
    int streams = 1;      // 实际使用的流数
    int chunks = 1;       // 角度分块数
    double pipelinedMs;   // 分块流水总耗时（kernel+拷贝重叠后的墙钟）
    double serialMs;      // 对照：不分块（kernel 全量 → 一次拷贝）的墙钟
    double dataMB;        // D2H 数据量
    double gbps;          // 流水吞吐 = dataMB / pipelinedMs
};

// d_img/d_sino 为设备 0 上的指针（分块 kernel 仍写完整 d_sino 的对应行区间）；
// host_out 接收整幅正弦图（函数内部经锁页缓冲中转）。
PipelineReport forwardProjectGPUPipeline(const float* d_img, const Geometry& g, float* d_sino,
                                         float* host_out, int nStreams);

struct MultiGpuReport {
    int deviceCount = 1;
    double totalMs = 0.0;
    std::string topology;  // 设备与 peer 可达性描述
};

// h_img 为主机内存图像；host_out 接收整幅正弦图。函数负责各卡的临时显存
// 分配/释放与设备切换，返回时恢复设备 0。
MultiGpuReport forwardProjectGPUMultiGPU(const float* h_img, const Geometry& g, float* host_out);
