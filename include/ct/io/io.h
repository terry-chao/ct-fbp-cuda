#pragma once

#include <string>
#include <vector>

// PGM 图像读写（零依赖）。正弦图 / 图像均为 float 数组，写文件时归一化到 0..255。
// 返回 false 表示 IO 失败（原因打印到 stderr）。

// 按 [vmin, vmax] 线性映射写 8-bit 二进制 PGM（P5）。多图对比时统一灰阶用。
bool writePGMRanged(const std::string& path, const float* img, int w, int h,
                    float vmin, float vmax, const std::string& comment = "");

// 先扫描 [min, max] 再写（单图快速可视化用）。
bool writePGM(const std::string& path, const float* img, int w, int h,
              const std::string& comment = "");

// 读 PGM（P5 二进制 / P2 文本均可），像素值线性映射到 [0,1] 的 float。
bool readPGM(const std::string& path, std::vector<float>& img, int& w, int& h);
