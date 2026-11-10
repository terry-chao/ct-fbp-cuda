#pragma once

#include <vector>

// 椭圆定义：中心 (x0, y0)，半轴 (a 沿旋转后 x, b 沿旋转后 y)，倾角 alpha（弧度），取值 val。
// 像素值 = 各椭圆 val 之和（后画的叠加在前面之上）。
struct Ellipse {
    float a;     // 半轴 x
    float b;     // 半轴 y
    float x0;    // 中心 x
    float y0;    // 中心 y
    float alpha; // 倾角（弧度，绕中心逆时针）
    float val;   // 叠加值
};

// 解析法生成椭圆叠加图像，n x n，row-major（同 Geometry 约定：行 0 = +y）
std::vector<float> makeEllipses(int n, const Ellipse* ellipses, int count);

// 标准 Shepp-Logan 体模（MATLAB 对比度增强版，10 个椭圆）
std::vector<float> makeSheppLogan(int n);

// Shepp-Logan 椭圆参数表（测试/文档共用），返回个数
const Ellipse* sheppLoganTable(int& count);
