// CPU Ramp 滤波：自写迭代式 radix-2 复数 FFT（正/逆变换均不归一化，1/M 折叠进频响，
// 与 GPU cuFFT 路径的数学完全一致）。
#include "filter.h"

#include <cmath>
#include <complex>
#include <vector>

int paddedLength(int n_det) {
    int m = 1;
    while (m < 2 * n_det) m <<= 1;  // M = pow2 >= 2*n_det，补零抑制 DFT 周期化伪影
    return m;
}

namespace {

// 原地 FFT（inverse=false 为正向，无归一化；twiddle 逐级直算，精度优先）
void fftInPlace(std::complex<float>* a, int M, bool inverse) {
    // 位反转置换
    for (int i = 1, j = 0; i < M; ++i) {
        int bit = M >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const float sign = inverse ? 1.f : -1.f;
    for (int len = 2; len <= M; len <<= 1) {
        float ang = sign * 2.f * CT_PI / len;
        int half = len >> 1;
        for (int base = 0; base < M; base += len) {
            for (int j = 0; j < half; ++j) {
                float warg = ang * j;
                std::complex<float> w(std::cos(warg), std::sin(warg));
                std::complex<float> u = a[base + j];
                std::complex<float> v = a[base + j + half] * w;
                a[base + j] = u + v;
                a[base + j + half] = u - v;
            }
        }
    }
}

// 频域滤波系数：H[m] = |ν_m| * w_m / M（1/M 为 IFFT 归一化，ν 单位 cycles/unit）
std::vector<float> buildRampResponse(int M, float ds, FilterWindow win) {
    std::vector<float> H(M);
    for (int m = 0; m < M; ++m) {
        int wrapped = (m <= M / 2) ? m : m - M;
        float nu = std::fabs((float)wrapped / ((float)M * ds));
        float w = 1.f;
        if (win == FilterWindow::Hamming) {
            w = 0.54f + 0.46f * std::cos(2.f * CT_PI * m / M);
        }
        H[m] = nu * w / M;
    }
    return H;
}

}  // namespace

void rampFilterCPU(const float* sino, const Geometry& g, FilterWindow win, float* out) {
    const int M = paddedLength(g.n_det);
    const std::vector<float> H = buildRampResponse(M, g.ds(), win);
    std::vector<std::complex<float>> buf(M);

    for (int a = 0; a < g.n_angle; ++a) {
        // 行末补零到 M
        for (int k = 0; k < g.n_det; ++k) buf[k] = std::complex<float>(sino[(size_t)a * g.n_det + k], 0.f);
        for (int k = g.n_det; k < M; ++k) buf[k] = std::complex<float>(0.f, 0.f);

        fftInPlace(buf.data(), M, /*inverse=*/false);
        for (int m = 0; m < M; ++m) buf[m] *= H[m];
        fftInPlace(buf.data(), M, /*inverse=*/true);

        for (int k = 0; k < g.n_det; ++k) out[(size_t)a * g.n_det + k] = buf[k].real();
    }
}
