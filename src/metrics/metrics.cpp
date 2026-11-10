#include "ct/metrics/metrics.h"

#include <cmath>

Metrics computeMetrics(const float* ref, const float* test, size_t n) {
    Metrics m{0.0, 0.0, 0.0};
    double se = 0.0;
    float vmin = ref[0], vmax = ref[0];
    for (size_t i = 0; i < n; ++i) {
        double d = (double)test[i] - (double)ref[i];
        se += d * d;
        float ad = (float)std::fabs(d);
        if (ad > m.max_abs) m.max_abs = ad;
        if (ref[i] < vmin) vmin = ref[i];
        if (ref[i] > vmax) vmax = ref[i];
    }
    m.rmse = std::sqrt(se / (double)n);
    double range = (double)vmax - (double)vmin;
    m.psnr = m.rmse > 1e-12 ? 20.0 * std::log10(range / m.rmse) : INFINITY;
    return m;
}
