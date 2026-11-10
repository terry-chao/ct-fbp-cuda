#include "ct/io/io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

bool writePGMRanged(const std::string& path, const float* img, int w, int h,
                    float vmin, float vmax, const std::string& comment) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "writePGM: cannot open '%s' for writing\n", path.c_str());
        return false;
    }
    std::fprintf(f, "P5\n# %s\n%d %d\n255\n", comment.c_str(), w, h);

    float range = vmax - vmin;
    std::vector<unsigned char> buf((size_t)w * h);
    for (size_t i = 0; i < buf.size(); ++i) {
        float t = range > 0.f ? (img[i] - vmin) / range : 0.5f;
        t = std::min(1.f, std::max(0.f, t));
        buf[i] = (unsigned char)(t * 255.0f + 0.5f);
    }
    size_t written = std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (written != buf.size()) {
        std::fprintf(stderr, "writePGM: short write to '%s'\n", path.c_str());
        return false;
    }
    return true;
}

bool writePGM(const std::string& path, const float* img, int w, int h,
              const std::string& comment) {
    float vmin = img[0], vmax = img[0];
    for (size_t i = 1; i < (size_t)w * h; ++i) {
        vmin = std::min(vmin, img[i]);
        vmax = std::max(vmax, img[i]);
    }
    return writePGMRanged(path, img, w, h, vmin, vmax, comment);
}

bool readPGM(const std::string& path, std::vector<float>& img, int& w, int& h) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "readPGM: cannot open '%s'\n", path.c_str());
        return false;
    }
    char magic[3] = {0, 0, 0};
    if (std::fscanf(f, "%2s", magic) != 1 || (std::strcmp(magic, "P5") != 0 && std::strcmp(magic, "P2") != 0)) {
        std::fprintf(stderr, "readPGM: '%s' is not a PGM (P5/P2) file\n", path.c_str());
        std::fclose(f);
        return false;
    }
    auto nextInt = [&](int& v) -> bool {
        char c;
        // 跳过空白与 '#' 注释行
        for (;;) {
            c = (char)std::fgetc(f);
            if (c == '#') {
                while (c != '\n' && c != EOF) c = (char)std::fgetc(f);
            } else if (!std::isspace((unsigned char)c)) {
                break;
            }
            if (c == EOF) return false;
        }
        std::ungetc(c, f);
        return std::fscanf(f, "%d", &v) == 1;
    };
    int maxval = 255;
    if (!nextInt(w) || !nextInt(h) || !nextInt(maxval) || w <= 0 || h <= 0 || maxval <= 0 || maxval > 65535) {
        std::fprintf(stderr, "readPGM: bad header in '%s'\n", path.c_str());
        std::fclose(f);
        return false;
    }
    img.resize((size_t)w * h);
    if (std::strcmp(magic, "P5") == 0) {
        std::fgetc(f);  // 头部后恰好一个空白字符
        if (maxval < 256) {
            std::vector<unsigned char> raw(img.size());
            if (std::fread(raw.data(), 1, raw.size(), f) != raw.size()) {
                std::fprintf(stderr, "readPGM: truncated pixel data in '%s'\n", path.c_str());
                std::fclose(f);
                return false;
            }
            for (size_t i = 0; i < img.size(); ++i) img[i] = (float)raw[i] / maxval;
        } else {
            std::vector<unsigned short> raw(img.size());
            if (std::fread(raw.data(), 2, raw.size(), f) != raw.size()) {
                std::fprintf(stderr, "readPGM: truncated pixel data in '%s'\n", path.c_str());
                std::fclose(f);
                return false;
            }
            for (size_t i = 0; i < img.size(); ++i) img[i] = (float)raw[i] / maxval;
        }
    } else {  // P2 文本
        for (size_t i = 0; i < img.size(); ++i) {
            int v;
            if (!nextInt(v)) {
                std::fprintf(stderr, "readPGM: truncated pixel data in '%s'\n", path.c_str());
                std::fclose(f);
                return false;
            }
            img[i] = (float)v / maxval;
        }
    }
    std::fclose(f);
    return true;
}
