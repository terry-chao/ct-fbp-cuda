// 控制台/管道输出编码适配（实现说明见 console_io.h）。
#include "core/console_io.h"

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <io.h>
#include <fcntl.h>  // _O_BINARY

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {

// 退出哨兵：写在最后一行输出之后，后台线程读到它即收尾。
// 正常文本里不会出现 0x01/0x02，因此不会与内容冲突。
constexpr char kEofMarker[] = "\x01\x02CT-EOF\x02\x01";
constexpr size_t kEofLen = sizeof(kEofMarker) - 1;

bool g_active = false;   // 是否已接管输出（atexit 用）
bool g_console = false;  // 真实输出是控制台（走 WriteConsoleW）还是管道/文件
int g_pipeRead = -1;     // 内部管道的读端
HANDLE g_sink = INVALID_HANDLE_VALUE;  // 真正要写入的句柄
std::thread g_pump;

std::wstring utf8ToWide(const char* data, size_t size) {
    if (size == 0) return {};
    const int need = MultiByteToWideChar(CP_UTF8, 0, data, static_cast<int>(size), nullptr, 0);
    if (need <= 0) return {};
    std::wstring w(static_cast<size_t>(need), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, data, static_cast<int>(size), w.data(), need);
    return w;
}

std::string wideToAnsi(const std::wstring& w) {
    if (w.empty()) return {};
    const int need = WideCharToMultiByte(CP_ACP, 0, w.data(), static_cast<int>(w.size()), nullptr, 0,
                                         nullptr, nullptr);
    if (need <= 0) return {};
    std::string s(static_cast<size_t>(need), '\0');
    WideCharToMultiByte(CP_ACP, 0, w.data(), static_cast<int>(w.size()), s.data(), need, nullptr,
                        nullptr);
    return s;
}

void writeRaw(const char* data, size_t size) {
    size_t off = 0;
    while (off < size) {
        const DWORD chunk = static_cast<DWORD>((size - off) > 0x8000 ? 0x8000 : (size - off));
        DWORD written = 0;
        if (!WriteFile(g_sink, data + off, chunk, &written, nullptr) || written == 0) return;
        off += written;
    }
}

// 转码并写出，utf8 必须是完整的一行（或完整的 UTF-8 片段）
void emit(const std::string& utf8) {
    if (utf8.empty()) return;
    const std::wstring w = utf8ToWide(utf8.data(), utf8.size());
    if (w.empty()) {  // 非法 UTF-8：原样写出，至少不丢输出
        writeRaw(utf8.data(), utf8.size());
        return;
    }
    if (g_console) {
        size_t off = 0;
        while (off < w.size()) {
            const DWORD chunk =
                static_cast<DWORD>((w.size() - off) > 0x2000 ? 0x2000 : (w.size() - off));
            DWORD written = 0;
            if (!WriteConsoleW(g_sink, w.data() + off, chunk, &written, nullptr) || written == 0) {
                writeRaw(utf8.data(), utf8.size());
                return;
            }
            off += written;
        }
        return;
    }
    const std::string ansi = wideToAnsi(w);
    if (ansi.empty()) writeRaw(utf8.data(), utf8.size());
    else writeRaw(ansi.data(), ansi.size());
}

// 后台线程：按行切分。UTF-8 的多字节序列不会包含 '\n'，
// 因此“整行转码”不会把字符截断在两个数据块之间。
void pump() {
    std::string pending;
    char buf[8192];
    for (;;) {
        const int n = _read(g_pipeRead, buf, static_cast<unsigned>(sizeof buf));
        if (n <= 0) break;
        pending.append(buf, static_cast<size_t>(n));

        const size_t mark = pending.find(kEofMarker);
        if (mark != std::string::npos) {
            emit(pending.substr(0, mark));
            return;
        }
        size_t nl = 0;
        while ((nl = pending.find('\n')) != std::string::npos) {
            emit(pending.substr(0, nl + 1));
            pending.erase(0, nl + 1);
        }
    }
    emit(pending);  // 进程异常退出（未写哨兵）时的兜底
}

void shutdown() {
    if (!g_active) return;
    g_active = false;
    std::fflush(stdout);
    std::fflush(stderr);
    _write(1, kEofMarker, static_cast<unsigned>(kEofLen));  // 绕过 stdio，直接写管道写端
    if (g_pump.joinable()) g_pump.join();
}

}  // namespace

void ct::initConsoleOutput() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    const char* forced = std::getenv("CT_OUTPUT_ENCODING");
    if (forced != nullptr && std::strcmp(forced, "utf8") == 0) return;  // 保持 UTF-8 字节

    const int fd = _fileno(stdout);
    if (fd < 0) return;
    HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    DWORD mode = 0;
    const bool isConsole = (h != INVALID_HANDLE_VALUE) && GetConsoleMode(h, &mode) != 0;

    // stderr 与 stdout 的宿主类型一致时才一起接管（避免把 2>file 重定向弄丢）
    const int efd = _fileno(stderr);
    bool sameKind = false;
    if (efd >= 0) {
        HANDLE eh = reinterpret_cast<HANDLE>(_get_osfhandle(efd));
        DWORD emode = 0;
        sameKind = (eh != INVALID_HANDLE_VALUE) && ((GetConsoleMode(eh, &emode) != 0) == isConsole);
    }

    int pfd[2] = {-1, -1};
    if (_pipe(pfd, 8192, _O_BINARY) != 0) return;
    const int saved = _dup(fd);
    if (saved < 0 || _dup2(pfd[1], fd) != 0 || (sameKind && _dup2(pfd[1], efd) != 0)) {
        _close(pfd[0]);
        _close(pfd[1]);
        return;
    }
    _close(pfd[1]);

    g_console = isConsole;
    g_sink = reinterpret_cast<HANDLE>(_get_osfhandle(saved));
    g_pipeRead = pfd[0];
    // 输出本来就是逐条 printf，关掉缓冲可以让长任务（如大尺寸 FBP）实时看到进度
    setvbuf(stdout, nullptr, _IONBF, 0);
    g_active = true;
    g_pump = std::thread(pump);
    std::atexit(shutdown);
}

#else

void ct::initConsoleOutput() {}

#endif
