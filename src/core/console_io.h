#pragma once

namespace ct {

// 统一 stdio 的输出编码，避免中文在 Windows 上的乱码。
//
// 背景：源码按 /utf-8 编译，printf 写出的是 UTF-8 字节；而 Windows 上不同宿主
// 会按自己的代码页解释这些字节：
//   - 控制台（cmd / Windows Terminal）：conhost 默认按系统 ANSI 页（中文为 GBK）
//     解码，且对伪终端（ConPTY）而言 SetConsoleOutputCP 并不总是生效；
//   - 管道/文件（VS Code 调试控制台、CMake Tools 输出面板、旧版日志工具）：
//     多数按系统 ANSI 页解码，UTF-8 字节会被显示成“閰嶇疆”这类乱码。
//
// 处理方式：把 stdout/stderr 重定向到内部管道，后台线程按行读取后转码：
//   - 控制台：UTF-8 → UTF-16 → WriteConsoleW（Unicode 控制台 API，不受代码页影响）；
//   - 管道/文件：UTF-8 → 系统 ANSI 代码页（GetACP，中文 Windows 即 GBK）。
// 进程退出时（atexit）先 flush，再写退出哨兵通知后台线程收尾，保证最后一行不丢。
//
// 设环境变量 CT_OUTPUT_ENCODING=utf8 可关闭转码、保持 UTF-8 字节
// （适合 CI 日志、按 UTF-8 解析的脚本）。
//
// 必须在 main() 的第一处输出之前调用一次。
void initConsoleOutput();

}  // namespace ct
