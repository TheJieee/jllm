#include "sysInfo.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#include <fstream>
#include <string>
#endif

JLLM_BEGIN
namespace utils {
sysUtils::sysUtils() {
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        m_info.available_memory = statex.ullAvailPhys * 1024;
    }
#else
    // Linux 实现：读取 /proc/meminfo 获取更准确的 Available
    std::ifstream file("/proc/meminfo");
    std::string line;
    unsigned long long mem_total = 0, mem_avail = 0;
    while (std::getline(file, line)) {
        if (line.compare(0, 13, "MemAvailable:") == 0) {
            sscanf(line.c_str(), "MemAvailable: %llu", &mem_avail);
            break; 
        }
    }
    m_info.available_memory = mem_avail * 1024; // 转换为字节
#endif
}
}
JLLM_END