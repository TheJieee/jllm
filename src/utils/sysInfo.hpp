#pragma once
#include "jllm.h"


JLLM_BEGIN
namespace utils {

struct SysInfo {
    size_t available_memory;
};

class sysUtils {
public:
    static sysUtils& instance() {
        static sysUtils utils;
        return utils;
    }

    static size_t get_available_memory() {
        return instance().m_info.available_memory;
    }

    sysUtils(const sysUtils&) = delete;
    sysUtils& operator=(const sysUtils&) = delete;
    sysUtils(sysUtils&&) = delete;
    sysUtils& operator=(sysUtils&&) = delete;
private:
    sysUtils();
    SysInfo m_info;
};

}
JLLM_END