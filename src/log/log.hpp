#pragma once

#include <string>
#include <sstream>
#include <iostream>

namespace spdlog {
template<typename T>
std::string to_str(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

// 递归终点
inline void format_impl(std::string& res, const char* s) {
    res += s; // 把剩下的字符串接上去
}

// 变分模板递归处理
template<typename T, typename... Args>
void format_impl(std::string& res, const char* s, T value, Args... args) {
    while (*s) {
        if (*s == '{' && *(s + 1) == '}') {
            res += to_str(value); // 替换占位符
            format_impl(res, s + 2, args...); // 递归处理剩下的
            return;
        }
        res += *s++;
    }
}

// 对外接口
template<typename... Args>
std::string my_format(const char* fmt, Args... args) {
    std::string res;
    format_impl(res, fmt, args...);
    return res;
}

template<typename... Args>
void info(const char* fmt, Args... args) {
    std::cout << my_format(fmt, args...);
}

template<typename... Args>
void error(const char* fmt, Args... args) {
    std::cerr << my_format(fmt, args...);
    exit(-1);
}

}