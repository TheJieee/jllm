#pragma once

#include <cstdint>
#include <cstddef>

#define JLLM_BEGIN namespace jllm {
#define JLLM_END } 

enum class jllmDeviceType_t {
    CPU,
};

enum class jllmDataType_t {
    INVALID,
    BOOL,
    BYTE,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F8,
    F16,
    BF16,
    F32,
    F64,
    C16,
    C32,
    C64,
    C128,
};