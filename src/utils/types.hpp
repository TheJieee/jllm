#include "jllm.h"

#include <iostream>
#include <stdexcept>
#include <cstring>

namespace jllm {
struct CustomFloat16 {
    uint16_t _v;
    bool operator==(const CustomFloat16 &other) const {
        return _v == other._v;
    }
    bool operator!=(const CustomFloat16 &other) const {
        return _v != other._v;
    }
    bool operator<(const CustomFloat16 &other) const;
    bool operator>(const CustomFloat16 &other) const;
    bool operator<=(const CustomFloat16 &other) const;
    bool operator>=(const CustomFloat16 &other) const;
    CustomFloat16 operator+(const CustomFloat16 &other) const;
    CustomFloat16 operator-(const CustomFloat16 &other) const;
    CustomFloat16 operator*(const CustomFloat16 &other) const;
    CustomFloat16 operator/(const CustomFloat16 &other) const;
};
typedef struct CustomFloat16 fp16_t;

struct CustomBFloat16 {
    uint16_t _v;
    bool operator==(const CustomBFloat16 &other) const {
        return _v == other._v;
    }
    bool operator!=(const CustomBFloat16 &other) const {
        return _v != other._v;
    }
    bool operator<(const CustomBFloat16 &other) const;
    bool operator>(const CustomBFloat16 &other) const;
    bool operator<=(const CustomBFloat16 &other) const;
    bool operator>=(const CustomBFloat16 &other) const;
    CustomBFloat16 operator+(const CustomBFloat16 &other) const;
    CustomBFloat16 operator-(const CustomBFloat16 &other) const;
    CustomBFloat16 operator*(const CustomBFloat16 &other) const;
    CustomBFloat16 operator/(const CustomBFloat16 &other) const;
};
typedef struct CustomBFloat16 bf16_t;

namespace utils {
inline size_t dsize(jllmDataType_t dtype) {
    switch (dtype) {
    case jllmDataType_t::BYTE:
        return sizeof(char);
    case jllmDataType_t::BOOL:
        return sizeof(char);
    case jllmDataType_t::I8:
        return sizeof(int8_t);
    case jllmDataType_t::I16:
        return sizeof(int16_t);
    case jllmDataType_t::I32:
        return sizeof(int32_t);
    case jllmDataType_t::I64:
        return sizeof(int64_t);
    case jllmDataType_t::U8:
        return sizeof(uint8_t);
    case jllmDataType_t::U16:
        return sizeof(uint16_t);
    case jllmDataType_t::U32:
        return sizeof(uint32_t);
    case jllmDataType_t::U64:
        return sizeof(uint64_t);
    case jllmDataType_t::F8:
        return 1; // usually 8-bit float (custom)
    case jllmDataType_t::F16:
        return 2; // 16-bit float
    case jllmDataType_t::BF16:
        return 2; // bfloat16
    case jllmDataType_t::F32:
        return sizeof(float);
    case jllmDataType_t::F64:
        return sizeof(double);
    case jllmDataType_t::C16:
        return 2; // 2 bytes complex (not standard)
    case jllmDataType_t::C32:
        return 4; // 4 bytes complex
    case jllmDataType_t::C64:
        return 8; // 8 bytes complex
    case jllmDataType_t::C128:
        return 16; // 16 bytes complex
    case jllmDataType_t::INVALID:
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

inline const char *dtype_to_str(jllmDataType_t dtype) {
    switch (dtype) {
    case jllmDataType_t::BYTE:
        return "byte";
    case jllmDataType_t::BOOL:
        return "bool";
    case jllmDataType_t::I8:
        return "int8";
    case jllmDataType_t::I16:
        return "int16";
    case jllmDataType_t::I32:
        return "int32";
    case jllmDataType_t::I64:
        return "int64";
    case jllmDataType_t::U8:
        return "uint8";
    case jllmDataType_t::U16:
        return "uint16";
    case jllmDataType_t::U32:
        return "uint32";
    case jllmDataType_t::U64:
        return "uint64";
    case jllmDataType_t::F8:
        return "float8";
    case jllmDataType_t::F16:
        return "float16";
    case jllmDataType_t::BF16:
        return "bfloat16";
    case jllmDataType_t::F32:
        return "float32";
    case jllmDataType_t::F64:
        return "float64";
    case jllmDataType_t::C16:
        return "complex16";
    case jllmDataType_t::C32:
        return "complex32";
    case jllmDataType_t::C64:
        return "complex64";
    case jllmDataType_t::C128:
        return "complex128";
    case jllmDataType_t::INVALID:
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

float _f16_to_f32(fp16_t val);
fp16_t _f32_to_f16(float val);

inline float _bf16_to_f32(bf16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;

    return *reinterpret_cast<float*>(&bits32);
}
bf16_t _f32_to_bf16(float val);

template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val) {
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val;
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16(val);
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && std::is_same<TypeTo, float>::value) {
        return _f16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_f16_to_f32(val));
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(val);
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && std::is_same<TypeTo, float>::value) {
        return _bf16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_bf16_to_f32(val));
    } else {
        return static_cast<TypeTo>(val);
    }
}

} // namespace utils
} // namespace llaisys
