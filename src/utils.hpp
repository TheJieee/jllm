#pragma once
#include "utils/check.hpp"
#include "utils/types.hpp"
#include "utils/sysInfo.hpp"
#include <functional>

JLLM_BEGIN
namespace utils {

inline size_t bind_hash(size_t fst, size_t scd) {
    return fst ^ (std::hash<size_t>()(scd) + 0x9e3779b9 + (fst << 6) + (fst >> 2));
}

template<typename It>
size_t vector_hash(It begin, It end) {
    size_t hash = 0;
    for (auto it = begin; it != end; ++it) {
        hash ^= std::hash<typename std::iterator_traits<It>::value_type>()(*it) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}
}
JLLM_END