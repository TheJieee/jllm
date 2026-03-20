#pragma once

#include "../storage/storage.hpp"

namespace jllm {
class Allocator {
public:
    static Allocator& instance() {
        static Allocator allocator;
        return allocator;
    }

    static storage_t allocate_mem(size_t size);

    static storage_t from_file(const char* filename);

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
private:
    Allocator() = default;
};
} // namespace jllm