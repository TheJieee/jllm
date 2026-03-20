#include "allocator.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>


namespace jllm {
storage_t Allocator::allocate_mem(size_t size) {
    void* data = std::malloc(size);
    if (!data) {
        throw std::bad_alloc();
    }
    return storage_t(new Storage(data, [data]() { std::free(data); }));
}

storage_t Allocator::from_file(const char* filename) {
        int fd = open(filename, O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file");
        }
        struct stat st;
        fstat(fd, &st);
        void* data = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map file");
        }
        close(fd);
        return storage_t(new Storage(data, [data, fd, st]() {
            int sus = munmap(data, st.st_size);
            if(sus == -1) {
                throw std::runtime_error("Failed to unmap file");
            }
        }));
}
} // namespace jllm