#pragma once

#include "../kernel/kernel.h"
#include <vector>

JLLM_BEGIN
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;

struct TensorMeta {
    jllmDataType_t dtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
};

class Tensor {
public:
    Tensor(
        const std::vector<size_t>& shape,
        jllmDataType_t dtype,
        storage_t storage,
        size_t offset = 0
    );
    static tensor_t create(
        const std::vector<size_t> &shape,
        jllmDataType_t dtype,
        jllmDeviceType_t device_type = jllmDeviceType_t::CPU,
        int device = 0
    );
    
    ~Tensor() = default;
    // Info
    std::byte *data();
    const std::byte *data() const;
    size_t ndim() const;
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    jllmDataType_t dtype() const;
    jllmDeviceType_t deviceType() const;
    int deviceId() const;
    size_t numel() const;
    size_t elementSize() const;

    std::string info() const;
    void debug(const std::string& message) const;
    void debug() const;

    bool isContiguous() const;

    // Meta Transform
    tensor_t permute(const std::vector<size_t> &order) const;
    tensor_t slice(size_t dim, size_t start, size_t end) const;
    tensor_t view(const std::vector<size_t> &shape) const;

    // Load data from host memory
    void load(const void *src);

    // Challenging features
    tensor_t contiguous() const;
    tensor_t reshape(const std::vector<size_t> &shape) const;
    tensor_t to(jllmDeviceType_t device_type, int device = -1) const;
private:
    TensorMeta m_meta;
    storage_t m_storage;
    size_t m_offset;
    Tensor(TensorMeta meta, storage_t storage, size_t offset = 0)
    : m_meta(std::move(meta)), m_storage(std::move(storage)), m_offset(offset) {}

};

JLLM_END // namespace jllm