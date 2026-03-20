#include "tensor.hpp"

#include "../kernel/kernel.h"
#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

JLLM_BEGIN
Tensor::Tensor(
    const std::vector<size_t>& shape,
    jllmDataType_t dtype,
    storage_t storage,
    size_t offset
) {
    m_meta.dtype = dtype;
    m_meta.shape = shape;
    m_meta.strides.resize(shape.size());
    size_t stride = 1;
    for (ptrdiff_t i = shape.size() - 1; i >= 0; --i) {
        m_meta.strides[i] = stride;
        stride *= shape[i];
    }
    m_storage = storage;
    m_offset = offset;
}

tensor_t Tensor::create(
    const std::vector<size_t> &shape,
    jllmDataType_t dtype,
    jllmDeviceType_t device_type,
    int device
) {
    size_t numel = 1;
    for (size_t dim : shape) {
        numel *= dim;
    }
    size_t element_size = utils::dsize(dtype);
    size_t total_size = numel * element_size;
    storage_t storage = Allocator::allocate_mem(total_size);
    return std::make_shared<Tensor>(shape, dtype, storage);
}

std::byte *Tensor::data() {
    return m_storage->memory() + m_offset;
}

const std::byte *Tensor::data() const {
    return m_storage->memory() + m_offset;
}

size_t Tensor::ndim() const {
    return m_meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return m_meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return m_meta.strides;
}

jllmDataType_t Tensor::dtype() const {
    return m_meta.dtype;
}

jllmDeviceType_t Tensor::deviceType() const {
    return m_storage->deviceType();
}

int Tensor::deviceId() const {
    return m_storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(m_meta.shape.begin(), m_meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(m_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << static_cast<int>(this->dtype());

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, jllmDataType_t dtype) {
    switch (dtype) {
    case jllmDataType_t::BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case jllmDataType_t::BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case jllmDataType_t::I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case jllmDataType_t::I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case jllmDataType_t::I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case jllmDataType_t::I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case jllmDataType_t::U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case jllmDataType_t::U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case jllmDataType_t::U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case jllmDataType_t::U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case jllmDataType_t::F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case jllmDataType_t::F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case jllmDataType_t::F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case jllmDataType_t::BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug(const std::string& message) const {
    std::cout << message << ":\n";
    this->debug();
}

void Tensor::debug() const
{
    std::cout << this->info() << std::endl;
    debug_print(this->data(), {this->shape().back()}, {1}, this->dtype());
}

bool Tensor::isContiguous() const {
    auto& shape = this->shape();
    auto& strides = this->strides();
    size_t ndim = shape.size();
    
    bool ret = true;
    if(ndim != 0) {
        ptrdiff_t expected_stride = 1;
        for(int i = static_cast<int>(ndim - 1); i >= 0; i--) {
            if(strides[i] != expected_stride) {
                ret = false;
                break;
            }
            expected_stride *= static_cast<ptrdiff_t>(shape[i]);
        }
    }
    return ret;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    auto meta = m_meta;
    for(size_t i = 0; i < order.size(); i++) {
        CHECK_ARGUMENT(order[i] < order.size(), "Permute order out of range");
        meta.shape[i] = m_meta.shape[order[i]];
        meta.strides[i] = m_meta.strides[order[i]];
    }
    return std::shared_ptr<Tensor>(new Tensor(meta, m_storage, m_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for (auto s : shape) {
        new_numel *= s;
    }
    if (new_numel != this->numel()) {
        EXCEPTION_SHAPE_MISMATCH;
    }
    if (!this->isContiguous()) {
        std::cerr << "[ERROR] View error" << EXCEPTION_LOCATION_MSG << std::endl;
        throw std::runtime_error("View on non-contiguous tensor, call contiguous() first.");
    }
    
    std::vector<ptrdiff_t> new_strides(shape.size());
    size_t stride = 1;
    for (size_t i = 1; i <= shape.size(); i++) {
        new_strides[shape.size() - i] = stride;
        stride *= shape[shape.size() - i];
    }
    auto new_meta = TensorMeta{
        this->dtype(),
        shape,
        new_strides,
    };
    return tensor_t(new Tensor(new_meta, m_storage, m_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < this->ndim(), "Slice dimension out of range");
    CHECK_ARGUMENT(start <= end && end <= this->shape()[dim], "Slice indices out of range");
    
    TensorMeta meta = m_meta;
    meta.shape[dim] = end - start;
    size_t offset = m_offset + start * m_meta.strides[dim] * this->elementSize();
    return tensor_t(new Tensor(meta, m_storage, offset));
}

void Tensor::load(const void *src_) {
    size_t total_bytes = this->numel() * this->elementSize();
    std::memcpy(this->data(), src_, total_bytes);
}

tensor_t Tensor::contiguous() const {
    if(this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(m_meta, m_storage, m_offset));
    }
    auto new_tensor = Tensor::create(shape(), dtype(), deviceType(), deviceId());
    //size_t total_bytes = this->numel() * this->elementSize();

    TO_BE_IMPLEMENTED();
    return new_tensor;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(m_meta, m_storage));
}

tensor_t Tensor::to(jllmDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(m_meta, m_storage));
}

JLLM_END