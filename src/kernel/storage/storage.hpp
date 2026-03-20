#pragma once

#include "jllm.h"

#include <functional>
#include <memory>


namespace jllm {
class Storage {
public:
    using deleter_type = std::function<void()>;

    friend class Allocator;

    std::byte* const memory() { return m_data; }
    jllmDeviceType_t deviceType() const { return m_device; }
    int deviceId() const { return m_device_id; }

    ~Storage() {
        if (m_deleter) {
            m_deleter();
        }
    }
private:
    Storage(void* data, const deleter_type& deleter)
        : m_data(reinterpret_cast<std::byte*>(data)), m_deleter(std::move(deleter)) {};
    std::byte* m_data{nullptr};
    jllmDeviceType_t m_device{jllmDeviceType_t::CPU};
    int m_device_id{0};
    deleter_type m_deleter;
};

using storage_t = std::shared_ptr<Storage>;
} // namespace jllm