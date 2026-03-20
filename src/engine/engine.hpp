#pragma once

#include "executor/executor.hpp"
#include "scheduler/scheduler.hpp"


#include <memory>
#include <future>
#include <thread>

JLLM_BEGIN

struct Request {
    std::vector<int64_t> prompt;
    int64_t request_id;
};

struct Config {
    size_t cache_num_block;
    size_t cache_block_size;
};

class Engine {
public:
    Engine();
    std::vector<size_t> generate(const Request& req);
    std::string model_path();
    void step();
protected:
    std::unique_ptr<Scheduler> m_scheduler;
    std::unique_ptr<Executor>  m_executor;
    std::map<size_t, std::queue<int64_t>> m_output;
    std::string m_model_path;
    Config m_config;
    bool m_has_output = false;
};

class asycEngine :public Engine {
public:
    asycEngine() = default;
    ~asycEngine();
    void push(const Request& req);
    void set_up();
    bool has_output();
    std::vector<std::pair<size_t, std::vector<int64_t>>> get_all();
private:
    std::jthread m_work_thread;
    std::condition_variable m_cv;
    std::mutex m_mutex;
    bool m_shut_down = false;
};

JLLM_END