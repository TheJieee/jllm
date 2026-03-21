#include "engine.hpp"

#include "log/log.hpp"

#include "json.hpp"
#include "utils.hpp"
#include <fstream>
#include <thread>

JLLM_BEGIN

using json = nlohmann::json;

Engine::Engine()
{
    spdlog::info("reading configuration...\n");
    std::ifstream file("jllm_config.json");
    if(!file.is_open()){
        spdlog::error("fail to open jllm_config.json.\n");
        exit(-1);
    }
    json config = json::parse(file);

    m_config.cache_num_block = config["kv_cache"]["num_blocks"];
    m_config.cache_block_size = config["kv_cache"]["block_size"];
    spdlog::info("cache block size: {}\n", m_config.cache_block_size);
    m_model_path = config["model_path"];
    spdlog::info("model path: {}\n", m_model_path);

    spdlog::info("Initializing executor...\n");
    m_executor = std::make_unique<Executor>(
        config["model_type"], config["model_path"],
        config["kv_cache"]["block_size"], config["kv_cache"]["num_blocks"]
    );
    spdlog::info("Executor OK.\n");

    auto& meta = m_executor->kv_cache_shape();
    m_scheduler = std::make_unique<Scheduler>(
        meta[0], meta[2]
    );
}

std::vector<size_t> Engine::generate(const Request &req)
{
    Sequence seq(req.request_id, req.prompt, m_config.cache_block_size);
    m_scheduler->push(seq);
    while(!m_scheduler->is_free()) {
        step();
    }
    auto& q = m_output[seq.id()];
    std::vector<size_t> ret; ret.reserve(q.size());
    while(!q.empty()) {
        ret.push_back(q.front());
        q.pop();
    }
    m_output.erase(seq.id());
    return ret;
}

std::string Engine::model_path()
{
    return m_model_path;
}

void Engine::step()
{
    m_scheduler->update();
    auto seqs = m_scheduler->schedule();
    if(!seqs.empty()){
        m_executor->send(seqs);
        for(auto& seq : seqs) {
            if(seq->prefill_finished()) {
                m_output[seq->id()].push(seq->get_last_token());
                m_has_output = true;
            }
        }
    }
}

asycEngine::~asycEngine()
{
    {
        std::lock_guard lock(m_mutex);
        m_shut_down = true;
    }
    m_cv.notify_all();
}

void asycEngine::push(const Request &req)
{
    Sequence seq(req.request_id, req.prompt, m_config.cache_block_size);
    m_scheduler->push(std::move(seq));
    m_cv.notify_one();
}

void asycEngine::set_up()
{
    m_work_thread = std::jthread([&](){
        while(true) {
            std::unique_lock lock(m_mutex);
            m_cv.wait(lock, [&](){ return !m_scheduler->is_free() || m_shut_down; });
            if (m_shut_down) break;
            while (!m_scheduler->is_free()) {

                lock.unlock(); 
                step(); 
                lock.lock();

                if (m_shut_down) break;
            }
        }
    });
}

bool asycEngine::has_output()
{
    return m_has_output;
}

std::vector<std::pair<size_t, std::vector<int64_t>>> asycEngine::get_all()
{
    std::vector<std::pair<size_t, std::vector<int64_t>>> ret;
    for(auto& pair : m_output) {
        if (!pair.second.empty()) {
            auto& q = pair.second;
            std::vector<int64_t> tmp; tmp.reserve(q.size());
            do {
                tmp.push_back(q.front());
                q.pop();
            } while (!q.empty());
            ret.emplace_back(pair.first, std::move(tmp));
        }
    }
    std::lock_guard lock(m_mutex);
    m_has_output = false;
    return ret;
}

JLLM_END