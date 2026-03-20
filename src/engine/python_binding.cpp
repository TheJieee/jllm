#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine.hpp"

namespace py = pybind11;
using namespace jllm;

namespace jllm {
void bind_Qwen2(py::module_ &);
}

PYBIND11_MODULE(jllm_engine, m) {
    m.doc() = "JLLM Engine - Python bindings for the large language model inference engine";

    // ============ Register model =============
    bind_Qwen2(m);

    // ============ Struct: Request ============
    py::class_<Request>(m, "Request")
        .def(py::init<>())
        .def_readwrite("prompt", &Request::prompt)
        .def_readwrite("request_id", &Request::request_id);

    // ============ Struct: Config ============
    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("cache_num_block", &Config::cache_num_block)
        .def_readwrite("cache_block_size", &Config::cache_block_size);

    // ============ Class: Engine ============
    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def("generate", &Engine::generate, "Generate tokens from a request")
        .def("step", &Engine::step, "Execute one step of inference")
        .def("model_path", &Engine::model_path, "Get model path");

    // ============ Class: asycEngine ============
    py::class_<asycEngine, Engine>(m, "AsyncEngine")
        .def(py::init<>())
        .def("push", &asycEngine::push, "Push a new request to the queue", py::arg("req"))
        .def("set_up", &asycEngine::set_up, "Initialize the async engine")
        .def("has_output", &asycEngine::has_output, "Check if there is output available")
        .def("get_all", &asycEngine::get_all, "Get all output result");
}
