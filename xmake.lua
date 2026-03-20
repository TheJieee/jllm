add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("src/common")
add_includedirs("src")

add_requires("openmp")
add_requires("pybind11")

if is_mode("release") then
    set_optimize("fastest")
end
add_vectorexts("avx2", "fma")

-- ============ Compiler Flags ============
local function add_compiler_flags()
    if is_plat("linux", "macosx") then
        add_cxflags("-mavx2", "-mfma", "-fopenmp", "-march=native", "-fPIC")
        
        if is_mode("release") then
            add_cxflags("-flto", "-funroll-loops", "-finline-functions", "-ffast-math")
        else
            add_cxflags("-O0", "-g")
        end
        
    elseif is_plat("windows") then
        if is_mode("release") then
            add_cxflags("/arch:AVX2", "/fp:fast", "/Ot")
        else
            add_cxflags("/Od", "/Zi")
        end
    end
end

-- ============ Static Library: tensor ============
target("tensor")
    set_kind("static")
    set_languages("c++20")
    
    add_files("src/common/*.cpp")
    add_files("src/tensor/*.cpp")
    add_files("src/utils/*.cpp")
    add_files("src/log/*.cpp")
    add_files("src/kernel/allocator/*.cpp")
    
    add_packages("openmp")
    add_compiler_flags()
    
    set_targetdir("$(builddir)/lib")
    set_objectdir("$(builddir)/obj")

-- ============ Static Library: model_runner ============
target("model_runner")
    set_kind("static")
    set_languages("c++20")
    
    add_files("src/model_runner/layer/*.cpp")
    add_files("src/model_runner/layer/kernel/*/*.cpp")
    add_files("src/model_runner/attention/*.cpp")
    add_files("src/model_runner/attention/*/*.cpp")
    add_files("src/model_runner/models/*.cpp")
    
    add_deps("tensor")
    add_packages("openmp")
    add_packages("pybind11")
    add_compiler_flags()
    
    set_targetdir("$(builddir)/lib")
    set_objectdir("$(builddir)/obj")

-- ============ Static Library: worker ============
target("worker")
    set_kind("static")
    set_languages("c++20")
    
    add_files("src/executor/worker/*.cpp")
    add_files("src/sampling/*.cpp")
    add_files("src/sampling/kernel/cpu/*.cpp")
    add_files("src/kv_cache_manager/kv_cache/*.cpp")
    
    add_deps("tensor", "model_runner")
    add_packages("openmp")
    add_packages("pybind11")
    add_compiler_flags()
    
    set_targetdir("$(builddir)/lib")
    set_objectdir("$(builddir)/obj")

-- ============ Static Library: executor ============
target("executor")
    set_kind("static")
    set_languages("c++20")
    
    add_files("src/executor/*.cpp")
    
    add_deps("tensor", "model_runner", "worker")
    add_packages("openmp")
    add_compiler_flags()
    
    set_targetdir("$(builddir)/lib")
    set_objectdir("$(builddir)/obj")

-- ============ Static Library: scheduler ============
target("scheduler")
    set_kind("static")
    set_languages("c++20")
    
    add_files("src/scheduler/*.cpp")
    add_files("src/kv_cache_manager/block_manager/*.cpp")
    
    add_deps("tensor", "model_runner", "executor")
    add_packages("openmp")
    add_compiler_flags()
    
    set_targetdir("$(builddir)/lib")
    set_objectdir("$(builddir)/obj")

-- ============ Shared Library: jllm_engine (Python Binding) ============
target("jllm_engine")
    set_kind("shared")
    set_languages("c++20")
    
    add_files("src/engine/engine.cpp")

    -- Python binding source file (to be created)
    add_files("src/engine/python_binding.cpp")
    
    -- Link all static libraries
    add_deps("executor", "scheduler")
    add_packages("openmp")
    add_packages("pybind11")
    add_compiler_flags()
    
    set_targetdir("$(builddir)/lib")
    set_objectdir("$(builddir)/obj")
    
    -- Set output name for Python
    if is_plat("linux") then
        set_filename("jllm_engine.so")
    elseif is_plat("macosx") then
        set_filename("jllm_engine.dylib")
    elseif is_plat("windows") then
        set_filename("jllm_engine.pyd")
    end

    after_build(function (target)
        import("core.project.project")
        -- 自动将生成的二进制拷贝到 python/ 目录下
        os.cp(target:targetfile(), "$(projectdir)/python/jllm/")
    end)

-- ============ BF16 Benchmark ============
target("benchmark_bf16_swiglu")
    set_kind("binary")
    set_languages("c++20")
    set_default(false)  -- Not built by default
    
    add_files("tests/benchmark_gate_up_swiglu_bf16.cpp")
    add_deps("tensor", "model_runner")
    add_packages("openmp")
    add_compiler_flags()
    
    set_targetdir("$(builddir)/bin")
    set_objectdir("$(builddir)/obj")


target("debug_main")
    set_kind("binary")
    set_languages("c++20")
    set_default(false)  -- Not built by default
     set_rundir("$(projectdir)") 

    add_files("tests/debug_main.cpp")
    add_deps("jllm_engine")

    set_targetdir("$(builddir)/bin")
    set_objectdir("$(builddir)/obj")