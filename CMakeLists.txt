cmake_minimum_required(VERSION 3.14)
project(llama-cpp-power8 C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# PowerPC include paths
set(POWERPC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu/arch/powerpc)
set(POWERPC_COMPAT_DIR ${CMAKE_SOURCE_DIR}/powerpc)

# AltiVec benchmark (standalone, fully self-contained)
add_executable(altivec_bench altivec_benchmark.c)
target_link_libraries(altivec_bench m)

# Header validation library (compile-checks the PowerPC headers)
add_library(powerpc_headers INTERFACE)
target_include_directories(powerpc_headers INTERFACE
    ${POWERPC_INCLUDE_DIR}
    ${POWERPC_COMPAT_DIR}
)
