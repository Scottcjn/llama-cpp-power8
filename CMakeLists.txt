cmake_minimum_required(VERSION 3.14)
project(llama-cpp-power8 C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# PowerPC include paths
set(POWERPC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu/arch/powerpc)
set(POWERPC_COMPAT_DIR ${CMAKE_SOURCE_DIR}/powerpc)

# AltiVec benchmark (standalone, fully self-contained)
include(CheckCSourceCompiles)
check_c_source_compiles("#include <altivec.h>\nint main(){return 0;}" HAS_ALTIVEC)
if(HAS_ALTIVEC)
    add_executable(altivec_bench altivec_benchmark.c)
    target_link_libraries(altivec_bench m)
endif()

# Header validation library (compile-checks the PowerPC headers)
add_library(powerpc_headers INTERFACE)
target_include_directories(powerpc_headers INTERFACE
    ${POWERPC_INCLUDE_DIR}
    ${POWERPC_COMPAT_DIR}
)

# ---------------------------------------------------------------------------
# Tests (run with `ctest --output-on-failure` from the build directory).
# When cross-compiling for ppc64le, pass
#   -DCMAKE_CROSSCOMPILING_EMULATOR="qemu-ppc64le;-L;/usr/powerpc64le-linux-gnu"
# so ctest can execute the target binaries under user-mode QEMU.
# ---------------------------------------------------------------------------
enable_testing()

# GPU offload dispatch logic (portable: builds on any host).
add_executable(test_gpu_offload_integration tests/test_gpu_offload_integration.c)
target_include_directories(test_gpu_offload_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/stubs
    ${POWERPC_INCLUDE_DIR}
)
add_test(NAME gpu_offload_integration COMMAND test_gpu_offload_integration)
