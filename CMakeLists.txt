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

# GPU offload dispatch logic (portable: builds on any host). Run against both
# copies of ggml-gpu-offload-integration.h so the two cannot drift apart
# untested.
foreach(variant IN ITEMS ggml powerpc)
    if(variant STREQUAL "ggml")
        set(_offload_dir ${POWERPC_INCLUDE_DIR})
    else()
        set(_offload_dir ${POWERPC_COMPAT_DIR})
    endif()
    add_executable(test_gpu_offload_integration_${variant} tests/test_gpu_offload_integration.c)
    target_include_directories(test_gpu_offload_integration_${variant} PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/stubs
        ${_offload_dir}
    )
    add_test(NAME gpu_offload_integration_${variant} COMMAND test_gpu_offload_integration_${variant})
endforeach()

# power8-compat.h shims (need a VSX-capable target, e.g. the ppc64le cross build).
if(HAS_ALTIVEC)
    foreach(variant IN ITEMS ggml powerpc)
        if(variant STREQUAL "ggml")
            set(_compat_dir ${POWERPC_INCLUDE_DIR})
        else()
            set(_compat_dir ${POWERPC_COMPAT_DIR})
        endif()
        add_executable(test_power8_compat_${variant} tests/test_power8_compat.c)
        target_include_directories(test_power8_compat_${variant} PRIVATE ${_compat_dir})
        target_compile_options(test_power8_compat_${variant} PRIVATE -Wall -Werror)
        add_test(NAME power8_compat_${variant} COMMAND test_power8_compat_${variant})
    endforeach()
endif()

# Header self-containment: every standalone header must compile as the first
# include of a translation unit (catches missing <stdint.h>/<stddef.h> etc.).
# ggml-pse-integration.h is excluded: it pulls in PSE headers that live only
# in the full llama.cpp POWER8 tree, not in this repository.
set(_standalone_headers
    ggml-dcbt-resident.h
    ggml-gpu-offload-integration.h
    ggml-mass-integration.h
    ggml-sparse-softmax.h
    power8-compat.h
    pse-entropy-burst.h
)
set(_selfcontained_srcs)
foreach(dir_var IN ITEMS POWERPC_INCLUDE_DIR POWERPC_COMPAT_DIR)
    foreach(hdr IN LISTS _standalone_headers)
        string(MAKE_C_IDENTIFIER "${dir_var}_${hdr}" _id)
        set(_tu ${CMAKE_BINARY_DIR}/selfcontained/${_id}.c)
        file(WRITE ${_tu} "#include \"${${dir_var}}/${hdr}\"\nint ${_id}_ok;\n")
        list(APPEND _selfcontained_srcs ${_tu})
    endforeach()
endforeach()
add_library(header_selfcontained OBJECT ${_selfcontained_srcs})
target_include_directories(header_selfcontained PRIVATE ${CMAKE_SOURCE_DIR}/tests/stubs)
target_compile_options(header_selfcontained PRIVATE -Wall -Werror)
