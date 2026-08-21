cmake_minimum_required(VERSION 3.20)
project(turbo_winfare LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Core engine library sources
set(TURBO_ENGINE_SOURCES
    src/manifest.cpp
    src/packed_experts.cpp
    src/resident_index.cpp
    src/d3d12_context.cpp
    src/pipeline.cpp
    src/streamer.cpp
    src/kv_cache.cpp
    src/tokenizer.cpp
    src/sampling.cpp
    src/detokenizer.cpp
    src/http.cpp
    src/openai_api.cpp
    src/cpu_reference.cpp
    src/runner.cpp
    src/server.cpp
    src/c_api.cpp
)

add_library(turbo_engine_lib STATIC ${TURBO_ENGINE_SOURCES})
add_library(turbo_engine SHARED ${TURBO_ENGINE_SOURCES})

if (WIN32)
    target_link_libraries(turbo_engine_lib PUBLIC d3d12 dxgi d3dcompiler dxguid ws2_32)
    target_link_libraries(turbo_engine PUBLIC d3d12 dxgi d3dcompiler dxguid ws2_32)
endif()

# Main Engine CLI & GUI Target
add_executable(turbo-winfare src/main.cpp)
target_link_libraries(turbo-winfare PRIVATE turbo_engine_lib)

# The repacker target is gone. It only ever wrote zero-filled placeholder bundles, and its
# main() had already been reduced to an error message. Use tools/convert_hf_to_gturbo.py.

# Format & Optimization Test Suite Target
add_executable(test_format tests/test_format.cpp)
target_link_libraries(test_format PRIVATE turbo_engine_lib)

# SentencePiece Tokenizer Test Suite Target
add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE turbo_engine_lib)

# GPU kernel verification against the CPU reference maths
add_executable(test_gpu_kernels tests/test_gpu_kernels.cpp)
target_link_libraries(test_gpu_kernels PRIVATE turbo_engine_lib)

# Expert cache slot policy: eviction, pinning, and the plan/fetch/release split. No GPU.
add_executable(test_streamer tests/test_streamer.cpp)
target_link_libraries(test_streamer PRIVATE turbo_engine_lib)

# Capacity planning: descriptor budget and the expert-slot sizing ladder. Pure logic --
# no GPU, no model bundle. Guards the ceiling that used to cap --slots at 44.
add_executable(test_capacity tests/test_capacity.cpp)
target_link_libraries(test_capacity PRIVATE turbo_engine_lib)

# Sampling: truncation order, seeding, repetition penalty. Shared by both paths. No GPU.
add_executable(test_sampling tests/test_sampling.cpp)
target_link_libraries(test_sampling PRIVATE turbo_engine_lib)

# Streaming detokenization and stop-string matching. No GPU.
add_executable(test_detokenizer tests/test_detokenizer.cpp)
target_link_libraries(test_detokenizer PRIVATE turbo_engine_lib)

# HTTP framing and JSON. Uses loopback sockets; no GPU, no model.
add_executable(test_server tests/test_server.cpp)
target_link_libraries(test_server PRIVATE turbo_engine_lib)

# OpenAI request validation, response rendering, request admission. No GPU, no model.
add_executable(test_openai tests/test_openai.cpp)
target_link_libraries(test_openai PRIVATE turbo_engine_lib)

enable_testing()
add_test(NAME test_format COMMAND test_format)
add_test(NAME test_tokenizer COMMAND test_tokenizer)
add_test(NAME test_gpu_kernels COMMAND test_gpu_kernels)
add_test(NAME test_streamer COMMAND test_streamer)
add_test(NAME test_capacity COMMAND test_capacity)
add_test(NAME test_sampling COMMAND test_sampling)
add_test(NAME test_detokenizer COMMAND test_detokenizer)
add_test(NAME test_server COMMAND test_server)
add_test(NAME test_openai COMMAND test_openai)

# Converter layout/streaming tests. These run offline against synthetic data and guard the
# expert scatter path, where a wrong offset would silently corrupt 12 GB of weights.
find_package(Python3 COMPONENTS Interpreter)
if (Python3_FOUND)
    add_test(NAME test_convert
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_convert_streaming.py)
endif()

if (MINGW)
    target_link_options(turbo-winfare PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(test_format PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(test_gpu_kernels PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(test_tokenizer PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(test_streamer PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(test_capacity PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(test_sampling PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(test_detokenizer PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(test_server PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(test_openai PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(turbo_engine PRIVATE -static-libgcc -static-libstdc++)
endif()

# Copy GUI assets and HLSL shaders to the build directory.
#
# This runs on every BUILD, not once at configure time. Shaders are compiled at runtime from
# build/shaders/, so a configure-time-only copy meant editing a .hlsl and rebuilding silently
# kept using the stale copy -- an edit would appear to have no effect at all.
add_custom_target(copy_runtime_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/gui ${CMAKE_BINARY_DIR}/gui
    COMMENT "Copying shaders/ and gui/ into the build directory")

add_dependencies(turbo-winfare copy_runtime_assets)
add_dependencies(test_gpu_kernels copy_runtime_assets)
