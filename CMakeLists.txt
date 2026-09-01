cmake_minimum_required(VERSION 3.20)
project(turbo_winfare LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization level. This is load-bearing and was missing entirely.
#
# Nothing here set CMAKE_BUILD_TYPE, and the documented build command was
# `cmake --build build --config Release` -- but `--config` only means something to a
# MULTI-config generator (Visual Studio, Xcode). Ninja is single-config, so it accepted the
# flag and ignored it, and CMake emitted `FLAGS = -std=gnu++23` with no -O at all. Every
# performance figure recorded before this line existed was measured on a -O0 binary: the
# per-token CPU work, the tokenizer, sampling, and the whole scalar --cpu reference.
#
# The build type is chosen at CONFIGURE time. `--config` on the build line is still inert.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

# Off by default on purpose: turbo-winfare.exe is a shipped artifact and the build host is not
# necessarily the Zen 4 target. Turn it on for a local benchmark build only.
option(TURBO_NATIVE_ARCH "Tune the build for the building machine (-march=native)" OFF)
if (TURBO_NATIVE_ARCH)
    add_compile_options(-march=native)
endif()

# Deliberately NOT -ffast-math. The GPU path is gated on matching the scalar CPU reference
# token-for-token under greedy decoding; relaxing IEEE semantics on one side of that
# comparison would retire the only end-to-end correctness check this project has.

# Guard against a cache that has had its per-config flags blanked.
#
# This is not hypothetical -- it is how the project shipped a -O0 "Release" build for its
# entire history. The cache in build/ had CMAKE_CXX_FLAGS_RELEASE, _DEBUG, _RELWITHDEBINFO
# and _MINSIZEREL all set to the empty string, so selecting a build type changed nothing and
# CMake emitted `FLAGS = -std=gnu++23`. A clean cache populates them correctly, which is what
# makes this so easy to miss: it reproduces only in the tree you already have. Restore the
# compiler defaults rather than trusting whatever is cached.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if (NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "-O")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG"
            CACHE STRING "Flags for Release" FORCE)
    endif()
    if (NOT CMAKE_CXX_FLAGS_RELWITHDEBINFO MATCHES "-O")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG"
            CACHE STRING "Flags for RelWithDebInfo" FORCE)
    endif()
endif()

# Print the effective flags. The whole failure above was silent; one STATUS line makes the
# next occurrence obvious at configure time instead of after a benchmark.
message(STATUS "turbo-winfare: CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'")
string(TOUPPER "${CMAKE_BUILD_TYPE}" _turbo_cfg)
message(STATUS "turbo-winfare: CXX flags = '${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_turbo_cfg}}'")


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

# Application icon. Without a resource the shell falls back to the generic executable glyph,
# which makes turbo-winfare.exe indistinguishable from every other .exe in a folder listing.
#
# windres resolves the ICON filename against its include path rather than against the .rc
# file's own directory, so assets/ has to be on that path explicitly. CMake feeds <INCLUDES>
# from the target's include directories, which would also apply to the C++ compiles, so this
# goes through CMAKE_RC_FLAGS instead.
if (WIN32)
    enable_language(RC)
    set(CMAKE_RC_FLAGS "${CMAKE_RC_FLAGS} -I \"${CMAKE_CURRENT_SOURCE_DIR}/assets\"")
    target_sources(turbo-winfare PRIVATE assets/turbo-winfare.rc)
endif()

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
