#pragma once
//
// Assertions that survive an optimized build.
//
// <cassert> compiles `assert` out entirely under NDEBUG, and NDEBUG is exactly what
// -DCMAKE_BUILD_TYPE=Release defines. Before this header existed the build carried no
// CMAKE_BUILD_TYPE at all, so the asserts happened to stay live -- but the moment the build
// was configured properly, 293 assertions across eight of the nine test binaries would have
// become no-ops and those binaries would have passed unconditionally. A test that cannot
// fail is worse than no test, because it is counted as coverage.
//
// Redefining `assert` here keeps every existing call site working while making the whole
// suite independent of the build type. Include this AFTER every other header in a test
// translation unit: <cassert> has no include guard and deliberately re-defines `assert` on
// each inclusion, so anything pulling it in later would silently undo this.
//
#include <cstdio>
#include <cstdlib>

namespace gturbo_test {
[[noreturn]] inline void check_failed(const char* expr, const char* file, int line) {
    std::fprintf(stderr, "\n  [FAIL] %s:%d: assertion failed: %s\n", file, line, expr);
    std::fflush(stderr);
    std::fflush(stdout);
    std::abort();
}
} // namespace gturbo_test

#undef assert
#define assert(cond) \
    ((cond) ? (void)0 : ::gturbo_test::check_failed(#cond, __FILE__, __LINE__))
