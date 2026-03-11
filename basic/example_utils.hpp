#ifndef COMMON_EXAMPLE_UTILS_HPP
#define COMMON_EXAMPLE_UTILS_HPP

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime.h>

constexpr int error_exit_code = -1;

/// \brief Checks i the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                       \
    {                                                                                              \
        const hipError_t error = condition;                                                        \
        if(error != hipSuccess)                                                                    \
        {                                                                                          \
            std::cerr << "An error occured encountered: \"" << hipGetErrorString(error) << "\" at" \
                      << __FILE__ << ':' << __LINE__ << std::endl;                                 \
            std::exit(error_exit_code);                                                            \
        }                                                                                          \
    }
#endif // COMMON_EXAMPLE_UTILS_HPP