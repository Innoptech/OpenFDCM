/*
MIT License

Copyright (c) 2024 Innoptech

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

#ifndef OPENFDCM_DEFAULTOPTIMIZE_H
#define OPENFDCM_DEFAULTOPTIMIZE_H
#include "openfdcm/matching/optimizestrategy.h"
#include "BS_thread_pool.hpp"

namespace openfdcm::matching
{
    class DefaultOptimize : public OptimizeStrategyInstance
    {
        std::shared_ptr<BS::thread_pool> threadPool_;

    public:
        explicit DefaultOptimize(std::shared_ptr<BS::thread_pool> pool = std::make_shared<BS::thread_pool>())
                : threadPool_{ std::move(pool)}
        {}
        explicit DefaultOptimize(BS::concurrency_t num_threads)
                : threadPool_{ std::make_shared<BS::thread_pool>(num_threads)}
        {}

        [[nodiscard]] auto getPool() const noexcept { return std::weak_ptr{threadPool_}; }
    };
}

#endif //OPENFDCM_DEFAULTOPTIMIZE_H
