#ifndef OPENFDCM_OPTIMIZESTRATEGIES_BATCHOPTIMIZE_H
#define OPENFDCM_OPTIMIZESTRATEGIES_BATCHOPTIMIZE_H
#include "openfdcm/matching/optimizestrategy.h"
#include "BS_thread_pool.hpp"

namespace openfdcm::matching {

    class BatchOptimize : public OptimizeStrategyInstance
    {
        size_t batchSize_;
        std::shared_ptr<BS::thread_pool> threadPool_;

    public:
        explicit BatchOptimize(size_t batchSize, std::shared_ptr<BS::thread_pool> pool = std::make_shared<BS::thread_pool>())
                : batchSize_{batchSize}, threadPool_{ std::move(pool)}
        {}
        explicit BatchOptimize(size_t batchSize, BS::concurrency_t num_threads)
                : batchSize_{batchSize}, threadPool_{ std::make_shared<BS::thread_pool>(num_threads)}
        {}

        [[nodiscard]] auto getBatchSize() const noexcept { return batchSize_; }
        [[nodiscard]] auto getPool() const noexcept { return std::weak_ptr{threadPool_}; }
    };

} //namespace openfdcm::optimizestrategies
#endif //OPENFDCM_OPTIMIZESTRATEGIES_BATCHOPTIMIZE_H
