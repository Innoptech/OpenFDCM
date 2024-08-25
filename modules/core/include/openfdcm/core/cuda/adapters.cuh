#ifndef OPENFDCM_CORE_CUDA_ADAPTERS_H
#define OPENFDCM_CORE_CUDA_ADAPTERS_H
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>


namespace openfdcm::core::cuda {
    using concurrency_t = std::invoke_result_t<decltype(std::thread::hardware_concurrency)>;


    class CudaStream {
        static cudaStream_t* allocateStream()
        {
            auto cuda_stream = new cudaStream_t{};
            auto cudaStatus = cudaStreamCreate(cuda_stream);
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error{"cudaStreamCreate failed: " + std::string(cudaGetErrorString(cudaStatus))};
            }
            return cuda_stream;
        };

        struct StreamDeleter
        {
            void operator ()(cudaStream_t* cuda_stream)
            {
                auto cudaStatus = cudaStreamDestroy(*cuda_stream);
                if (cudaStatus != cudaSuccess) {
                    std::cerr << "cudaStreamDestroy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                }
                delete cuda_stream;
            }
        };

        std::unique_ptr<cudaStream_t, StreamDeleter> cuda_stream_;

    public:
        CudaStream() : cuda_stream_{allocateStream()} {}

        [[nodiscard]] auto getStream() const noexcept { return *cuda_stream_; }
    };
    using CudaStreamPtr = std::unique_ptr<CudaStream>;


    class CudaStreamPool {
        std::queue<std::unique_ptr<CudaStream>> stream_pool_;
        std::mutex pool_mutex_;
        std::condition_variable cv_;
        size_t num_streams_;

    public:
        explicit CudaStreamPool() : CudaStreamPool{0} {}

        // Constructor to initialize the pool with a certain number of CUDA streams
        explicit CudaStreamPool(concurrency_t num_streams)
                : num_streams_{determine_thread_count(num_streams)}
        {
            for (size_t i = 0; i < num_streams_; ++i) {
                stream_pool_.emplace(std::make_unique<CudaStream>());
            }
        }

        // Get a stream from the pool (blocks if none are available)
        std::unique_ptr<CudaStream> getStream()
        {
            std::unique_lock<std::mutex> lock(pool_mutex_);
            cv_.wait(lock, [this]() { return !stream_pool_.empty(); });  // Wait until a stream is available

            auto stream = std::move(stream_pool_.front());
            stream_pool_.pop();
            return stream;
        }

        // Return a stream back to the pool
        void returnStream(std::unique_ptr<CudaStream> stream)
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            stream_pool_.push(std::move(stream));
            cv_.notify_one();  // Notify one waiting thread that a stream is available
        }

        // Destructor
        ~CudaStreamPool() = default;

    private:
        /**
         * @brief Determine how many threads the pool should have, based on the parameter passed to the constructor.
         *
         * @param num_threads If the parameter is a positive number, then the pool will be created with this number of threads. If the parameter is non-positive, or a parameter was not supplied (in which case it will have the default value of 0), then the pool will be created with the total number of hardware threads available, as obtained from `std::thread::hardware_concurrency()`. If the latter returns zero for some reason, then the pool will be created with just one thread.
         * @return The number of threads to use for constructing the pool.
         */
        [[nodiscard]] static concurrency_t determine_thread_count(const concurrency_t num_threads)
        {
            if (num_threads > 0)
                return num_threads;
            if (std::thread::hardware_concurrency() > 0)
                return std::thread::hardware_concurrency();
            return 1;
        }
    };

    inline void synchronize(const CudaStreamPtr &cuda_stream)
    {
        cudaError_t cudaStatus = cudaStreamSynchronize(cuda_stream->getStream());
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error{"cudaStreamSynchronize failed: " + std::string(cudaGetErrorString(cudaStatus))};
        }
    }

} //namespace openfdcm::cuda
#endif //OPENFDCM_CUDA_ADAPTERS_H
