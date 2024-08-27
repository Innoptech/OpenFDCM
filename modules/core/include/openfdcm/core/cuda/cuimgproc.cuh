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

#ifndef OPENFDCM_CUDA_CUIMGPROC_CUH
#define OPENFDCM_CUDA_CUIMGPROC_CUH
#include "openfdcm/core/cuda/cumath.cuh"
#include "openfdcm/core/cuda/cudrawing.cuh"
#include "openfdcm/core/imgproc.h"
#include "openfdcm/core/drawing.h"

#include <nppdefs.h>
#include <nppcore.h>
#include <nppi_filtering_functions.h>


namespace openfdcm::core::cuda {

#include <stdio.h>


    template<typename DerivedCuda> requires (std::is_floating_point_v<typename DerivedCuda::Scalar>)
    __host__ inline void lineIntegral(CudaArrayBase<DerivedCuda>& d_img, float lineAngle, CudaStreamPtr const& stream) noexcept
    {
        Point2 rastvec = openfdcm::core::rasterizeVector(Point2{std::cos(lineAngle), std::sin(lineAngle)});

        bool transposed{false};
        if (std::abs(rastvec.y()) == 1)
        {
            transposed = true;
            transpose(d_img, stream);
            rastvec = Point2{-rastvec.y(), -rastvec.x()};
        }

        // Launch the kernel
        int rows = d_img.rows();
        int cols = d_img.cols();
        prescan<<<1, rows / 2, 512 * sizeof(float)>>>(d_img.derived().dataTmp(), d_img.derived().data(), 512);
        synchronize(stream);
        swap(d_img.derived().data(), d_img.derived().dataTmp());

        if(transposed)
            transpose(d_img, stream);
    }


    /**
     * @brief Perform the distance transform for a given set of lines.
     * See https://github.com/NVIDIA/CUDALibrarySamples/tree/master/NPP/distanceTransform
     *
     * This function computes the distance transform of an image based on the provided line array
     * and image size. It supports different distance metrics (L1, L2, and L2 squared) depending on
     * the template parameter `D`. The algorithm is based on the "Distance Transforms of Sampled Functions".
     *
     * - **L1 Distance**: Computes the Manhattan distance.
     * - **L2 Distance**: Computes the Euclidean distance.
     * - **L2 Squared Distance**: Computes the square of the Euclidean distance.
     *
     * Depending on the chosen distance type, different algorithms are applied in both the column
     * and row passes.
     *
     * @tparam D The type of distance metric to use (default is `Distance::L2`).
     * @param d_img The pre-allocated cuda image
     * @return The resulting image with the distance transform applied, where each pixel holds the distance value.
     */
    template<core::Distance D=core::Distance::L1, typename DerivedCuda>
    __host__ inline void distanceTransform(CudaArray<Npp8u,-1,-1> const& inputImg,
            CudaArrayBase<DerivedCuda> &outputImg, CudaStreamPtr const& stream)
    {
        using T = typename DerivedCuda::Scalar;
        static_assert(D!=core::Distance::L1, "NO available CUDA impl for L1 distance");
        static_assert((std::is_same_v<T, float>) || (std::is_same_v<T, uint16_t>));

        NppStreamContext nppStreamCtx;
        nppStreamCtx.hStream = stream->getStream();

        cudaError_t cudaError = cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
        if (cudaError != cudaSuccess){
            throw std::runtime_error("CUDA error: no devices supporting CUDA.\n");
        }

        cudaError = cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags);

        cudaDeviceProp oDeviceProperties{};
        cudaError = cudaGetDeviceProperties(&oDeviceProperties, nppStreamCtx.nCudaDeviceId);

        nppStreamCtx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
        nppStreamCtx.nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor;
        nppStreamCtx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
        nppStreamCtx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;

        NppiSize oImageSizeROI(inputImg.cols(), inputImg.rows());

        size_t nScratchBufferSize;
        if(nppiDistanceTransformPBAGetBufferSize(oImageSizeROI, &nScratchBufferSize) != NPP_NO_ERROR) {
            throw std::runtime_error{"CUDA error: Not able to get PBA buffer size.\n"};
        }
        cuVector<Npp8u,-1> deviceBufferPBA(nScratchBufferSize);

        if constexpr (std::is_same_v<T, float>)
        {
            if (nppiDistanceTransformPBA_8u32f_C1R_Ctx(
                    inputImg.derived().data(), outputImg.cols() * sizeof(Npp8u), 0u, 0u,
                    nullptr, 0, nullptr, 0, nullptr, 0,
                    outputImg.derived().data(), outputImg.cols() * sizeof(Npp32f),
                    oImageSizeROI, deviceBufferPBA.data(), nppStreamCtx) != NPP_SUCCESS){
                throw std::runtime_error{"CUDA error: Distance Transform failed.\n"};
            }
        }
        if constexpr (std::is_same_v<T, uint16_t>)
        {
            if (nppiDistanceTransformPBA_8u16u_C1R_Ctx(
                    inputImg.derived().data(), outputImg.cols() * sizeof(Npp8u), 0u, 0u,
                    nullptr, 0, nullptr, 0, nullptr, 0,
                    outputImg.derived().data(), outputImg.cols() * sizeof(Npp16u),
                    oImageSizeROI, deviceBufferPBA.data(), nppStreamCtx) != NPP_SUCCESS){
                throw std::runtime_error{"CUDA error: Distance Transform failed.\n"};
            }
        }
        synchronize(stream);

        if constexpr (D==core::Distance::L2_SQUARED) {
            core::cuda::pow(outputImg, static_cast<T>(2), stream);
        }
    }

    template<typename DerivedCuda>
    __global__ void minCoeffKernel(CudaArrayBase<DerivedCuda>& array1, 
                                   CudaArrayBase<DerivedCuda> const& array2, float coeff) {
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

        // Ensure we are within bounds
        if (idx_x < array1.cols() && idx_y < array1.rows()) {
            array1(idx_y, idx_x) = fminf(array1(idx_y, idx_x), array2(idx_y, idx_x) + coeff);
        }
    }

    /**
     * @brief Propagate the distance transform in the orientation (feature) space
     * @param featuremap The cuda feature map
     * @param coeff The propagation coefficient
     * @param stream The cuda stream on which to run the kernel
     */
    __host__ inline void propagateOrientation(
            std::vector<std::shared_ptr<CudaArray<float,-1,-1>>> &featuremap,
            std::vector<float> const& angles, float coeff, CudaStreamPtr const& stream) noexcept
    {
        assert(!featuremap.empty());
        assert(featuremap.size() == angles.size());

        // Precompute constants
        const int m = static_cast<int>(angles.size());
        const int one_and_a_half_cycle_forward = static_cast<int>(std::ceil(1.5 * m));
        const int one_and_a_half_cycle_backward = -static_cast<int>(std::floor(1.5 * m));

        // Set the dimensions of the grid and block
        int numThreadsX{16}, numThreadsY{16};
        dim3 threadsPerBlock(numThreadsX, numThreadsY);
        dim3 numBlocks((featuremap[0]->cols() + numThreadsX - 1) / numThreadsX,
                       (featuremap[0]->rows() + numThreadsY - 1) / numThreadsY);

        auto propagate = [&](int start, int end, int step) {
            for (int c = start; c != end; c += step) {
                int c1 = (m + ((c - step) % m)) % m;
                int c2 = (m + (c % m)) % m;

                const float angle1 = angles[c1];
                const float angle2 = angles[c2];

                const float h = std::abs(angle1 - angle2);
                const float min_h = std::min(h, std::abs(h - M_PIf));

                minCoeffKernel<<<numBlocks, threadsPerBlock, 0, stream->getStream()>>>(*featuremap[c2], *featuremap[c1], coeff * min_h);
                synchronize(stream);
            }
        };

        propagate(0, one_and_a_half_cycle_forward, 1);
        propagate(m, one_and_a_half_cycle_backward, -1);
    }
} //namespace openfdcm::cuda
#endif //OPENFDCM_CUDA_CUIMGPROC_CUH
