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
#include "openfdcm/core/drawing.h"

namespace openfdcm::core::cuda {

    template<typename DerivedArray> requires (std::is_same_v<typename DerivedArray::Scalar, float>)
    __global__ void lineIntegralKernelCol(DerivedArray& d_img, float rastvec_x, float rastvec_y, int forwardIdx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int col = forwardIdx; // for readability
        int previous_p1x = (rastvec_x < 0) ? d_img.cols() - 1 - forwardIdx : forwardIdx;

        int p1x = previous_p1x + col * static_cast<int>(rastvec_x);
        int p1y = static_cast<int>(std::round(col * rastvec_y)) - static_cast<int>(std::round((col - 1) * rastvec_y));

        // Bound check and process the column
        int y1 = max(p1y, 0);
        int y2 = max(-p1y, 0);
        int col_len = d_img.rows() - std::abs(p1y);
        if (idx >= col_len || idx < y1) return;
        d_img(idx,p1x) += d_img(y2,previous_p1x);
    }

    template<typename DerivedArray> requires (std::is_same_v<typename DerivedArray::Scalar, float>)
    __host__ inline void lineIntegral(DerivedArray& d_img, float lineAngle, CudaStreamPtr const& stream) noexcept
    {
        Point2 rastvec = openfdcm::core::rasterizeVector(Point2{std::cos(lineAngle), std::sin(lineAngle)});

        // Set up grid and block sizes
        int threadsPerBlock = 256;
        int blocksPerGrid = (d_img.cols() + threadsPerBlock - 1) / threadsPerBlock;

        bool transposed{false};
        if (std::abs(rastvec.y()) == 1)
        {
            transposed = true;
            transpose(d_img, stream);
            rastvec = Point2{-rastvec.y(), -rastvec.x()};
        }

        // Launch the kernel based on the rasterized vector
        for(int colIdx{0}; colIdx<d_img.cols(); ++colIdx)
        {
            lineIntegralKernelCol<<<blocksPerGrid, threadsPerBlock, 0, stream->getStream()>>>(d_img, rastvec.x(), rastvec.y(), colIdx);
            synchronize(stream);
        }

        if(transposed)
            transpose(d_img, stream);
    }

    template<typename DerivedArray> requires (std::is_same_v<typename DerivedArray::Scalar, float>)
    __global__ void distanceTransformColumnPassL2Kernel(DerivedArray& d_img) {
        int col = blockIdx.x;
        if (col >= d_img.cols()) return;

        extern __shared__ int sharedMemory[];
        int* v = sharedMemory;              // Use shared memory for v
        auto* z = (float*)&v[d_img.rows()];  // Use shared memory for z

        int k = 0;
        v[0] = 0;
        z[0] = -INFINITY;
        z[1] = INFINITY;

        for (int q = 1; q < d_img.rows(); ++q) {
            while (true) {
                int v_k = v[k];
                float const s = (d_img(q, col) + float(q * q) - d_img(v_k, col) - float(v_k * v_k)) / float(2 * q - 2 * v_k);
                if (s > z[k]) {
                    k++;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = INFINITY;
                    break;
                }
                k--;
            }
        }

        k = 0;
        for (int q = 0; q < d_img.rows(); ++q) {
            while (z[k + 1] < (float)q) k++;
            int v_k = v[k];
            int q_ = q - v_k;
            d_img(q, col) = d_img(v_k, col) + float(q_ * q_);
        }
    }

    /**
     * @brief Performs a 1D pass of the algorithm Distance Transforms of Sampled Functions for L2_SQUARED distance
     * @param d_img The device image that will be transformed in place
     */
    template<typename DerivedArray> requires (std::is_same_v<typename DerivedArray::Scalar, float>)
    __host__ static inline void _distanceTransformColumnPassL2(DerivedArray &d_img, CudaStreamPtr const& stream) noexcept {
        // Determine block and grid sizes
        int threadsPerBlock = d_img.rows(); // Each thread handles one row in a column
        int blocksPerGrid = d_img.cols();   // Each block handles one column
        size_t sharedMemorySize = d_img.rows() * sizeof(int) + (d_img.rows() + 1) * sizeof(float);
        distanceTransformColumnPassL2Kernel<<<blocksPerGrid, threadsPerBlock, sharedMemorySize, stream->getStream()>>>(d_img);
        synchronize(stream);
    }

    template<typename DerivedArray> requires (std::is_same_v<typename DerivedArray::Scalar, float>)
    __global__ void distanceTransformColumnPassL1Kernel(DerivedArray &d_img) {
        int col = blockIdx.x;
        if (col >= d_img.cols()) return;

        // Forward pass
        for (int q = 1; q < d_img.rows(); ++q) {
            d_img(q, col) = min(d_img(q, col), d_img(q-1, col) + 1.0f);
        }

        // Backward pass
        for (int q = d_img.rows() - 2; q >= 0; --q) {
            d_img(q, col) = min(d_img(q, col), d_img(q+1, col) + 1.0f);
        }
    }

    /**
     * @brief Performs a 1D pass of the algorithm Distance Transforms of Sampled Functions
     * @param d_img The device image that will be transformed in place
     */
    template<typename DerivedArray> requires (std::is_same_v<typename DerivedArray::Scalar, float>)
    __host__ static inline void _distanceTransformColumnPassL1(DerivedArray &d_img, CudaStreamPtr const& stream) noexcept {
        int threadsPerBlock = d_img.rows();  // Each thread handles one row in a column
        int blocksPerGrid = d_img.cols();    // Each block handles one column
        distanceTransformColumnPassL1Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream->getStream()>>>(d_img);
        synchronize(stream);
    }

    /**
     * @brief Perform the distance transform for a given set of lines.
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
     * @param linearray The given set of lines for which the distance transform is to be computed.
     * @return The resulting image with the distance transform applied, where each pixel holds the distance value.
     */
    template<core::Distance D=core::Distance::L2, typename DerivedArray>
    __host__ inline void distanceTransform(DerivedArray &colmaj_img, cuLineArray const& linearray,
                                  CudaStreamPtr const& stream) noexcept {

        if constexpr (D == core::Distance::L1) {
            _distanceTransformColumnPassL1(colmaj_img, stream);
            transpose(colmaj_img, stream);
            _distanceTransformColumnPassL1(colmaj_img, stream);
            transpose(colmaj_img, stream);
            return;
        }

        _distanceTransformColumnPassL2(colmaj_img, stream);
        transpose(colmaj_img, stream);
        _distanceTransformColumnPassL2(colmaj_img, stream);
        transpose(colmaj_img, stream);

        if constexpr (D == core::Distance::L2) {
            core::cuda::sqrt(colmaj_img, stream);
        }
    }

    template<typename DerivedArray>
    __global__ void minCoeffKernel(DerivedArray& array1, DerivedArray const& array2, float coeff) {
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
    __host__
    inline void propagateOrientation(std::vector<std::shared_ptr<CudaArray<float,-1,-1>>> &featuremap,
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
