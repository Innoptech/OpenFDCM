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

#ifndef OPENFDCM_CUDA_CUMATH_CUH
#define OPENFDCM_CUDA_CUMATH_CUH
#include "openfdcm/core/math.h"

namespace openfdcm::core::cuda {

    /**
     * @brief Return true if two number are equals relatively to their respective precision
     * @tparam U The type of the numbers
     * @param a The first number to compare
     * @param b The second number to compare
     * @param rtol The relative tolerance
     * @param atol The absolute tolerance
     * @return true if two number are equals relatively to their respective precision
     */
    template<typename T, typename U>
    __host__ __device__
    inline bool relativelyEqual(
            T const& a, U const& b,
            double const& rtol=1e-10, double const& atol=std::numeric_limits<T>::epsilon()) noexcept
    {
        return std::fabs(a - b) <= atol + rtol * max(std::fabs(a), std::fabs(b));
    }

    __host__ __device__ inline Point2 p1(Line const& line) noexcept { return line.block<2,1>(0,0); }
    __host__ __device__ inline Point2 p2(Line const& line) noexcept { return line.block<2,1>(2,0); }
    __host__ __device__
    inline auto getLine(LineArray const& linearray, long const idx) noexcept {return linearray.block<4,1>(0, idx);}

    __host__ __device__
    inline Eigen::Matrix<float,2,-1> getCenter(LineArray const& linearray) noexcept {
        return (linearray.block(2,0,2,linearray.cols()) + linearray.block(0,0,2,linearray.cols()))/2;
    }
} //namespace openfdcm::core::cuda
#endif //OPENFDCM_CUDA_CUMATH_CUH
