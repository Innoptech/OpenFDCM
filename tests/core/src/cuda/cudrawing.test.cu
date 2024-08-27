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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include "openfdcm/core/cuda/cudrawing.cuh"
#include "test-utils/utils.h"

using namespace openfdcm;

TEST_CASE("cudrawing", "[openfdcm::core::cuda, Dt3Cuda]")
{
    int const size{10};
    float minVal{0.f}, maxVal{1.f};
    auto cudaStream = std::make_unique<core::cuda::CudaStream>();
    core::cuda::CudaArray<float,-1,-1> d_img(size, size);

    core::LineArray cpuScene(4,size/2);
    for(int row{0}; row < size; row+=2) {
        // Draw all even horizontal lines x1, y1, x2, y2
        cpuScene.block<4,1>(0,row/2) = core::Line{0,static_cast<float>(row), size-1, static_cast<float>(row)};
    }
    core::cuda::CudaArray cuScene(cpuScene);

    // Draw maxVal on minVal
    setAll(d_img, minVal, cudaStream);
    drawLines(d_img, cuScene, maxVal, cudaStream);
    core::cuda::drawCudaFeature("draw.pgm", d_img, maxVal, cudaStream);

    auto const& cpufeature = core::cuda::copyToCpu(d_img);

    for(int row{0}; row < size; ++row) {
        auto expectedValue = (row % 2) ? minVal : maxVal;
        REQUIRE(cpufeature.block(row, 0, 1, size).isApprox(Eigen::ArrayXXf::Constant(1, size, expectedValue)));
    }
}
