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
#include "openfdcm/core/cuda/cuimgproc.cuh"
#include "test-utils/utils.h"

using namespace openfdcm;

TEST_CASE("distanceTransform", "[openfdcm::core::cuda, cuimgproc]")
{
    int const size{100};
    auto cudaStream = std::make_unique<core::cuda::CudaStream>();
    core::cuda::CudaArray<Npp8u ,-1,-1> inputImg(size, size);

    core::LineArray cpuScene(4,1);
    cpuScene << 0, 0, size-1, 0; // x1, y1, x2, y2
    core::cuda::CudaArray cuScene(cpuScene);

    setAll(inputImg, 1, cudaStream);
    drawLines(inputImg, cuScene, 0, cudaStream);
    core::cuda::CudaArray<float ,-1,-1> outpuImg(size, size);

    /*
    SECTION("Distance::L1"){
        core::cuda::distanceTransform<core::Distance::L1>(d_img, cudaStream);
        core::cuda::drawCudaFeature("distanceTransform_L1.pgm", d_img, size, cudaStream);
        auto const& cpufeature = core::cuda::copyToCpu(d_img);

        for(int row{0}; row < size; ++row) {
            auto expectedValue = static_cast<float>(row);
            REQUIRE(cpufeature.block(row, 0, 1, size).isApprox(Eigen::ArrayXXf::Constant(1, size, expectedValue)));
        }
    }*/
    SECTION("Distance::L2"){
        core::cuda::distanceTransform<core::Distance::L2>(inputImg, outpuImg, cudaStream);
        //core::cuda::drawCudaFeature("distanceTransform_L2.pgm", outpuImg, size, cudaStream);
        auto const& cpufeature = core::cuda::copyToCpu(outpuImg);

        REQUIRE(cpufeature.block(size-1, 0, 1, size).round().isApprox(Eigen::Array<float,-1,-1>::Constant(1, size, size-1)));
    }
    /*
    // There is an issue with exact EDT
    // See https://github.com/NVIDIA/CUDALibrarySamples/issues/213
    SECTION("Distance::L2_SQUARED"){
        core::cuda::distanceTransform<core::Distance::L2_SQUARED>(inputImg, outpuImg, cudaStream);
        //core::cuda::drawCudaFeature("distanceTransform_L2_SQUARED.pgm", outpuImg, size*size, cudaStream);
        auto const& cpufeature = core::cuda::copyToCpu(outpuImg);

        std::cout << "L2_SQUARED\n";
        for(int row{0}; row < size; ++row) {
            std::cout << "[" << row << "]: " << cpufeature.block(row, 0, 1, 10) << " vs " << pow(row,2)<< "\n";
        }
        REQUIRE(cpufeature.block(size-1, 0, 1, size).round().isApprox(Eigen::Array<float,-1,-1>::Constant(1, size, pow(size-1,2))));
    }*/
}


TEST_CASE("lineIntegral", "[openfdcm::core::cuda, cuimgproc]")
{
    int const size{10};
    auto cudaStream = std::make_unique<core::cuda::CudaStream>();
    core::cuda::CudaArray<float,-1,-1> d_img(size, size);

    core::LineArray cpuScene(4,size/2);
    for(int row{0}; row < size; row+=2) {
        // Draw all even horizontal lines x1, y1, x2, y2
        cpuScene.block<4,1>(0,row/2) = core::Line{0,static_cast<float>(row), size-1, static_cast<float>(row)};
    }
    core::cuda::CudaArray cuScene(cpuScene);

    setAll(d_img, 0, cudaStream);
    drawLines(d_img, cuScene, 1, cudaStream);
    lineIntegral(d_img, -M_PI_2f, cudaStream);
    auto const& cpufeature = core::cuda::copyToCpu(d_img);

    REQUIRE(cpufeature.block(size-1, 0, 1, size).isApprox(Eigen::ArrayXXf::Constant(size, 1, size/2)));
}
