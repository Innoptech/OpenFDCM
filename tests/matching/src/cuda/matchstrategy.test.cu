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
#include "test-utils/utils.h"
#include "openfdcm/matching/matchstrategies/defaultmatch.h"
#include "openfdcm/matching/searchstrategies/defaultsearch.h"
#include "openfdcm/matching/optimizestrategies/defaultoptimize.h"
#include "openfdcm/matching/featuremaps/cuda/dt3cuda.cuh"

using namespace openfdcm;
using namespace openfdcm::matching::cuda;


void run_cuda_test(float scene_ratio, BS::concurrency_t num_threads) {
    size_t const max_tmpl_lines{3}, max_scene_lines{3};
    size_t const depth{30};
    float const scene_padding{2.2f};
    float const coeff{5.f};

    auto threadpool = std::make_shared<BS::thread_pool>(num_threads);
    auto streampool = std::make_shared<core::cuda::CudaStreamPool>(num_threads);

    matching::DefaultSearch searchStrategy{max_tmpl_lines, max_scene_lines};
    matching::DefaultOptimize optimizerStrategy{threadpool};
    matching::DefaultMatch matcher{};
    size_t const numberOfLines{10};
    size_t const lineLength{10};
    core::LineArray tmpl = tests::createLines(numberOfLines, lineLength);

    // Test for rotation
    {
        core::Mat23 scene_transform{{-1, 0, lineLength}, {0, -1, lineLength}};
        core::LineArray scene =  core::transform(tmpl, scene_transform);

        Dt3CudaParameters params{depth, coeff, scene_padding};
        const Dt3Cuda& featuremap = buildCudaFeaturemap<core::Distance::L2>(scene, params, threadpool, streampool);

        auto matches = search(matcher, searchStrategy, optimizerStrategy, featuremap, {tmpl}, scene);
        sortMatches(matches);

        core::Mat22 best_match_rotation = matches[0].transform.block<2, 2>(0, 0);
        core::Point2 best_match_translation = matches[0].transform.block<2, 1>(0, 2);

        REQUIRE(matches.size() == std::min(max_tmpl_lines, numberOfLines) * std::min(numberOfLines, max_scene_lines) * 2);
        REQUIRE( core::allClose(scene_transform.block<2, 2>(0, 0), best_match_rotation, 1e-5));
        REQUIRE( core::allClose(scene_transform.block<2, 1>(0, 2), best_match_translation, 1e0 * 1 / scene_ratio));
        REQUIRE(matches[0].tmplIdx == 0);
    }

    // Test for translation
    {
        core::Mat23 scene_transform{{1, 0, 0}, {0, 1, 0}};
        core::LineArray scene = core::transform(tmpl, scene_transform);

        Dt3CudaParameters params{depth, coeff, scene_padding};
        const Dt3Cuda& featuremap = buildCudaFeaturemap(scene, params, threadpool, streampool);

        auto matches = search(matcher, searchStrategy, optimizerStrategy, featuremap, {tmpl}, scene);
        std::sort(std::begin(matches), std::end(matches));
        core::Mat22 best_match_rotation = matches[0].transform.block<2, 2>(0, 0);
        core::Point2 best_match_translation = matches[0].transform.block<2, 1>(0, 2);

        REQUIRE(matches.size() == max_tmpl_lines * max_scene_lines * 2);
        REQUIRE(core::allClose(scene_transform.block<2, 2>(0, 0), best_match_rotation, 1e-5));
        REQUIRE(core::allClose(scene_transform.block<2, 1>(0, 2), best_match_translation, 1e0 * 1 / scene_ratio));
        REQUIRE(matches[0].tmplIdx == 0);
    }

    // Test for empty scene
    {
        core::LineArray scene(4, 0);

        Dt3CudaParameters params{depth, coeff, scene_padding};
        const Dt3Cuda& featuremap = buildCudaFeaturemap(scene, params, threadpool, streampool);

        auto matches = search(matcher, searchStrategy, optimizerStrategy, featuremap, {tmpl}, scene);
        std::sort(std::begin(matches), std::end(matches));
        REQUIRE(matches.empty());
    }

    // Test for empty template
    {
        std::vector<core::LineArray> templates;
        core::LineArray scene = tmpl;

        Dt3CudaParameters params{depth, coeff, scene_padding};
        const Dt3Cuda& featuremap = buildCudaFeaturemap(scene, params, threadpool, streampool);

        auto matches = search(matcher, searchStrategy, optimizerStrategy, featuremap, templates, scene);
        std::sort(std::begin(matches), std::end(matches));
        REQUIRE(matches.empty());
    }

    // Test for template with empty line
    {
        std::vector<core::LineArray> templates{core::LineArray(4, 0)};
        core::LineArray scene = tmpl;

        Dt3CudaParameters params{depth, coeff, scene_padding};
        const Dt3Cuda& featuremap = buildCudaFeaturemap(scene, params, threadpool, streampool);

        auto matches = search(matcher, searchStrategy, optimizerStrategy, featuremap, templates, scene);
        std::sort(std::begin(matches), std::end(matches));
        REQUIRE(matches.empty());
    }
}

TEST_CASE("DefaultMatch on cuda dt3") {
    run_cuda_test(1.0f, 1);
    run_cuda_test(1.0f, 2);
}
