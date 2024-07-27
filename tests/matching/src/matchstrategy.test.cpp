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
#include "openfdcm/matching/optimizerstrategies/defaultoptimize.h"

using namespace openfdcm::core;
using namespace openfdcm::matching;

TEST_CASE("getSceneCenteredTranslation")
{
    SECTION("Centered scene")
    {
        // Center a scene in an image given the final image area ratio (scene size vs image size)
        const LineArray scene{
                {0, 0},
                {0, 0},
                {9, 0},
                {0, 9}
        };
        float const scene_padding{1.f};
        SceneShift const &scene_shift = getSceneCenteredTranslation(scene, scene_padding);
        REQUIRE(allClose(scene_shift.sceneSize, Size{10, 10}));
        REQUIRE(allClose(scene_shift.translation, Point2{0, 0}));

        auto const& [min_point, max_point] = minmaxPoint(scene);
        Point2 const new_center = ((max_point + scene_shift.translation) +
                (min_point + scene_shift.translation))/2.f;
        REQUIRE(allClose(new_center, (scene_shift.sceneSize.cast<float>().array()-1.f)/2.f));
    }
    SECTION("Uncentered scene")
    {
        const LineArray scene{
                {-6, 0},
                {1, -10},
                {4, 0},
                {1, 10}
        };
        float const scene_padding{2.f};
        SceneShift const &scene_shift = getSceneCenteredTranslation(scene, scene_padding);
        REQUIRE(allClose(scene_shift.sceneSize, Size{41, 41}));
        REQUIRE(allClose(scene_shift.translation, Point2{21, 20}));

        auto const& [min_point, max_point] = minmaxPoint(scene);
        Point2 const new_center = ((max_point + scene_shift.translation) +
                                   (min_point + scene_shift.translation))/2.f;
        REQUIRE(allClose(new_center, (scene_shift.sceneSize.cast<float>().array()-1.f)/2.f));
    }
}

void run_test(float scene_ratio) {
    size_t const max_tmpl_lines{4}, max_scene_lines{10};
    size_t const depth{30};
    float const scene_padding{2.2f};
    float const coeff{5.f};
    DefaultSearch searchStrategy{max_tmpl_lines, max_scene_lines};
    DefaultOptimize optimizerStrategy{};
    DefaultMatch matcher{depth, coeff, scene_ratio, scene_padding};
    size_t const numberOfLines{10};
    size_t const lineLength{100};
    LineArray tmpl = tests::createLines(numberOfLines, lineLength);

    // Test for rotation
    {
        Mat23 scene_transform{{-1, 0, lineLength}, {0, -1, lineLength}};
        LineArray scene = transform(tmpl, scene_transform);
        std::vector<Match> matches = search(matcher, searchStrategy, optimizerStrategy, {tmpl}, scene);
        Mat22 best_match_rotation = matches[0].transform.block<2, 2>(0, 0);
        Point2 best_match_translation = matches[0].transform.block<2, 1>(0, 2);

        REQUIRE(matches.size() == std::min(max_tmpl_lines, numberOfLines) * std::min(numberOfLines, max_scene_lines) * 2);
        REQUIRE(allClose(scene_transform.block<2, 2>(0, 0), best_match_rotation, 1e-5));
        REQUIRE(allClose(scene_transform.block<2, 1>(0, 2), best_match_translation, 1e0 * 1 / scene_ratio));
    }

    // Test for translation
    {
        Mat23 scene_transform{{1, 0, 0}, {0, 1, 0}};
        LineArray scene = transform(tmpl, scene_transform);
        std::vector<Match> matches = search(matcher, searchStrategy, optimizerStrategy, {tmpl}, scene);
        Mat22 best_match_rotation = matches[0].transform.block<2, 2>(0, 0);
        Point2 best_match_translation = matches[0].transform.block<2, 1>(0, 2);

        REQUIRE(matches.size() == max_tmpl_lines * max_scene_lines * 2);
        REQUIRE(allClose(scene_transform.block<2, 2>(0, 0), best_match_rotation, 1e-5));
        REQUIRE(allClose(scene_transform.block<2, 1>(0, 2), best_match_translation, 1e0 * 1 / scene_ratio));
    }

    // Test for empty scene
    {
        LineArray scene(4, 0);
        std::vector<Match> matches = search(matcher, searchStrategy, optimizerStrategy, {tmpl}, scene);
        REQUIRE(matches.empty());
    }

    // Test for empty template
    {
        std::vector<LineArray> templates;
        LineArray scene = tmpl;
        std::vector<Match> matches = search(matcher, searchStrategy, optimizerStrategy, templates, scene);
        REQUIRE(matches.empty());
    }

    // Test for template with empty line
    {
        std::vector<LineArray> templates{LineArray(4, 0)};
        LineArray scene = tmpl;
        std::vector<Match> matches = search(matcher, searchStrategy, optimizerStrategy, templates, scene);
        REQUIRE(matches.empty());
    }
}

TEST_CASE("DefaultMatch") {
    run_test(1.0f);
}

TEST_CASE("Scale down scene") {
    run_test(0.3f);
}