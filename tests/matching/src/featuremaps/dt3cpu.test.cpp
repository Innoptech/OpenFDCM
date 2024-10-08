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
#include "openfdcm/matching/featuremaps/dt3cpu.h"
#include "test-utils/utils.h"

#include <iostream>

using namespace openfdcm::core;
using namespace openfdcm::matching;
using namespace openfdcm::matching::detail;

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

TEST_CASE("minmaxTranslation")
{
    SECTION("along X")
    {
        const LineArray tmpl{
                {4,5},
                {0,0},
                {5,6},
                {0,0}
        };
        Point2 const align_vec{1,0};
        auto const& [neg_mul, pos_mul] = minmaxTranslation(tmpl, align_vec, Size{10,1});
        REQUIRE(relativelyEqual(neg_mul,-4.f));
        REQUIRE(relativelyEqual(pos_mul,3.f));
    }
    SECTION("along Y")
    {
        const LineArray tmpl{
                {0,0},
                {4,5},
                {0,0},
                {5,6}
        };
        Point2 const align_vec{0,1};
        auto const& [neg_mul, pos_mul] = minmaxTranslation(tmpl, align_vec, Size{1,10});
        REQUIRE(relativelyEqual(neg_mul,-4.f));
        REQUIRE(relativelyEqual(pos_mul,3.f));
    }
    SECTION("along X & Y")
    {
        const LineArray tmpl{
                {3,4},
                {4,5},
                {4,4},
                {5,6}
        };
        Point2 const align_vec{0.5f,0.5f};
        auto const& [neg_mul, pos_mul] = minmaxTranslation(tmpl, align_vec, Size{10,10});
        REQUIRE(relativelyEqual(neg_mul,-6.f));
        REQUIRE(relativelyEqual(pos_mul,6.f));
    }
    SECTION("Null align vector")
    {
        auto const& [neg_mul, pos_mul] = minmaxTranslation(LineArray{}, Point2{0,0}, Size{});
        REQUIRE(neg_mul == std::numeric_limits<float>::infinity());
        REQUIRE(pos_mul == std::numeric_limits<float>::infinity());
    }
    SECTION("Template out of bound in X axis")
    {
        const LineArray tmpl1{
                {3,4},
                {4,5},
                {4,10},
                {5,6}
        };
        const LineArray tmpl2{
                {-1,4},
                {4,5},
                {4,9},
                {5,6}
        };
        auto const& [neg_mul1, pos_mul1] = minmaxTranslation(tmpl1, Point2{1,1}, Size{10,10});
        auto const& [neg_mul2, pos_mul2] = minmaxTranslation(tmpl2, Point2{1,1}, Size{10,10});
        REQUIRE(std::isnan(neg_mul1));
        REQUIRE(std::isnan(pos_mul1));
        REQUIRE(std::isnan(neg_mul2));
        REQUIRE(std::isnan(pos_mul2));
    }
    SECTION("Template out of bound in Y axis")
    {
        const LineArray tmpl1{
                {3,4},
                {4,10},
                {4,9},
                {5,6}
        };
        const LineArray tmpl2{
                {1,4},
                {4,-1},
                {4,9},
                {5,6}
        };
        auto const& [neg_mul1, pos_mul1] = minmaxTranslation(tmpl1, Point2{1,1}, Size{10,10});
        auto const& [neg_mul2, pos_mul2] = minmaxTranslation(tmpl2, Point2{1,1}, Size{10,10});
        REQUIRE(std::isnan(neg_mul1));
        REQUIRE(std::isnan(pos_mul1));
        REQUIRE(std::isnan(neg_mul2));
        REQUIRE(std::isnan(pos_mul2));
    }
    SECTION("Touching left top border tmpl lines")
    {
        const LineArray tmpl{
                {0},
                {0},
                {10},
                {10}
        };
        auto const& [neg_mul, pos_mul] = minmaxTranslation(tmpl, Point2{1,0}, Size{20,20});
        REQUIRE(relativelyEqual(neg_mul, 0.f));
        REQUIRE(relativelyEqual(pos_mul, 9.f));
    }
    SECTION("Touching right bottom border in the tmpl lines")
    {
        const LineArray tmpl{
                {19},
                {0},
                {19},
                {19}
        };
        auto const& [neg_mul, pos_mul] = minmaxTranslation(tmpl, Point2{1,0}, Size{20,20});
        REQUIRE(relativelyEqual(neg_mul, -19.f));
        REQUIRE(relativelyEqual(pos_mul, 0.f));
    }
    SECTION("Touching both borders")
    {
        const LineArray tmpl{
                {0},
                {0},
                {19},
                {19}
        };
        auto const& [neg_mul, pos_mul] = minmaxTranslation(tmpl, Point2{1,0}, Size{20,20});
        REQUIRE(relativelyEqual(neg_mul, 0.f));
        REQUIRE(relativelyEqual(pos_mul, 0.f));
    }
    SECTION("Negative align vector in X axis")
    {
        const LineArray tmpl{
                {10},
                {0},
                {10},
                {10}
        };
        auto const& [neg_mul, pos_mul] = minmaxTranslation(tmpl, Point2{-1,0}, Size{20,20});
        REQUIRE(relativelyEqual(neg_mul, -9.f));
        REQUIRE(relativelyEqual(pos_mul, 10.f));
    }
    SECTION("Negative align vector in Y axis")
    {
        const LineArray tmpl{
                {0},
                {10},
                {10},
                {10}
        };
        auto const& [neg_mul, pos_mul] = minmaxTranslation(tmpl, Point2{0,-1}, Size{20,20});
        REQUIRE(relativelyEqual(neg_mul, -9.f));
        REQUIRE(relativelyEqual(pos_mul, 10.f));
    }
}


TEST_CASE("Dt3Cpu")
{
    SECTION("closestOrientation")
    {
        std::map<float, void*> angle_set{
                {-M_PI_2f + M_PI / 100, nullptr},
                {-M_PI / 4.f,            nullptr},
                {0.f,                  nullptr},
                {M_PI / 4.f,             nullptr},
                {M_PI_2f - M_PI / 100,  nullptr},
                {M_PI,                   nullptr}
        };

        for (auto const& [angle, _] : angle_set)
        {
            const Line initial_line{0,0,1,0};
            const Line rotated_line = rotate(initial_line, tests::makeRotation(angle));
            auto const& it = closestOrientation(angle_set, rotated_line);
            REQUIRE(relativelyEqual(it->first, constrainHalfAngle(angle)));
        }
    }
    SECTION("classifyLines")
    {
        const LineArray linearray{
                {0,  0,  0,  0,  10},
                {0,  0,  0, 10, 10},
                {0, 20, 10, 10,  10},
                {10, 20, 0,  0,  0}
        };
        std::set<float> const angle_set{-M_PI_4f, 0.f, M_PI_4f, M_PI_2f};
        std::map<float, std::vector<Eigen::Index>> const& indices = classifyLines(angle_set, linearray);

        std::map<float, std::vector<Eigen::Index>> expected = {
                {-M_PI_4f, {3}},
                {0.f, {2}},
                {M_PI_4f, {1}},
                {M_PI_2f, {0,4}},
        };
        REQUIRE(indices == expected);
    }
    SECTION("propagateOrientation")
    {
        const float coeff{0.5f};
        const Size featuresize{30, 40};
        const LineArray linearray{
                {0},
                {0},
                {0},
                {39}
        };
        Dt3CpuMap<float> featuremap{};
        for(const float angle : {-M_PI_4f,  0.f, M_PI_4f})
            featuremap[angle] = RawImage<float>::Constant(featuresize.y(), featuresize.x(), INFINITY);
        RawImage<float> const& img = distanceTransform<float>(linearray, featuresize);

        featuremap[-M_PI_2f] = img;
        propagateOrientation(featuremap, coeff);

        auto [lineAngle1, feature1] = *featuremap.begin();
        const float distance1 = feature1(0,29);
        REQUIRE(distance1 == 29.f);
        for(auto const& [lineAngle2, feature2] : featuremap)
        {
            const float dist_angle2 = std::abs(constrainHalfAngle(lineAngle1 - lineAngle2));
            const float propagated = distance1 + dist_angle2*coeff;
            REQUIRE(relativelyEqual(propagated, feature2(0,29), 0.f, 1e-5f));
        }
    }
    SECTION("buildCpuFeaturemap")
    {
        const float coeff{50.f};
        const size_t depth{4};
        const float padding{1.f};
        const LineArray scene{
                {0,  0,  0,  0,  1},
                {0,  0,  0, 1, 1},
                {0, 1, 1, 1,  1},
                {1, 1, 0,  0,  0}
        };
        const Dt3Cpu& featuremap = buildCpuFeaturemap(scene, Dt3CpuParameters{depth, coeff, padding});

        for(auto const& line : scene.colwise())
        {
            const auto& [angle, feature] = *closestOrientation(featuremap.getDt3Map(), line);
            const Eigen::Matrix<int,2,1> pt1 = p1(line).array().round().cast<int>();
            const Eigen::Matrix<int,2,1> pt2 = p2(line).array().round().cast<int>();
            const float distance = std::abs(feature(pt2.y(), pt2.x()) - feature(pt1.y(), pt1.x()));
            REQUIRE(distance <= 1.f);
        }
    }
    SECTION("buildCpuFeaturemap: precision")
    {
        const LineArray scene{
                {2},
                {0},
                {5},
                {0}
        };
        const Dt3Cpu& featuremap = buildCpuFeaturemap(scene, Dt3CpuParameters{4, 1, 2.0});
        // Validation:
        const auto& [lineAngle, feature] = *closestOrientation(featuremap.getDt3Map(), scene);
        REQUIRE(allClose(feature.row(int(feature.rows()/2)), Eigen::Array<float, 1, 7>{2,3,3,3,3,3,4}, 0.0, 1e-5));
    }
    SECTION("buildCpuFeaturemap: precision + scaling distance")
    {
        const float distanceScale{2.f};
        LineArray scene{
                {2},
                {0},
                {5},
                {0}
        };
        scene *= distanceScale;
        const Dt3Cpu& featuremap = buildCpuFeaturemap(scene, Dt3CpuParameters{4, 1, 2.0});
        // Validation:
        const auto& [lineAngle, feature] = *closestOrientation(featuremap.getDt3Map(), scene);
        REQUIRE(allClose(feature.row(int(feature.rows()/2)), Eigen::Array<float, 1, 13>{3,5,6,6,6,6,6,6,6,6,7,9,12}, 0.0, 1e-5));
    }
}

