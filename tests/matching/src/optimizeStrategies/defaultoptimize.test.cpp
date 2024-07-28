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

#include "catch2/catch_test_macros.hpp"
#include "openfdcm/matching/optimizestrategies/defaultoptimize.h"
#include "openfdcm/matching/featuremaps/dt3cpu.h"

using namespace openfdcm::core;
using namespace openfdcm::matching;

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

TEST_CASE("DefaultOptimize", "[openfdcm::matching]")
{
    OptimizeStrategy optimizer = DefaultOptimize{};

    SECTION("DefaultOptimize: Perfect optimization")
    {
        const LineArray tmpl{
                {10,0},
                {0,0},
                {10,10},
                {10,0}
        };

        const LineArray scene{
                {15,5},
                {0,0},
                {15,15},
                {10,0}
        };
        // A tmpl line is already aligned and centered with a scene line
        Point2 const align_vec{1,0};

        Dt3Cpu featuremap{scene, Dt3CpuParameters{4, 20*20}};

        auto optimal_translation = optimize(optimizer, tmpl, align_vec, featuremap);
        REQUIRE(optimal_translation.has_value());

        OptimalTranslation const& optrans = optimal_translation.value();
        REQUIRE(allClose(optrans.translation, Point2{5,0}));
        REQUIRE(optrans.score == 0);
    }
    SECTION("DefaultOptimize: Larger template")
    {
        const LineArray tmpl{
                {0},
                {0},
                {5},
                {0}
        };
        const LineArray scene{
                {3},
                {0},
                {6},
                {0}
        };
        // The tmpl line is already aligned but not centered with the scene line
        // The minimal score correspond to the right most position where the tmpl line overlap all the scene line
        Point2 const align_vec{1,0};
        Dt3Cpu featuremap{scene, Dt3CpuParameters{4, 1}};

        auto optimal_translation = optimize(optimizer, tmpl, align_vec, featuremap);
        REQUIRE(optimal_translation.has_value());

        OptimalTranslation const& optrans = optimal_translation.value();
        REQUIRE(allClose(optrans.translation, Point2{1,0}));
        REQUIRE(optrans.score == 1.f);
    }
    SECTION("DefaultOptimize: template out of boundaries")
    {
        const LineArray tmpl{
                {0},
                {0},
                {10},
                {10}
        };
        const LineArray scene{
                {0},
                {0},
                {1},
                {0}
        };
        Point2 const align_vec{1,0};
        Dt3Cpu featuremap{scene, Dt3CpuParameters{4, 1}};

        auto optimal_translation = optimize(optimizer, tmpl, align_vec, featuremap);
        REQUIRE_FALSE(optimal_translation.has_value());
    }
}

