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
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "openfdcm/matching/optimizestrategies/defaultoptimize.h"
#include "openfdcm/matching/featuremaps/dt3cpu.h"

#include <iostream>

using namespace openfdcm::core;
using namespace openfdcm::matching;


TEST_CASE("DefaultOptimize", "[openfdcm::matching, openfdcm::matching::DefaultOptimize, openfdcm::matching::Dt3Cpu]")
{
    OptimizeStrategy optimizer = DefaultOptimize{1};

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
        Mat23 const transf{
                {1,0,5},
                {0,1,0}
        };

        const Dt3Cpu& featuremap = buildCpuFeaturemap(scene, Dt3CpuParameters{4, 1, 1.f});

        auto optimal_translation = optimize(optimizer, {transform(tmpl, transf)}, {align_vec}, featuremap).at(0);
        REQUIRE(optimal_translation.has_value());

        OptimalTranslation const& optrans = optimal_translation.value();
        REQUIRE(allClose(optrans.translation, Point2{0,0}));
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
                {3,0},
                {0,1},
                {6,7},
                {0,1}
        };
        // The tmpl line is already aligned but not centered with the scene line
        // The minimal score correspond to the right most position where the tmpl line overlap all the scene line
        Point2 const align_vec{1,0};

        const Dt3Cpu& featuremap = buildCpuFeaturemap(scene, Dt3CpuParameters{4, 1, 1.f});

        auto optimal_translation = optimize(optimizer, {tmpl}, {align_vec}, featuremap).at(0);
        REQUIRE(optimal_translation.has_value());

        OptimalTranslation const& optrans = optimal_translation.value();
        REQUIRE(allClose(optrans.translation, Point2{1,0}));
        REQUIRE_THAT(optrans.score, Catch::Matchers::WithinRel(1.f));
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
        const Dt3Cpu& featuremap = buildCpuFeaturemap(scene, Dt3CpuParameters{4, 1, 1.f});

        auto optimal_translation = optimize(optimizer, {tmpl}, {align_vec}, featuremap).at(0);
        REQUIRE_FALSE(optimal_translation.has_value());
    }
}

