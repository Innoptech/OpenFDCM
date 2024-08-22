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
#include "openfdcm/matching/searchstrategies/defaultsearch.h"
#include "openfdcm/matching/searchstrategies/concentricrange.h"

#include "test-utils/utils.h"

using namespace openfdcm::core;
using namespace openfdcm::matching;

TEST_CASE("Default search strategy", "[openfdcm::matching]")
{
    SECTION("sliceVector")
    {
        std::vector<int> const initial_vec{-2, -1, 0, 1, 2, 3, 4, 5};
        std::vector<int> const slice_indices{0, 2, 4, 6};
        std::vector<int> const &result_vec = sliceVector(initial_vec, slice_indices);
        REQUIRE(result_vec == std::vector<int>{-2, 0, 2, 4});
    }SECTION("DefaultSearch")
    {
        const LineArray scene{
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {1, 2, 3, 6, 5},
                {0, 0, 0, 0, 0}
        };
        const LineArray tmpl{
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {2, 3, 1, 8},
                {0, 0, 0, 0}
        };
        const std::vector<SearchCombination> expected{
                SearchCombination{3, 3}, SearchCombination{3, 4},
                SearchCombination{1, 2}, SearchCombination{1, 4}
        };

        const SearchStrategy strategy = DefaultSearch(2, 2);
        const std::vector<SearchCombination> combinations = strategy.establishSearchStrategy(tmpl, scene);

        // Order does not matter
        REQUIRE(
                std::all_of(combinations.begin(), combinations.end(), [&expected](SearchCombination const &comb) {
                    return std::find(expected.begin(), expected.end(), comb) != expected.end();
                })
        );
    }
}

TEST_CASE("ConcentricRangeStrategy", "[openfdcm::matching]"){
    SECTION("getCenteredRange")
    {
        const CenteredRange range1 = getCenteredRange(30, 60, 60);
        REQUIRE(range1.begin == 0);
        REQUIRE(range1.end == 60);

        const CenteredRange range2 = getCenteredRange(3, 6, 10);
        REQUIRE(range2.begin == 0);
        REQUIRE(range2.end == 6);

        const CenteredRange range3 = getCenteredRange(0, 6, 2);
        REQUIRE(range3.begin == 0);
        REQUIRE(range3.end == 2);

        const CenteredRange range4 = getCenteredRange(5, 6, 2);
        REQUIRE(range4.begin == 4);
        REQUIRE(range4.end == 6);
    }SECTION("filterInRange")
    {
        Point2 img_center{2.5, 2.5};
        float const min_radius{0.f}, max_radius{2.f};
        const LineArray tmpl{
                {0, 2, 0, 0, 0, 3, 4.f},
                {0, 2, 0, 0, 0, 3, 0},
                {5, 4, 5, 0, 2, 4, 5},
                {5, 4, 0, 5, 2, 4, 5}
        };
        std::vector<long> const &filtered_tmpl = filterInRange(tmpl, img_center, min_radius, max_radius);
        REQUIRE(filtered_tmpl == std::vector<long>{0, 1, 5});
    }
    SECTION("Empty scene")
    {
        const LineArray scene(4,0);
        const LineArray tmpl{
                {0,0,0,0},
                {0,0,0,0},
                {2,3,1,8},
                {0,0,0,0}
        };
        const SearchStrategy strategy = ConcentricRangeStrategy(2,2,{0,0},5,15);
        const std::vector<SearchCombination> combinations = strategy.establishSearchStrategy(tmpl, scene);
        REQUIRE(combinations.empty());
    }
    SECTION("Empty template")
    {
        const LineArray tmpl(4,0);
        const LineArray scene{
                {0,0,0,0},
                {0,0,0,0},
                {2,3,1,8},
                {0,0,0,0}
        };
        const SearchStrategy strategy = ConcentricRangeStrategy(2,2,{0,0},5,15);
        const std::vector<SearchCombination> combinations = strategy.establishSearchStrategy(tmpl, scene);
        REQUIRE(combinations.empty());
    }
    SECTION("0-Centered scene")
    {
        const LineArray scene{
                {0,0,0,0,0},
                {0,0,0,0,0},
                {1,13,30,20,5},
                {0,0,0,0,0}
        };
        const LineArray tmpl{
                {0,0,0,0},
                {0,0,0,0},
                {2,3,1,8},
                {0,0,0,0}
        };
        const std::vector<SearchCombination> expected{
                SearchCombination{3,1}, SearchCombination{3,3},
                SearchCombination{1,1}, SearchCombination{1,3}
        };

        const SearchStrategy strategy = ConcentricRangeStrategy(2,2,{0,0},5,15);
        const std::vector<SearchCombination> combinations = strategy.establishSearchStrategy(tmpl, scene);

        // Order does not matter
        REQUIRE(
                std::all_of(combinations.begin(), combinations.end(), [&expected](SearchCombination const& comb)
                {
                    return std::find(expected.begin(), expected.end(), comb) != expected.end();
                })
        );
    }
    SECTION("Offset centered scene")
    {
        const LineArray scene{
                {0,2,4,7},
                {0,0,0,0},
                {2,4,7,15},
                {0,0,0,0}
        };
        const LineArray tmpl{
                {0},
                {0},
                {2},
                {0}
        };

        {
            const SearchStrategy strategy = ConcentricRangeStrategy(1, 1, {4, 0}, 0, 2);
            const std::vector<SearchCombination> combinations = strategy.establishSearchStrategy(tmpl, scene);
            REQUIRE(combinations.at(0) == SearchCombination{0, 1});
        }
        {
            const SearchStrategy strategy = ConcentricRangeStrategy(1, 1, {4, 0}, 3, 15);
            const std::vector<SearchCombination> combinations = strategy.establishSearchStrategy(tmpl, scene);
            REQUIRE(combinations.at(0) == SearchCombination{0, 3});
        }
        {
            auto inf = std::numeric_limits<float>::infinity();
            const SearchStrategy strategy = ConcentricRangeStrategy(1, 1, {4, 0}, 3, inf);
            const std::vector<SearchCombination> combinations = strategy.establishSearchStrategy(tmpl, scene);
            REQUIRE(combinations.at(0) == SearchCombination{0, 3});
        }
        {
            const SearchStrategy strategy = ConcentricRangeStrategy(1, 1, {4, 0}, 2, 4);
            const std::vector<SearchCombination> combinations = strategy.establishSearchStrategy(tmpl, scene);
            REQUIRE(combinations.at(0) == SearchCombination{0, 0});
        }
    }
}