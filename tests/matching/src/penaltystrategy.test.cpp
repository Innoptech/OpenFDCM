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
#include <catch2/catch_template_test_macros.hpp>
#include "test-utils/utils.h"
#include "openfdcm/matching/penaltystrategies/defaultpenalty.h"
#include "openfdcm/matching/penaltystrategies/exponentialpenalty.h"

using namespace openfdcm::core;
using namespace openfdcm::matching;


TEMPLATE_TEST_CASE("PenaltyStrategy Test", "[openfdcm::matching, penalty]", DefaultPenalty, ExponentialPenalty)
{
    using PenaltyType = TestType;

    PenaltyType penalty = []() {
        // Construct the penalty object based on the type
        if constexpr (std::is_same_v<PenaltyType, DefaultPenalty>) {
            // Customize construction for DefaultPenalty
            return DefaultPenalty{};
        } else if constexpr (std::is_same_v<PenaltyType, ExponentialPenalty>) {
            // Customize construction for ExponentialPenalty
            return ExponentialPenalty{2.f};
        }
    }();

    SECTION("Penalize null length")
    {
        LineArray tmpl(4,1); tmpl << 0,0,0,0;
        std::vector templates{tmpl};
        const auto& lengths = getTemplateLengths(templates);

        std::vector<Match> const original_matches{
                Match(0, 1, Mat23{{1,2,3},{4,5,6}}),
        };
        std::vector<Match> const &penalized_matches = penalize(penalty, original_matches, lengths);
        REQUIRE_FALSE(std::isnan(penalized_matches.front().score));
    }
    SECTION("Penalize with inconsistent templatelengths")
    {
        LineArray tmpl = tests::createLines(4, 4);
        std::vector templates{tmpl};
        std::vector<Match> const original_matches{
                Match(0, 1, Mat23{{1,2,3},{4,5,6}}),
                Match(1, 1, Mat23{{4,5,6}, {1,2,3}}),
        };
        REQUIRE_THROWS_AS(penalize(penalty, original_matches, {}), std::out_of_range);
    }
}


TEST_CASE("Validate penalty", "[openfdcm::matching]")
{
    SECTION("DefaultPenalty")
    {
        LineArray tmpl1 = tests::createLines(4, 4);
        LineArray tmpl2 = tests::createLines(3, 3);
        std::vector<LineArray> const templates{
                rotate(tmpl1, Mat22{{-1, 0},
                                    {0,  -1}}),
                translate(tmpl2, Point2{1, 2})
        };
        const auto &lengths = getTemplateLengths(templates);

        std::vector<Match> const original_matches{
                Match(0, 1, Mat23{{1, 2, 3},
                                  {4, 5, 6}}),
                Match(1, 1, Mat23{{4, 5, 6},
                                  {1, 2, 3}}),
        };
        PenaltyStrategy const penalty = DefaultPenalty{};
        std::vector<Match> const &penalized_matches = penalize(penalty, original_matches, lengths);

        REQUIRE(penalized_matches.size() == original_matches.size());
        for (size_t i{0}; i < penalized_matches.size(); ++i) {
            REQUIRE(penalized_matches.at(i).tmplIdx == original_matches.at(i).tmplIdx);
            REQUIRE(relativelyEqual(
                    penalized_matches.at(i).score,
                    original_matches.at(i).score / getLength(templates.at(i)).sum()
            ));
            REQUIRE(allClose(
                    penalized_matches.at(i).transform,
                    original_matches.at(i).transform
            ));
        }
    }
    SECTION("ExponentialPenalty")
    {
        LineArray tmpl1 = tests::createLines(4,4);
        LineArray tmpl2 = tests::createLines(3,3);
        std::vector<LineArray> const templates{
                rotate(tmpl1, Mat22{{-1, 0},{0,-1}}),
                translate(tmpl2, Point2{1,2})
        };
        std::vector<Match> const original_matches{
                Match(0, 1, Mat23{{1,2,3},{4,5,6}}),
                Match(1, 1, Mat23{{4,5,6}, {1,2,3}}),
        };
        const auto& lengths = getTemplateLengths(templates);

        float tau{1.45f};
        PenaltyStrategy const penalty = ExponentialPenalty{tau};
        std::vector<Match> const& penalized_matches = penalize(penalty, original_matches, lengths);

        REQUIRE(penalized_matches.size() == original_matches.size());
        for(size_t i{0};i<penalized_matches.size();++i)
        {
            REQUIRE(penalized_matches.at(i).tmplIdx == original_matches.at(i).tmplIdx);
            REQUIRE(relativelyEqual(
                    penalized_matches.at(i).score,
                    original_matches.at(i).score/std::pow(getLength(templates.at(i)).sum(),tau)
            ));
            REQUIRE(allClose(
                    penalized_matches.at(i).transform,
                    original_matches.at(i).transform
            ));
        }
    }
}