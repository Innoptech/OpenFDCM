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
#include "openfdcm/core/math.h"

using namespace openfdcm::core;

TEST_CASE( "Operation on vectors and arrays" )
{
    SECTION("argsort on vector")
    {
        const Eigen::Matrix<int,1,4> unsorted{-4, 3, -1, 2};
        std::vector<long> expected_sorted_arg{1,3,2,0};
        REQUIRE(argsort(unsorted,std::greater<>()) == expected_sorted_arg);
        std::reverse(expected_sorted_arg.begin(), expected_sorted_arg.end());
        REQUIRE(argsort(unsorted, std::less<>()) == expected_sorted_arg);
    }
    SECTION("binarySearch")
    {
        const Eigen::Vector<int, 10> vec{0,2,3,6,7,10,14,30,40,123};
        REQUIRE(binarySearch(vec, 0) == 0);
        REQUIRE(binarySearch(vec, 123) == 9);
        REQUIRE(binarySearch(vec, 2) == 1);
        REQUIRE(binarySearch(vec, 40) == 8);
        REQUIRE(binarySearch(vec, 5) == 3);
        REQUIRE(binarySearch(vec, 4) == 2);
    }
    SECTION("minmaxPoint")
    {
        const LineArray linearray{
                {0,  0, 0, 0},
                {-4, 0, 0, 0},
                {0,  2, 8, 0},
                {0,  0, 8, 16}
        };
        auto const& [min_point, max_point] = minmaxPoint(linearray);
        REQUIRE(allClose(min_point, Point2{0,-4}));
        REQUIRE(allClose(max_point, Point2{8,16}));
    }

}
TEST_CASE( "Operation on angles" )
{
    SECTION("constrainAngle")
    {
        REQUIRE(constrainAngle(M_PI*3) == -M_PI);
        Eigen::Vector4d constrained_angles = constrainAngle(Eigen::Vector4d{M_PI*3, 0, M_PI_4*3, 100*M_PI});
        REQUIRE(allClose(constrained_angles, Eigen::Vector4d{-M_PI, 0, M_PI_4*3, 0}, 0.0, 1e-10));
    }
    SECTION("constrainHalfAngle")
    {
        REQUIRE(constrainHalfAngle(M_PI*3) == 0.0);
        Eigen::Vector4d constrained_angles = constrainHalfAngle(Eigen::Vector4d{M_PI*3, 0, M_PI_4*3, -M_PI_4});
        REQUIRE(allClose(constrained_angles, Eigen::Vector4d{0.0, 0, -M_PI_4, -M_PI_4}, 0.0, 1e-10));
        Eigen::Vector4d constrained_angles2 = constrainHalfAngle(Eigen::Vector4d{-M_PI*3, M_PI_4, -M_PI_4*3, 100*M_PI});
        REQUIRE(allClose(constrained_angles2, Eigen::Vector4d{0.0, M_PI_4, M_PI_4, 0.0}, 0.0, 1e-10));
    }
}
TEST_CASE( "Operation on lines")
{
    const LineArray linearray{
            {0, 0, -1},
            {0, 0, 1},
            {2, 1, 0},
            {2, 0, 0}
    };

    SECTION("p1 & p2")
    {
        const Line l1{0, 1, 2, 3};
        REQUIRE(allClose(p1(l1), Point2{0, 1}));
        REQUIRE(allClose(p2(l1), Point2{2, 3}));
    }
    SECTION("center")
    {
        Eigen::Matrix<float, 2, 3> expected_center{
                {1.0, 0.5f, -0.5},
                {1.f, 0.0, 0.5}
        };
        REQUIRE(allClose(getCenter(linearray), expected_center));
    }
    SECTION("lineAngle")
    {
        Eigen::Matrix<float, 1, 3> expected_angles{M_PI / 4, 0.f, -M_PI / 4};
        REQUIRE(allClose(getAngle(linearray), expected_angles));
    }
    SECTION("length")
    {
        Eigen::Matrix<float, 1, 3> expected_lengths{std::sqrt(8.f), 1.f, std::sqrt(2.f)};
        REQUIRE(allClose(getLength(linearray), expected_lengths));
    }
    SECTION("getTemplateLengths")
    {
        auto lengths = getTemplateLengths({linearray,linearray,linearray});
        for(auto len : lengths)
            REQUIRE(len == getLength(linearray).sum());
    }
    SECTION("normalize")
    {
        Eigen::Matrix<float, 2, 3> expected_normalized_lines{
                {std::sqrt(2.f) / 2, 1.0, std::sqrt(2.f) / 2},
                {std::sqrt(2.f) / 2, 0.0, -std::sqrt(2.f) / 2}
        };
        REQUIRE(allClose(normalize(linearray), expected_normalized_lines));
    }
}

TEST_CASE("transform lines")
{
    SECTION("transform")
    {
        const LineArray linearray{
                {0, 0, 1, -1},
                {0, 0, 1, -2},
                {0, 1, 2, -3},
                {1, 0, 2, 4}
        };
        const Mat23 transform_mat{
                {-1, 0,  1},
                {0,  -1, 2}
        };
        const LineArray expected{
                {1, 1, 0,  2},
                {2, 2, 1,  4},
                {1, 0, -1, 4},
                {1, 2, 0,  -2}
        };
        const LineArray &result = transform(linearray, transform_mat);
        REQUIRE(allClose(result, expected));
    }
    SECTION("rotate around the origin")
    {
        const Mat22 rotation_mat{
                {-1, 0},
                {0,  -1}
        };
        const LineArray original{
                {0, 1},
                {0, 1},
                {1, 2},
                {1, 2}
        };
        const LineArray expected{
                {0,  -1},
                {0,  -1},
                {-1, -2},
                {-1, -2}
        };
        REQUIRE(allClose(rotate(original, rotation_mat), expected));
    }
    SECTION("rotate around point")
    {
        const Mat22 rotation_mat{
                {-1, 0},
                {0,  -1}
        };
        const LineArray original{
                {0, 1},
                {0, 0},
                {0, 2},
                {1, 0}
        };
        const Point2 rot_point{1, 0};
        const LineArray expected{
                {2,  1},
                {0,  0},
                {2,  0},
                {-1, 0}
        };
        REQUIRE(allClose(rotate(original, rotation_mat, rot_point), expected));
    }
    SECTION("translate")
    {
        const Point2 translation{1, 2};
        const LineArray original{
                {0, 1},
                {0, 1},
                {1, 2},
                {1, 2}
        };
        const LineArray expected{
                {1, 2},
                {2, 3},
                {2, 3},
                {3, 4}
        };
        const LineArray &translated = translate(original, translation);
        REQUIRE(allClose(translated, expected));
    }
    SECTION("combine")
    {
        const Mat23 transform{
                {-1, 0, 1},
                {0, -1, 2}
        };
        const Point2 translation{3, 4};
        const Mat23 expected{
                {-1, 0, 1-3},
                {0, -1, 2-4}
        };
        const Mat23 result = combine(transform, translation);
        REQUIRE(allClose(result, expected));
    }
    SECTION("aling")
    {
        const LineArray linearray{
                {0,  0, 0, 0},
                {-4, 0, 0, 0},
                {0,  2, 8, 0},
                {0,  0, 8, 16}
        };
        const Line alignment_line{-1,-1,1,1};
        const Point2& align_vec = normalize(alignment_line);

        for(Mat23 const& transf : aling(getLine(linearray,0), alignment_line))
        {
            const LineArray& alligned_linearray = transform(linearray, transf);
            const float angle_diff = getAngle(alignment_line)(0,0) - getAngle(getLine(linearray,0))(0,0);
            // Line at idx 0 must have the same center than the alignment line
            REQUIRE(allClose(getCenter(alignment_line), getCenter(getLine(alligned_linearray,0))));
            REQUIRE(getAngle(Line{0,0,align_vec.x(),align_vec.y()}) == getAngle(alignment_line));
            // All lines must have been turned by the same lineAngle
            REQUIRE(allClose(getAngle(alligned_linearray), constrainHalfAngle(getAngle(linearray).array()+angle_diff)));
        }
    }
}

TEST_CASE("matching::rasterizeVector")
{
    Point2 const vec{2.f,0.f};
    float const eps{M_PI/12};
    float const tan60{1.f/std::sqrt(3.f)};
    SECTION("rasterize for angle -PI/4 - epsilon")
    {
        Mat22 const rot = tests::makeRotation(-M_PI_4f-eps);
        REQUIRE(allClose(rasterizeVector(rot*vec), Point2{tan60,-1.f}));
    }
    SECTION("rasterize for angle -PI/4 + epsilon")
    {
        Mat22 const rot = tests::makeRotation(-M_PI_4f+eps);
        REQUIRE(allClose(rasterizeVector(rot*vec), Point2{1.f,-tan60}));
    }
    SECTION("rasterize for angle PI/4 - epsilon")
    {
        Mat22 const rot = tests::makeRotation(M_PI_4-eps);
        REQUIRE(allClose(rasterizeVector(rot*vec), Point2{1.f,tan60}));
    }
    SECTION("rasterize for angle PI/4 + epsilon")
    {
        Mat22 const rot = tests::makeRotation(M_PI_4f+eps);
        REQUIRE(allClose(rasterizeVector(rot*vec), Point2{tan60,1.f}));
    }
    SECTION("rasterize for angle 3PI/4 - epsilon")
    {
        Mat22 const rot = tests::makeRotation(3*M_PI_4f-eps);
        REQUIRE(allClose(rasterizeVector(rot*vec), Point2{-tan60,1.f}));
    }
    SECTION("rasterize for angle 3PI/4 + epsilon")
    {
        Mat22 const rot = tests::makeRotation(3*M_PI_4f+eps);
        REQUIRE(allClose(rasterizeVector(rot*vec), Point2{-1.f,tan60}));
    }
    SECTION("rasterize for angle -3PI/4 - epsilon")
    {
        Mat22 const rot = tests::makeRotation(-3*M_PI_4f-eps);
        REQUIRE(allClose(rasterizeVector(rot*vec), Point2{-1.f,-tan60}));
    }
    SECTION("rasterize for angle -3PI/4 + epsilon")
    {
        Mat22 const rot = tests::makeRotation(-3*M_PI_4f+eps);
        REQUIRE(allClose(rasterizeVector(rot*vec), Point2{-tan60,-1.f}));
    }
    SECTION("Null vector")
    {
        REQUIRE(rasterizeVector(Point2{0,0}).array().isNaN().any());
    }
}