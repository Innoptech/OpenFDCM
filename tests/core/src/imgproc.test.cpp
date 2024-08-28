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
#include "openfdcm/core/imgproc.h"
#include "test-utils/utils.h"



#include "iostream"

using namespace openfdcm::core;

TEST_CASE( "rasterizeLine for specific angle", "[openfdcm::core]")
{
    const Line line{8, 8, 11, 8};
    const Point2 rot_point{8, 8};

    SECTION("Rasterize for -PI/2")
    {
        Line const& line_1 = rotate(line, tests::makeRotation(-M_PI_2f), rot_point);
        Eigen::Matrix<Eigen::Index, 2, 4> const expected_1st{
                {8, 8, 8, 8},
                {8, 7, 6, 5}
        };
        REQUIRE(allClose(rasterizeLine(line_1), expected_1st));
    }
    SECTION("Rasterize for -PI/4")
    {
        Line const& line_2 = rotate(line, tests::makeRotation(-M_PI / 4.f), rot_point);
        Eigen::Matrix<Eigen::Index, 2, 3> const expected_2nd{
                {8, 9, 10},
                {8, 7, 6}
        };
        REQUIRE(allClose(rasterizeLine(line_2), expected_2nd));
    }
    SECTION("Rasterize for 0")
    {
        Eigen::Matrix<Eigen::Index, 2, 4> const expected_3rd{
                {8, 9, 10, 11},
                {8, 8, 8,  8}
        };
        REQUIRE(allClose(rasterizeLine(line), expected_3rd));
    }
    SECTION("Rasterize for PI/4")
    {
        Line const& line_4 = rotate(line, tests::makeRotation(M_PI / 4.f), rot_point);
        Eigen::Matrix<Eigen::Index, 2, 3> const expected_4th{
                {8, 9, 10},
                {8, 9, 10}
        };
        REQUIRE(allClose(rasterizeLine(line_4), expected_4th));
    }
    SECTION("Rasterize for PI/2")
        {
        Line const& line_5 = rotate(line, tests::makeRotation(M_PI_2f), rot_point);
        Eigen::Matrix<Eigen::Index, 2, 4> const expected_5th{
                {8, 8, 8,  8},
                {8, 9, 10, 11}
        };
        REQUIRE(allClose(rasterizeLine(line_5), expected_5th));
    }
}

TEST_CASE( "rasterizeLine check for first and last points", "[openfdcm::core]")
{
    SECTION("Rasterize line shorter than 0.5")
    {
        Line const line{{0}, {0}, {0.4}, {0}};
        Eigen::Matrix<Eigen::Index, 2, -1> const& rasterization = rasterizeLine(line);
        REQUIRE(rasterization.cols() == 1);
        REQUIRE(allClose(rasterization.block<2,1>(0,0), Eigen::Matrix<Eigen::Index, 2, 1>{0,0}));
    }
}

TEST_CASE("drawLines", "[openfdcm::core]")
{
    SECTION("Draw line with clipped line (points out of bound)")
    {
        const LineArray linearray1{{-1},{-1},{3},{3}};
        RawImage<uint8_t> img = Eigen::Array<uint8_t,2,2>::Zero();
        REQUIRE_NOTHROW(drawLines(img, linearray1, 1u));
        REQUIRE_FALSE(allClose(Eigen::Array<uint8_t,2,2>::Zero(), img));
    }
    SECTION("Draw line with line out of bound")
    {
        const LineArray linearray2{{1},{-1},{-1},{0}};
        RawImage<uint8_t> img = Eigen::Array<uint8_t,2,2>::Zero();
        REQUIRE_NOTHROW(drawLines(img, linearray2, 1u));
        REQUIRE(allClose(Eigen::Array<uint8_t,2,2>::Zero(), img));
    }
    SECTION("Empty line array")
    {
        const LineArray linearray{};
        RawImage<uint8_t> img = Eigen::Array<uint8_t,2,2>::Zero();
        RawImage<uint8_t> drawn_img = img.eval();
        drawLines(drawn_img, linearray, 1u);
        REQUIRE(allClose(img, drawn_img));
    }
    SECTION("Horizontal Line")
    {
        const LineArray linearray{{2},{0},{5},{0}};
        RawImage<uint8_t> img = Eigen::Array<uint8_t,1,7>::Zero();
        drawLines(img, linearray, 1u);
        REQUIRE(allClose(img, Eigen::Array<uint8_t,1,7>{0,0,1,1,1,1,0}));
    }
    SECTION("Vertical Line")
    {
        const LineArray linearray{{0},{2},{0},{5}};
        RawImage<uint8_t> img = Eigen::Array<uint8_t,7,1>::Zero();
        drawLines(img, linearray, 1u);
        REQUIRE(allClose(img, Eigen::Array<uint8_t,7,1>{0,0,1,1,1,1,0}));
    }
    SECTION("Diagonal Line")
    {
        const LineArray linearray{{1},{1},{3},{3}};
        auto img = Eigen::Matrix<uint8_t,5,5>::Zero().eval();
        drawLines(img, linearray, 1u);
        auto expected_img = Eigen::Matrix<uint8_t,5,5>::Identity().eval();
        expected_img(0,0) = 0u; expected_img(4,4) = 0u;
        REQUIRE(allClose(img, expected_img));
    }
}

TEST_CASE( "lineIntegral", "[openfdcm::core]")
    {
    SECTION("Various orientations")
    {
        const Line line{8,8,11,8};
        const Point2 rot_point{8,8};

        // Evaluate the line integral for various line orientation
        for(const float angle : {-M_PI_2f, -M_PI_4f,  0.f, M_PI_4f, (M_PI_2f-1e-4f)})
        {
            const Line line_rot = rotate(line, tests::makeRotation(angle), rot_point);
            RawImage<float> img = RawImage<float>::Zero(20,20);
            drawLines(img, line_rot, 1.f);
            lineIntegral(img, angle);
            float const max = img.maxCoeff();
            REQUIRE((max == 4.f or max == 3.f));
        }
    }
}

// Helper function template for testing different distance types
template <Distance DistanceType>
void testDistanceTransform(const Eigen::Array<float,1,4>& expected_single_point,
                           const Eigen::Array<float,1,8>& expected_line) {

    SECTION("Test for validity") {
        const LineArray scene{{0}, {0}, {0}, {9}};
        auto dist_trans = RawImage<float>::Constant(10,5, std::numeric_limits<float>::max()).eval();
        drawLines(dist_trans, scene, static_cast<float>(0));
        distanceTransform<DistanceType>(dist_trans);
        REQUIRE(dist_trans.col(0).sum() == 0);
        for (Eigen::Index i{0}; i < dist_trans.cols(); ++i) {
            REQUIRE(dist_trans.col(i).isApproxToConstant(std::pow(i, DistanceType == Distance::L2_SQUARED ? 2 : 1)));
        }
        REQUIRE(relativelyEqual(dist_trans.col(1).sum(), (float)dist_trans.rows(), 0.f, 1e-5f));
    }

    SECTION("Test for line") {
        const LineArray scene{{2}, {0}, {5}, {0}};
        auto line_dist_trans = RawImage<float>::Constant(2,8, std::numeric_limits<float>::max()).eval();
        drawLines(line_dist_trans, scene, static_cast<float>(0));
        distanceTransform<DistanceType>(line_dist_trans);
        REQUIRE(allClose(line_dist_trans.row(0), expected_line, 0.f, 1e-5f));
    }

    SECTION("Test for single point") {
        const LineArray scene{{2}, {0}, {2}, {0}};
        auto single_pt_dist_trans = RawImage<float>::Constant(1,4, std::numeric_limits<float>::max()).eval();
        drawLines(single_pt_dist_trans, scene, static_cast<float>(0));
        distanceTransform<DistanceType>(single_pt_dist_trans);
        std::cout << single_pt_dist_trans <<"\n";
        REQUIRE(allClose(single_pt_dist_trans.row(0), expected_single_point, 0.f, 1e-5f));
    }
}

// Test cases using the helper function
TEST_CASE("distanceTransform L2", "[openfdcm::core]") {
    Eigen::Array<float,1,4> expected_single_point{2, 1, 0, 1};
    Eigen::Array<float,1,8> expected_line{2, 1, 0, 0, 0, 0, 1, 2};
    testDistanceTransform<Distance::L2>(expected_single_point, expected_line);
}

TEST_CASE("distanceTransform L1", "[openfdcm::core]") {
    Eigen::Array<float,1,4> expected_single_point{2, 1, 0, 1};
    Eigen::Array<float,1,8> expected_line{2, 1, 0, 0, 0, 0, 1, 2};
    testDistanceTransform<Distance::L1>(expected_single_point, expected_line);
}

TEST_CASE("distanceTransform L2_SQUARED", "[openfdcm::core]") {
    Eigen::Array<float,1,4> expected_single_point{4, 1, 0, 1};
    Eigen::Array<float,1,8> expected_line{4, 1, 0, 0, 0, 0, 1, 4};
    testDistanceTransform<Distance::L2_SQUARED>(expected_single_point, expected_line);
}