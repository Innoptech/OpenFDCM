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
#include "openfdcm/core/drawing.h"

using namespace openfdcm::core;


TEST_CASE("Cropping line segments based on Box", "[clipLines]") {
    Box Box{0.0f, 10.0f, 0.0f, 10.0f}; // Define a Box for testing

    SECTION("Completely inside Box")
    {
        const Eigen::Matrix<float,4,1> line{2.0f, 3.0f, 7.0f, 8.0f};
        LineArray expected_cropped_lines = line;
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }

    SECTION("Partially inside Box")
    {
        const Eigen::Matrix<float,4,1> line{-2.0f, 1.0f, 7.0f, 1.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{0.0f, 1.0f, 7.0f, 1.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }

    SECTION("Partially inside Box with outside points in X")
    {
        const Eigen::Matrix<float,4,1> line{-2.0f, 1.0f, 12.0f, 1.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{0.0f, 1.0f, 10.0f, 1.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }

    SECTION("Crossing Box with outside points in Y")
    {
        const Eigen::Matrix<float,4,1> line{1.0f, -12.0f, 1.0f, 9.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{1.0f, 0.0f, 1.0f, 9.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }
    SECTION("Crossing Box corner0 -> out corner 2")
    {
        // 1 |¯¯¯¯¯¯¯| 2
        //   |       |
        // 0 |_______| 3
        const Eigen::Matrix<float,4,1> line{-2.0f, -2.0f, 12.0f, 12.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{0.0f, 0.0f, 10.0f, 10.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }
    SECTION("Crossing Box corner1 -> out corner 3")
    {
        const Eigen::Matrix<float,4,1> line{-2.0f, 12.0f, 12.0f, -2.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{0.0f, 10.0f, 10.0f, 0.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }
    SECTION("Crossing Box corner2 -> out corner 0")
    {
        const Eigen::Matrix<float,4,1> line{12.0f, 12.0f, -2.0f, -2.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{10.0f, 10.0f, 0.0f, 0.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }
    SECTION("Crossing Box corner3 -> out corner 1")
    {
        const Eigen::Matrix<float,4,1> line{-2.0f, 12.0f, 12.0f, -2.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{0.0f, 10.0f, 10.0f, 0.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }
    SECTION("Crossing Box side 01 -> side 23")
    {
        const Eigen::Matrix<float,4,1> line{-2.0f, 5.0f, 12.0f, 5.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{0.0f, 5.0f, 10.0f, 5.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }
    SECTION("Crossing Box side 23 -> side 01")
    {
        const Eigen::Matrix<float,4,1> line{12.0f, 5.0f, -2.0f, 5.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{10.0f, 5.0f, 0.0f, 5.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }
    SECTION("Crossing Box side 12 -> side 03")
    {
        const Eigen::Matrix<float,4,1> line{5.0f, 12.0f, 5.0f, -2.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{5.0f, 10.0f, 5.0f, 0.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }
    SECTION("Crossing Box side 03 -> side 12")
    {
        const Eigen::Matrix<float,4,1> line{5.0f, -2.0f, 5.0f, 12.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{5.0f, 0.0f, 5.0f, 10.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }
    SECTION("Completely outside Box")
    {
        const Eigen::Matrix<float,4,1> line{-2.0f, -3.0f, -7.0f, -8.0f};
        REQUIRE(clipLines(line, Box).size() == 0);
    }
    SECTION("Completely outside Box with option deleteOob=false")
    {
        const Eigen::Matrix<float,4,1> line{-2.0f, -3.0f, -7.0f, -8.0f};
        REQUIRE(clipLines(line, Box, false).cols() == 1);
    }
    SECTION("Crossing Box corner0 -> out corner 2 overflow")
    {
        const Eigen::Matrix<float,4,1> line{-2000001.0f, -2000001.0f, 12000001.0f, 12000001.0f};
        const Eigen::Matrix<float,4,1> expected_cropped_lines{0.0f, 0.0f, 10.0f, 10.0f};
        REQUIRE(clipLines(line, Box) == expected_cropped_lines);
    }
}