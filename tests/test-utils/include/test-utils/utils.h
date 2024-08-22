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

#ifndef OPENFDCM_UTILS_H
#define OPENFDCM_UTILS_H
#include <cmath>
#include "openfdcm/core/math.h"
#include "openfdcm/core/imgproc.h"

namespace tests
{
    /**
     * @brief Make a 2x2 rotation matrix given a rotation angle
     * @param lineAngle The given angle in radians
     * @return The 2x2 rotation matrix
     */
    inline openfdcm::core::Mat22 makeRotation(const float& lineAngle){
        float sin(std::sin(lineAngle)), cos(std::cos(lineAngle));
        return openfdcm::core::Mat22({
            {cos, -sin},
            {sin, cos}
        });
    }

    /**
     * Generates a logarithmic spaced array of values between start and end.
     *
     * @param start The starting value for the logspace.
     * @param end The ending value for the logspace.
     * @param num The number of values to generate.
     * @return std::vector<float> A vector containing logarithmically spaced values.
     */
    inline std::vector<float> logspace(float start, float end, size_t num) {
        std::vector<float> result(num);
        float log_start = std::log10(start);
        float log_end = std::log10(end);
        float step = (log_end - log_start) / (num - 1);
        for (size_t i = 0; i < num; ++i) {
            result[i] = std::pow(10, log_start + i * step);
        }
        return result;
    }

    /**
     * Generates an array of lines with their endpoints calculated using a logarithmic space of angles to avoid symmetries.
     *
     * @param line_number The number of lines to generate.
     * @param length The length of each line.
     * @return openfdcm::core::LineArray A 4 x line_number matrix where each column represents a line.
     *         The first two rows represent the starting point (always [0, 0]),
     *         and the last two rows represent the endpoint of each line.
     */
    inline openfdcm::core::LineArray createLines(size_t const line_number, const size_t length) {
        openfdcm::core::LineArray linearray(4, line_number);

        // Generate logspace angles
        std::vector<float> line_angles = logspace(2 * M_PI, 4 * M_PI, line_number);

        for (size_t i = 0; i < line_number; ++i) {
            float lineAngle = line_angles[i];
            Eigen::Matrix2f rotation_matrix = makeRotation(lineAngle);
            Eigen::Vector2f endpoint = rotation_matrix * Eigen::Vector2f(float(length), 0);
            linearray.block<4,1>(0, i) << 0.0f, 0.0f, endpoint[0], endpoint[1];
        }
        return linearray;
    }
}


#endif //OPENFDCM_UTILS_H
