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
     * @brief Create a uniformly distributed set of lines in [-PI/2, Pi/2]
     * @param line_number The desired number of lines
     * @param length The length of the lines
     * @return The resulting array of lines
     */
    inline openfdcm::core::LineArray createLines(size_t const line_number, const size_t length) {
        openfdcm::core:: LineArray linearray(4, line_number);
        for(size_t i{0};i<line_number;i++){
            float lineAngle = float(i)/float(line_number)*M_PI-M_PI_2;
            linearray.block<4,1>(0,i) << openfdcm::core::rotate(openfdcm::core::Line{0,0, (float)length, 0}, makeRotation(lineAngle));
        }
        return linearray;
    }
}


#endif //OPENFDCM_UTILS_H
