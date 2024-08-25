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

#include "openfdcm/core/cuda/cudrawing.cuh"

namespace openfdcm::core::cuda {
    enum class OutCode {
        INSIDE = 0, // 0000
        LEFT = 1,   // 0001
        RIGHT = 2,  // 0010
        BOTTOM = 4, // 0100
        TOP = 8     // 1000
    };

    __device__
    OutCode computeOutCode(const Eigen::Vector2f& point, const Box& cropbox) {
        auto code = static_cast<int>(OutCode::INSIDE);

        if (point(0) < cropbox.xmin) {
            code |= static_cast<int>(OutCode::LEFT);
        } else if (point(0) > cropbox.xmax) {
            code |= static_cast<int>(OutCode::RIGHT);
        }
        if (point(1) < cropbox.ymin) {
            code |= static_cast<int>(OutCode::BOTTOM);
        } else if (point(1) > cropbox.ymax) {
            code |= static_cast<int>(OutCode::TOP);
        }
        return static_cast<OutCode>(code);
    }

    __device__
    void clipAgainstY(Eigen::Ref<Eigen::Vector2f> p1, Eigen::Ref<Eigen::Vector2f> p2, float y_crop) {
        p1.x() = p1.x() + (p2.x() - p1.x()) * (y_crop - p1.y()) / (p2.y() - p1.y());
        p1.y() = y_crop;
    }

    __device__
    void clipAgainstX(Eigen::Ref<Eigen::Vector2f> p1, Eigen::Ref<Eigen::Vector2f> p2, float x_crop) {
        p1.y() = p1.y() + (p2.y() - p1.y()) * (x_crop - p1.x()) / (p2.x() - p1.x());
        p1.x() = x_crop;
    }

    __device__
    Line clipLine(Line const& line, const Box &cropbox) {
        Point2 p1 = line.block<2, 1>(0, 0); // First 2D point
        Point2 p2 = line.block<2, 1>(2, 0); // Second 2D point

        OutCode code1 = computeOutCode(p1, cropbox);
        OutCode code2 = computeOutCode(p2, cropbox);

        while (true) {
            if ((static_cast<int>(code1) == 0) && (static_cast<int>(code2) == 0)) {
                // If both endpoints lie within rectangle
                break;
            } else if (static_cast<int>(code1) & static_cast<int>(code2)) {
                // If both endpoints are outside rectangle in the same region
                break;
            } else { // Line segment may need to be clipped
                if (static_cast<int>(code1) != 0) {
                    if (int(code1) & static_cast<int>(OutCode::TOP)) clipAgainstY(p1, p2, cropbox.ymax);
                    else if (int(code1) & static_cast<int>(OutCode::BOTTOM)) clipAgainstY(p1, p2, cropbox.ymin);
                    else if (int(code1) & static_cast<int>(OutCode::RIGHT)) clipAgainstX(p1, p2, cropbox.xmax);
                    else if (int(code1) & static_cast<int>(OutCode::LEFT)) clipAgainstX(p1, p2, cropbox.xmin);
                    code1 = computeOutCode(p1, cropbox);
                    continue;
                }

                if (int(code2) & static_cast<int>(OutCode::TOP)) clipAgainstY(p2, p1, cropbox.ymax);
                else if (int(code2) & static_cast<int>(OutCode::BOTTOM)) clipAgainstY(p2, p1, cropbox.ymin);
                else if (int(code2) & static_cast<int>(OutCode::RIGHT)) clipAgainstX(p2, p1, cropbox.xmax);
                else if (int(code2) & static_cast<int>(OutCode::LEFT)) clipAgainstX(p2, p1, cropbox.xmin);
                code2 = computeOutCode(p2, cropbox);
            }
        }
        return {p1.x(), p1.y(), p2.x(), p2.y()};
    }
} // namespace openfdcm::core::cuda