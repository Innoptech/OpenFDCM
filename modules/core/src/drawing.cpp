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

#include "openfdcm/core/drawing.h"

namespace openfdcm::core
{
    enum class OutCode {
        INSIDE = 0, // 0000
        LEFT = 1,   // 0001
        RIGHT = 2,  // 0010
        BOTTOM = 4, // 0100
        TOP = 8     // 1000
    };

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

    void clipAgainstY(Eigen::Ref<Eigen::Vector2f> p1, Eigen::Ref<Eigen::Vector2f> p2, float y_crop) {
        p1.x() = p1.x() + (p2.x() - p1.x()) * (y_crop - p1.y()) / (p2.y() - p1.y());
        p1.y() = y_crop;
    }

    void clipAgainstX(Eigen::Ref<Eigen::Vector2f> p1, Eigen::Ref<Eigen::Vector2f> p2, float x_crop) {
        p1.y() = p1.y() + (p2.y() - p1.y()) * (x_crop - p1.x()) / (p2.x() - p1.x());
        p1.x() = x_crop;
    }


    LineArray clipLines(const LineArray &lines, const Box &cropbox, bool deleteOob) {
        LineArray cropped_lines = lines;
        std::vector<int> keep_indices{};
        std::vector<int> purge_indices{};

        for (int i = 0; i < lines.cols(); ++i) {
            auto p1 = cropped_lines.block<2, 1>(0, i); // First 2D point
            auto p2 = cropped_lines.block<2, 1>(2, i); // Second 2D point

            OutCode code1 = computeOutCode(p1, cropbox);
            OutCode code2 = computeOutCode(p2, cropbox);

            while (true) {
                if ((static_cast<int>(code1) == 0) && (static_cast<int>(code2) == 0)) {
                    // If both endpoints lie within rectangle
                    keep_indices.push_back(i);
                    break;
                } else if (static_cast<int>(code1) & static_cast<int>(code2)) {
                    // If both endpoints are outside rectangle in the same region
                    purge_indices.push_back(i);
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
        }

        if(deleteOob)
            return cropped_lines(Eigen::all, keep_indices);

        for(auto purgeIdx : purge_indices)
        {
            cropped_lines.col(purgeIdx).setConstant(0.f);
        }
        return cropped_lines;
    }
}