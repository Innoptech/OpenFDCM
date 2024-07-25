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

#ifndef OPENFDCM_DRAWING_H
#define OPENFDCM_DRAWING_H
#include "openfdcm/core/math.h"
#include "openfdcm/core/error.h"

namespace openfdcm::core
{
    struct Box
    {
        float xmin, xmax, ymin, ymax;
    };

    /**
     * Clip line segments based on a given CropBox.
     *
     * This function takes a set of line segments represented as a core::LineArray and crops each segment
     * based on the provided CropBox. The Cohen-Sutherland line clipping algorithm is used to determine
     * whether a segment lies inside or outside the CropBox and clip it accordingly. Line segments that are
     * completely outside the CropBox are set to zero.
     *
     * @param lines The input set of line segments to be cropped.
     * @param cropbox The box defining the crop zone.
     * @param deleteOob If true, delete the Out of Bound lines. Else replace them by a singular (0,0) point.
     * @return A core::LineArray containing the cropped line segments.
     */
    core::LineArray clipLines(core::LineArray const& lines, const Box &cropbox, bool deleteOob=true);

    /**
     * @brief Rasterize a vector in such a way that the angle is conserved and either x or y has a value of 1.
     * @param align_vec The desired vector to rasterize
     * @return The rasterized vector
     */
    inline Point2 rasterizeVector(Point2 const& align_vec) noexcept
    {
        const float tan_angle = align_vec.y()/align_vec.x();
        if (tan_angle >= -1.0 and tan_angle < 1) // [-PI/4, 0[ U [0, PI/4[ U [-PI, -3PI/4] U [3PI/4, PI]
        {
            bool cond1{align_vec.x() < 0};
            return {1 - 2*cond1, tan_angle - 2.0*cond1*tan_angle};
        }
        bool cond2{align_vec.y() < 0};
        return {1 / tan_angle - 2.0*cond2*(1 / tan_angle), 1 - 2*cond2};
    }

    /**
     * @brief  Rasterize a line between two points
     * @param line The line to rasterize
     * @return The rasterized line expressed as an array of points
     */
    inline Eigen::Array<Eigen::Index, 2, -1> rasterizeLine(Line const& line) noexcept
    {
        if (allClose(p2(line), p1(line)))
            return p1(line).array().round().cast<Eigen::Index>();
        Point2 const& line_vec = p2(line) - p1(line);
        Point2 const& rastvec = rasterizeVector(line_vec);
        if (relativelyEqual(rastvec.x(), 0.0f))
        {
            int const size = int(line_vec.y() / rastvec.y()) + 1;
            Eigen::Matrix<float, 2, -1> rasterization(2, size);
            rasterization.row(0).setConstant(p1(line).x());
            rasterization.row(1) = Eigen::Matrix<float, 1, -1>::LinSpaced(size, p1(line).y(), p2(line).y());
            return rasterization.array().round().cast<Eigen::Index>();
        }
        if (relativelyEqual(rastvec.y(), 0.0f))
        {
            int const size = int(line_vec.x() / rastvec.x()) + 1;
            Eigen::Matrix<float, 2, -1> rasterization(2, size);
            rasterization.row(0) = Eigen::Matrix<float, 1, -1>::LinSpaced(size, p1(line).x(), p2(line).x());
            rasterization.row(1).setConstant(p1(line).y());
            return rasterization.array().round().cast<Eigen::Index>();
        }

        int size = static_cast<int>(std::max(line_vec.x() / rastvec.x(), line_vec.y() / rastvec.y())) + 1;
        Eigen::Matrix<float, 2, -1> rasterization(2, size);
        rasterization.row(0) = Eigen::Matrix<float, 1, -1>::LinSpaced(size, p1(line).x(), p2(line).x());
        rasterization.row(1) = Eigen::Matrix<float, 1, -1>::LinSpaced(size, p1(line).y(), p2(line).y());
        return rasterization.array().round().cast<Eigen::Index>();
    }

    /**
     * @brief Draw a rasterized line on a greyscale image
     * @exception Error: In splitAndDrawLines, at least one of the line points is beyond of image boundaries
     * @param src The inputoutput image
     * @param linearray The array of lines to draw
     * @param color The greyscale color to draw
     */
    template<typename Derived, typename U>
    inline void drawLines(Eigen::DenseBase<Derived>& img, LineArray const& linearray, U const color) noexcept(false)
    {
        if (linearray.cols() == 0) return;

        // Clip lines
        auto const& clipped_lines = clipLines(linearray, Box{0, static_cast<float>(img.cols()-1),
                                                             0, static_cast<float>(img.rows()-1)});
        for (auto const& line : clipped_lines.colwise())
        {
            Eigen::Matrix<Eigen::Index, 2, -1> const& rasterization = rasterizeLine(line);
            for ( auto const& pt : rasterization.colwise())
                img(pt.y(), pt.x()) = (typename Derived::Scalar)color;
        }
    }
} // namespace openfdcm::core
#endif //OPENFDCM_DRAWING_H
