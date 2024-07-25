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

#ifndef OPENFDCM_IMGPROC_H
#define OPENFDCM_IMGPROC_H
#include "openfdcm/core/drawing.h"
#include "openfdcm/core/error.h"
#include "openfdcm/core/math.h"

namespace openfdcm::core
{
    /**
     * @brief Perform the line integral of a greyscale image
     * @param src The inputoutput image
     * @param lineAngle The integration lineAngle
     */
     template<typename Derived>
    inline void lineIntegral(Eigen::DenseBase<Derived>& img, float const &lineAngle) noexcept
    {
        using T = typename Derived::Scalar;
        Point2 const& rastvec = rasterizeVector(Point2{std::cos(lineAngle), std::sin(lineAngle)});

        Eigen::Array<long,2,1> p0{0,0};
        if (rastvec.x() < 0)
            p0.x() += img.cols()-1;
        if (rastvec.y() < 0)
            p0.y() += img.rows()-1;

        if(std::abs(rastvec.x()) == 1)
        {
            // For best cache efficiency, keep image in column major storage
            Eigen::Index previous_p1x{p0.x()};
            for (Eigen::Index i{1};i<img.cols();++i)
            {
                Eigen::Array<Eigen::Index,2,1> const p1{
                    p0.x()+i*Eigen::Index(rastvec.x()), // rastvec.x() is always 1.0
                    roundf((float(i)*rastvec.y())) - roundf((float(i-1)*rastvec.y()))
                };
                Eigen::Index const y1(std::max((Eigen::Index)p1.y(),Eigen::Index(0)));
                Eigen::Index const y2(std::max((Eigen::Index)-p1.y(),Eigen::Index(0)));
                Eigen::Index const col_len = img.rows() - (size_t)std::abs(p1.y());
                img.block(y1, p1.x(), col_len, 1) += img.block(y2, previous_p1x, col_len, 1);
                previous_p1x = p1.x();
            }
            return;
        }
        // rastvec.y() == 1
        // For best cache efficiency, the image needs to be copied to row major storage
        Eigen::Array<T, -1, -1, Eigen::RowMajor> rowmaj_img = img;
        Eigen::Index previous_p1y{p0.y()};
        for (Eigen::Index i{1};i<img.rows();++i)
        {
            Eigen::Array<Eigen::Index,2,1> const p1{
                    roundf((float(i)*rastvec.x())) - roundf((float(i-1)*rastvec.x())),
                    p0.y()+i*Eigen::Index(rastvec.y()), // rastvec.y() is always 1.0
            };
            Eigen::Index const x1(std::max((Eigen::Index)p1.x(),Eigen::Index(0)));
            Eigen::Index const x2(std::max((Eigen::Index)-p1.x(),Eigen::Index(0)));
            Eigen::Index const row_len = img.cols() - std::abs(p1.x());
            rowmaj_img.block(p1.y(), x1, 1, row_len) += rowmaj_img.block(previous_p1y, x2, 1, row_len);
            previous_p1y = p1.y();
        }
        img = rowmaj_img;
    }

    /**
     * @brief Performs a 1D pass of the algorithm Distance Transforms of Sampled Functions
     * @tparam Derived The type of the image
     * @param img The resulting distance transform as an image
     */
    template<typename Derived>
    static inline void _distanceTransformColumnPass(Eigen::DenseBase<Derived>& img) noexcept
    {
        using T = typename Derived::Scalar;
        using IdxArray = Eigen::Array<Eigen::Index, 1, -1>;
        IdxArray const square_idx = IdxArray::LinSpaced(img.rows(), 0, img.rows()-1).square();

#pragma omp parallel default(none) shared(square_idx, img)
        {
#pragma omp for
            for (Eigen::Index i=0; i < img.cols(); ++i) {
                Eigen::Index k{0};
                std::vector<Eigen::Index> v(img.rows());
                std::vector<T> z(img.rows() + 1);
                v[0] = 0;
                z[0] = -std::numeric_limits<T>::infinity();
                z[1] = std::numeric_limits<T>::infinity();
                // Image has colmajor storage order
                for (Eigen::Index q{1}; q < img.rows(); ++q) {
                    while (true) {
                        Eigen::Index const v_k = v.at(k);
                        T const s = (img(q, i) + square_idx(q) - img(v_k, i) - square_idx(v_k)) / (2 * q - 2 * v_k);
                        if (s>z.at(k))
                        {
                            ++k;
                            v[k] = q;
                            z[k] = s;
                            z[k + 1] = std::numeric_limits<T>::infinity();
                            break;
                        }
                        --k;
                    }
                }
                k = 0;
                for (Eigen::Index q{0}; q < img.rows(); ++q) {
                    while (z.at(k + 1) < (T) q) ++k;
                    Eigen::Index const v_k = v.at(k);
                    Eigen::Index const q_ = q - v_k;
                    img(q, i) = img(v_k, i) + square_idx(std::abs(q_));
                }
            }
        }
    }


    /**
     * @brief Perform the distance transform given a set of lines
     * Use the algorithm Distance Transforms of Sampled Functions
     * @tparam T The required image type
     * @param linearray The given set of lines
     * @param size The required size of the image
     * @return The distance transform as an image
     */
    template<typename T>
    inline RawImage<T> distanceTransform(LineArray const& linearray, Size const& size) noexcept
    {
        static_assert(std::numeric_limits<T>::has_infinity);

        RawImage<T> colmaj_img = RawImage<T>::Constant(size.y(), size.x(), std::numeric_limits<T>::max());
        drawLines(colmaj_img, linearray, 0);

        // For best cache efficiency during column iteration, keep image in column major storage
        _distanceTransformColumnPass(colmaj_img);
        colmaj_img = colmaj_img.transpose().eval();
        _distanceTransformColumnPass(colmaj_img);
        return colmaj_img.transpose().sqrt().eval();
    }
} //namespace openfdcm
#endif //OPENFDCM_IMGPROC_H
