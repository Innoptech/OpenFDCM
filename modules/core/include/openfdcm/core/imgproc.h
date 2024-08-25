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
     * @param img The input-output image
     * @param lineAngle The integration lineAngle
     */
    template<typename Derived>
    inline void lineIntegral(Eigen::DenseBase<Derived>& img, float  lineAngle) noexcept
    {
        using T = typename Derived::Scalar;
        Point2 rastvec = rasterizeVector(Point2{std::cos(lineAngle), std::sin(lineAngle)});

        bool transposed{false};
        if (std::abs(rastvec.y()) == 1)
        {
            transposed = true;
            img = img.transpose().eval();
            rastvec = Point2{-rastvec.y(), -rastvec.x()};
        }

        Eigen::Array<long, 2, 1> p0{0, 0};
        if (rastvec.x() < 0)
            p0.x() += img.cols() - 1;
        if (rastvec.y() < 0)
            p0.y() += img.rows() - 1;

        Eigen::Index previous_p1x{p0.x()};
        for (Eigen::Index i{1}; i < img.cols(); ++i)
        {
            Eigen::Array<Eigen::Index, 2, 1> const p1{
                    p0.x() + i * Eigen::Index(rastvec.x()),
                    static_cast<Eigen::Index>(std::round(i * rastvec.y())) - static_cast<Eigen::Index>(std::round((i - 1) * rastvec.y()))
            };
            Eigen::Index const y1 = std::max(p1.y(), Eigen::Index(0));
            Eigen::Index const y2 = std::max(-p1.y(), Eigen::Index(0));
            Eigen::Index const col_len = img.rows() - std::abs(p1.y());
            img.block(y1, p1.x(), col_len, 1) += img.block(y2, previous_p1x, col_len, 1);
            previous_p1x = p1.x();
        }
        if (transposed)
            img = img.transpose().eval();
    }

    /**
     * @brief Performs a 1D pass of the algorithm Distance Transforms of Sampled Functions for L2_SQUARED distance
     * @tparam Derived The type of the image
     * @param img The resulting distance transform as an image
     */
    template<typename Derived>
    static inline void _distanceTransformColumnPassL2(Eigen::DenseBase<Derived>& img) noexcept {
        using T = typename Derived::Scalar;
        using IdxArray = Eigen::Array<Eigen::Index, 1, -1>;
        IdxArray const square_idx = IdxArray::LinSpaced(img.rows(), 0, img.rows() - 1).square();

        // Temporary storage for the computation
        std::vector<Eigen::Index> v(img.rows());
        std::vector<T> z(img.rows() + 1);

        for (Eigen::Index i = 0; i < img.cols(); ++i) {
            Eigen::Index k{0};
            v[0] = 0;
            z[0] = -std::numeric_limits<T>::infinity();
            z[1] = std::numeric_limits<T>::infinity();

            // Image has colmajor storage order
            for (Eigen::Index q{1}; q < img.rows(); ++q) {
                while (true) {
                    Eigen::Index const v_k = v[k];
                    T const s = (img(q, i) + square_idx(q) - img(v_k, i) - square_idx(v_k)) / (2 * q - 2 * v_k);
                    if (s > z[k]) {
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
                while (z[k + 1] < (T)q) ++k;
                Eigen::Index const v_k = v[k];
                Eigen::Index const q_ = q - v_k;
                img(q, i) = img(v_k, i) + square_idx(std::abs(q_));
            }
        }
    }

    /**
     * @brief Performs a 1D pass of the algorithm Distance Transforms of Sampled Functions
     * @tparam Derived The type of the image
     * @param img The resulting distance transform as an image
     */
    template<typename Derived>
    static inline void _distanceTransformColumnPassL1(Eigen::DenseBase<Derived>& img) noexcept {
        for (Eigen::Index q{1}; q < img.cols(); ++q) {
            img.col(q) = img.col(q).min(img.col(q-1)+1);
        }
        // Backward pass
        for (Eigen::Index q{img.cols()-2}; q >=0 ; --q) {
            img.col(q) = img.col(q).min(img.col(q+1)+1);
        }
    }

    enum class Distance {L2=0, L2_SQUARED, L1 };
    /**
     * @brief Perform the distance transform for a given set of lines.
     *
     * This function computes the distance transform of an image based on the provided line array
     * and image size. It supports different distance metrics (L1, L2, and L2 squared) depending on
     * the template parameter `D`. The algorithm is based on the "Distance Transforms of Sampled Functions".
     *
     * - **L1 Distance**: Computes the Manhattan distance.
     * - **L2 Distance**: Computes the Euclidean distance.
     * - **L2 Squared Distance**: Computes the square of the Euclidean distance.
     *
     * Depending on the chosen distance type, different algorithms are applied in both the column
     * and row passes.
     *
     * @tparam T The type of the image (e.g., `float`, `double`), which must support infinity.
     * @tparam D The type of distance metric to use (default is `Distance::L2`).
     * @param linearray The given set of lines for which the distance transform is to be computed.
     * @param size The size of the resulting image.
     * @return The resulting image with the distance transform applied, where each pixel holds the distance value.
     */
    template<typename T, Distance D=Distance::L2>
    inline RawImage<T> distanceTransform(LineArray const& linearray, Size const& size) noexcept {
        static_assert(std::numeric_limits<T>::has_infinity);

        // Initialize the image with max values
        RawImage<T> colmaj_img = RawImage<T>::Constant(size.y(), size.x(), std::numeric_limits<T>::max());
        if (linearray.cols() == 0) return colmaj_img;
        drawLines(colmaj_img, linearray, 0);

        // Perform column pass
        if constexpr (D == Distance::L1) {
            _distanceTransformColumnPassL1(colmaj_img);
            // Perform row pass (using the transpose of the column pass)
            colmaj_img = colmaj_img.transpose().eval(); // transpose require eval to be actually computed
            _distanceTransformColumnPassL1(colmaj_img);
            return colmaj_img.transpose();
        }

        _distanceTransformColumnPassL2(colmaj_img);
        // Perform row pass (using the transpose of the column pass)
        colmaj_img = colmaj_img.transpose().eval(); // transpose require eval to be actually computed
        _distanceTransformColumnPassL2(colmaj_img);

        if constexpr (D == Distance::L2)
            return colmaj_img.transpose().sqrt();
        return colmaj_img.transpose();
    }
} //namespace openfdcm
#endif //OPENFDCM_IMGPROC_H
