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
    inline void lineIntegral(Eigen::ArrayBase<Derived>& img, float  lineAngle)
    {
        Point2 rastvec = rasterizeVector(Point2{std::cos(lineAngle), std::sin(lineAngle)});

        bool transposed{false};
        if (std::abs(rastvec.y()) == 1)
        {
            // We avoid noncoalesced memory reads
            transposed = true;
            img = img.transpose().eval();
            rastvec = Point2{-rastvec.y(), -rastvec.x()}; // This way, rastvec.x > rastvec.y always and |rastvec.x| == 1
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
    static inline void _distanceTransformColumnPassL1(Eigen::ArrayBase<Derived>& img) noexcept {
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
     * An input array that do no contains any site points (sites have value==0) is considered invalid.
     *
     * @tparam T The type of the image (e.g., `float`, `double`), which must support infinity.
     * @tparam D The type of distance metric to use (default is `Distance::L2`).
     * @param linearray The given set of lines for which the distance transform is to be computed.
     * @param size The size of the resulting image.
     * @return The resulting image with the distance transform applied, where each pixel holds the distance value.
     */
    template<Distance D=Distance::L2, IsEigen Derived>
    inline void distanceTransform(ArrayBase<Derived> &colmaj_img) {
        static_assert(std::numeric_limits<typename Derived::Scalar>::has_infinity);
        assert(colmaj_img.cols() > 0 && colmaj_img.rows() > 0);

        // Perform column pass
        if constexpr (D == Distance::L1) {
            _distanceTransformColumnPassL1(colmaj_img);
            // Perform row pass (using the transpose of the column pass to avoid noncoalesced memory reads)
            colmaj_img = colmaj_img.transpose().eval();
            _distanceTransformColumnPassL1(colmaj_img);
            colmaj_img = colmaj_img.transpose().eval();
            return;
        }

        _distanceTransformColumnPassL2(colmaj_img);
        // Perform row pass (using the transpose of the column pass)
        colmaj_img = colmaj_img.transpose().eval(); // transpose require eval to be actually computed
        _distanceTransformColumnPassL2(colmaj_img);

        if constexpr (D == Distance::L2) {
            colmaj_img = colmaj_img.transpose().sqrt().eval();
            return;
        }
        colmaj_img = colmaj_img.transpose().eval();
    }

    /**
    * @brief Propagate the distance transform in the orientation (feature) space
    * @param featuremap The feature map as a vector
    * @param angles The angle for each feature map as a vector
    * @param coeff The propagation coefficient
    */
    template<IsEigen Derived>
    inline void propagateOrientation(std::vector<Derived>& featuremap,
                                     std::vector<float> const& angles, float coeff) noexcept
    {
        assert(featuremap.size() == angles.size());

        // Precompute constants
        const int m = static_cast<int>(angles.size());
        const int one_and_a_half_cycle_forward = static_cast<int>(std::ceil(1.5 * m));
        const int one_and_a_half_cycle_backward = -static_cast<int>(std::floor(1.5 * m));

        auto propagate = [&](int start, int end, int step) {
            for (int c = start; c != end; c += step) {
                int c1 = (m + ((c - step) % m)) % m;
                int c2 = (m + (c % m)) % m;

                const float angle1 = angles[c1];
                const float angle2 = angles[c2];

                const float h = std::abs(angle1 - angle2);
                const float min_h = std::min(h, std::abs(h - M_PIf));

                featuremap[c2] = featuremap[c2].min(featuremap[c1] + coeff * min_h);
            }
        };

        propagate(0, one_and_a_half_cycle_forward, 1);
        propagate(m, one_and_a_half_cycle_backward, -1);
    }
} //namespace openfdcm
#endif //OPENFDCM_IMGPROC_H
