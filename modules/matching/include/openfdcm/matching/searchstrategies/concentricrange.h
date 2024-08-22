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

#ifndef OPENFDCM_CONCENTRICRANGE_H
#define OPENFDCM_CONCENTRICRANGE_H
#include "openfdcm/matching/searchstrategies/defaultsearch.h"

namespace openfdcm::matching
{
    /**
     *  ConcentricRangeStrategy establish a search strategy for which each N longest template lines
     *  are matched against the M most similar scene lines in terms of length where all the scene lines
     *  are filtered given their position from the center of the image.
     */
    class ConcentricRangeStrategy : public SearchStrategyInstance
    {
    public:
        /**
         * @brief ConcentricRangeStrategy constructor
         * @param max_tmpl_lines The max number of longest template lines
         * @param max_scene_lines The max number of longest scene lines
         * @param center_position The position of the center of the image
         * @param low_boundary The lower bound of the radius
         * @param high_boundary The higher bound of the radius
         */
        ConcentricRangeStrategy(size_t const max_tmpl_lines, size_t const max_scene_lines,
                                core::Point2 center_position, float const low_boundary, float const high_boundary)
                : max_tmpl_lines_{max_tmpl_lines}, max_scene_lines_{max_scene_lines},
                  center_position_{std::move(center_position)}, low_boundary_{low_boundary}, high_boundary_{high_boundary}
        {}

        [[nodiscard]] size_t getMaxTmplLines() const noexcept {return max_tmpl_lines_;}
        [[nodiscard]] size_t getMaxSceneLines() const noexcept {return max_scene_lines_;}
        [[nodiscard]] core::Point2 getCenterPosition() const noexcept {return center_position_;}
        [[nodiscard]] float getLowBoundary() const noexcept {return low_boundary_;}
        [[nodiscard]] float getHighBoundary() const noexcept {return high_boundary_;}

    private:
        size_t max_tmpl_lines_, max_scene_lines_;
        core::Point2 center_position_;
        float low_boundary_, high_boundary_;
    };

    /**
     * @brief Filters an array of line given the position of their center w.r.t. the center of the image
     * @param linearray The array of lines to filter
     * @param center_position The position of the center of the image
     * @param min_radius The lower bound of the radius
     * @param max_radius The higher bound of the radius
     * @return A vector of the filter indices
     */
    inline std::vector<long> filterInRange(core::LineArray const& linearray, core::Point2 const& center_position,
                                    float const min_radius, float const max_radius) noexcept
    {
        const Eigen::Matrix<float, 1, -1> radius = (core::getCenter(linearray).array().colwise() - center_position.array()).colwise().norm();
        std::vector<long> idx_in_range{};
        for (long i{0}; i < linearray.cols(); ++i){
            float const rad = radius(i);
            if ( rad > (min_radius-std::numeric_limits<float>::epsilon()) && rad < max_radius)
                idx_in_range.emplace_back(i);
        }
        return idx_in_range;
    }

    /**
     * @brief Slice a vector given a vector of indices
     * @tparam T The type of the vector to slice
     * @tparam U The type of the indices
     * @return The sliced vector
     */
    template<typename T, typename U>
    inline std::vector<T> sliceVector(std::vector<T> const& vec, std::vector<U> const& slice_indices)
    {
        std::vector<T> sliced_vec{};
        sliced_vec.reserve(slice_indices.size());
        for(const U& idx : slice_indices )
            sliced_vec.emplace_back(vec.at(idx));
        return sliced_vec;
    }

} // namespace openfdcm::matching
#endif //OPENFDCM_CONCENTRICRANGE_H
