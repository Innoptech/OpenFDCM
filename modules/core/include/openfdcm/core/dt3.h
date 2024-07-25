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

#ifndef OPENFDCM_DT3_H
#define OPENFDCM_DT3_H
#include <map>
#include "openfdcm/core/imgproc.h"

namespace openfdcm::core
{
    template<typename T> using FeatureMap = std::map<const float, RawImage<T>>;

    /**
     * @brief Get the size of the features
     * @tparam T The feature type
     * @param featuremap The given featuremap
     * @return The size of the features in pixels
     */
    template<typename T>
    inline Size getFeatureSize(FeatureMap<T> const& featuremap) noexcept
    {
        const RawImage<T>& feature = featuremap.begin()->second;
        return Size{feature.cols(), feature.rows()};
    }

    /**
     * @brief Get the depth of the featuremap
     * @tparam T The feature type
     * @param featuremap The given featuremap
     * @return The size of the featuremap in pixels
     */
    template<typename T>
    inline size_t getFeatureDepth(FeatureMap<T> const& featuremap)
    {
        return featuremap.size();
    }

    /**
     * @brief Get the best match in orientation of the reference line given the featuremap
     * @tparam T The tan values type
     * @tparam U The mapped value
     * @param featuremap The given featuremap
     * @param line The reference line
     * @return A tuple containing the lineAngle and the corresponding feature
     */

    template <class Map>
    inline auto closestOrientation(Map& map, Line const& line) noexcept
    {
        auto line_angle = getAngle(line)(0,0);
        auto itlow = map.upper_bound(line_angle);
        if (itlow != end(map) and itlow != begin(map)) {
            const auto upper_bound_diff = std::abs(line_angle - itlow->first);
            itlow--;
            const auto lower_bound_diff = std::abs(line_angle - itlow->first);
            if (lower_bound_diff < upper_bound_diff)
                return itlow;
            itlow++;
            return itlow;
        }
        itlow = end(map);
        itlow--;
        const float angle1 = line_angle - begin(map)->first;
        const float angle2 = line_angle - itlow->first;
        if (std::min(angle1, std::abs(angle1 - M_PIf)) < std::min(angle2, std::abs(angle2 - M_PIf)))
            return begin(map);
        return itlow;
    }

    /**
     * @brief Classify each line given a restricted set of angles
     * @tparam T The feature type
     * @param angle_set The set of angles
     * @param linearray The array of lines
     * @return A map associating each angle with a vector of indices
     */
    template<typename T> inline std::map<T, std::vector<Eigen::Index>>
    classifyLines(std::set<T> const& angle_set, LineArray const& linearray) noexcept(false)
    {
        std::map<T, std::vector<Eigen::Index>> classified{};
        for(T const& angle : angle_set) classified[angle] = {};

        for (Eigen::Index i{0}; i<linearray.cols(); ++i) {
            auto const& it = closestOrientation(classified, getLine(linearray,i));
            it->second.emplace_back(i);
        }
        return classified;
    }

    /**
     * @brief Propagate the distance transform in the orientation (feature) space
     * @tparam T The feature type
     * @param featuremap The feature map (32bits)
     * @param coeff The propagation coefficient
     */
    template<typename T>
    inline void propagateOrientation(FeatureMap<T> &featuremap, float coeff) noexcept
    {
        //Propagate through orientations
        std::vector<float> angles;
        angles.reserve(featuremap.size());
        for (auto& [lineAngle, feature]: featuremap)
            angles.push_back(lineAngle);

        auto forward = [&featuremap, &angles, coeff]() -> void {
            int c = 0;
            const int m = int(featuremap.size());
            const int one_and_a_half_cycle = c + int(std::ceil(1.5 * m));
            while (c <= one_and_a_half_cycle){
                const int c1 = (m + ((c - 1) % m)) % m;
                const int c2 = (m + (c % m)) % m;
                const float h = std::abs(angles.at(c1) - angles.at(c2));
                featuremap[angles.at(c2)] = featuremap[angles.at(c2)].min(
                        featuremap[angles.at(c1)] + coeff*std::min(h,std::abs(h-M_PIf)));
                c ++;
            }
        };

        auto backward = [&featuremap, &angles, coeff]() -> void {
            int c = int(featuremap.size());
            const int m = int(featuremap.size());
            const int one_and_a_half_cycle = c - int(std::floor(1.5 * m));
            while (c >= one_and_a_half_cycle){
                const int c1 = (m + ((c + 1) % m)) % m;
                const int c2 = (m + (c % m)) % m;
                const float h = std::abs(angles.at(c1) - angles.at(c2));
                featuremap[angles.at(c2)] = featuremap[angles.at(c2)].min(
                        featuremap[angles.at(c1)] + coeff*std::min(h,std::abs(h-M_PIf)));
                c --;
            }
        };
        forward();
        backward();
    }

    /**
     * @brief Build the featuremap to perform the Fast Directional Chamfer Matching algorithm
     * @param depth The number of features (discrete orientation \in [-Pi/2, PI/2])
     * @param size The size of the features in pixels
     * @param coeff The orientation propagation coefficient
     * @param linearray The array of lines
     * @param distanceScale A scale up factor applied only on distance and not on orientation
     * @return The FDCM featuremap
     */
    inline FeatureMap<float>
    buildFeaturemap(size_t depth, Size const& size, float coeff, LineArray const& linearray,
                    float distanceScale=1.f) noexcept(false)
    {
        // Define a number of linearly spaced angles
        std::set<float> angles{};
        for (size_t i{0}; i<depth; i++)
            angles.insert(float(i)*M_PIf/float(depth) - M_PI_2f);

        FeatureMap<float> featuremap{};
        for (auto const& [angle, indices] : classifyLines(angles, linearray))
            featuremap[angle] = distanceTransform<float>(linearray(Eigen::all, indices), size)*distanceScale;
        propagateOrientation(featuremap, coeff);
        for (auto& [lineAngle, feature]: featuremap)
            lineIntegral(feature, lineAngle);
        return featuremap;
    }

} //namespace openfdcm
#endif  // OPENFDCM_DT3_H