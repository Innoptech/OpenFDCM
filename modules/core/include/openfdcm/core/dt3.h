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
#include <thread>

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
        // Collect angles
        std::vector<float> angles;
        angles.reserve(featuremap.size());
        for (const auto& item : featuremap) {
            angles.push_back(item.first);
        }

        // Precompute constants
        const int m = static_cast<int>(featuremap.size());
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

                featuremap[angle2] = featuremap[angle2].min(featuremap[angle1] + coeff * min_h);
            }
        };

        propagate(0, one_and_a_half_cycle_forward, 1);
        propagate(m, one_and_a_half_cycle_backward, -1);
    }

    inline void parallelForEach(const std::vector<float>& angles, std::function<void(float)> func, size_t num_threads) {
        std::vector<std::thread> threads;
        size_t angles_per_thread = angles.size() / num_threads;
        std::mutex mtx;

        auto thread_func = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                func(angles[i]);
            }
        };

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * angles_per_thread;
            size_t end = (t == num_threads - 1) ? angles.size() : start + angles_per_thread;
            threads.emplace_back(thread_func, start, end);
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

    inline Eigen::VectorXi vectorToEigenVector(const std::vector<long>& vec) {
        Eigen::VectorXi eigenVec(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            eigenVec[i] = vec[i];
        }
        return eigenVec;
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
    inline FeatureMap<float> buildFeaturemap(size_t depth, Size const& size, float coeff, LineArray const& linearray, float distanceScale = 1.f) noexcept(false)
    {
        // Step 1: Define a number of linearly spaced angles
        std::set<float> angles{};
        for (size_t i{0}; i < depth; i++)
            angles.insert(float(i) * M_PIf / float(depth) - M_PI_2f);

        // Step 2: Classify lines
        auto classified_lines = classifyLines(angles, linearray);

        // Step 3: Build featuremap with distance transform
        FeatureMap<float> featuremap{};
        std::vector<float> angle_vec(angles.begin(), angles.end());
        size_t num_threads = std::thread::hardware_concurrency();

        std::mutex mtx;
        auto func = [&](float angle) {
            Eigen::VectorXi indices = vectorToEigenVector(classified_lines[angle]);
            RawImage<float> distance_transformed = distanceTransform<float>(linearray(Eigen::all, indices), size) * distanceScale;
            std::lock_guard<std::mutex> lock(mtx);
            featuremap[angle] = std::move(distance_transformed);
        };

        parallelForEach(angle_vec, func, num_threads);

        // Step 4: Propagate orientation
        propagateOrientation(featuremap, coeff);

        // Step 5: Line integral
        for (auto& [lineAngle, feature] : featuremap)
            lineIntegral(feature, lineAngle);

        return featuremap;
    }

} //namespace openfdcm
#endif  // OPENFDCM_DT3_H