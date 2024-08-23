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

#ifndef OPENFDCM_FEATUREMAPS_DT3CPU_H
#define OPENFDCM_FEATUREMAPS_DT3CPU_H
#include <map>
#include "openfdcm/matching/featuremap.h"
#include "openfdcm/core/imgproc.h"
#include "BS_thread_pool.hpp"

namespace openfdcm::matching {

    struct Dt3CpuParameters {
        size_t depth{};   // The number of features (discrete orientation \in [-Pi/2, PI/2])
        float dt3Coeff{}, // The orientation propagation coefficient
        padding;   // The padding ratio (paddedSceneSize = padding * sceneSize

        explicit Dt3CpuParameters(size_t _depth=30, float _dt3Coeff=5.f, float _padding=2.2f) :
        depth{_depth}, dt3Coeff{_dt3Coeff}, padding{_padding}
        {};
    };

    template<typename T> using Dt3CpuMap = std::map<const float, core::RawImage<T>>;

    class Dt3Cpu : public FeatureMapInstance
    {
        template<typename T> using Dt3CpuMap = std::map<const float, core::RawImage<T>>;

        Dt3CpuMap<float> dt3map_;
        core::Point2 sceneTranslation_;
        core::Size featureSize_;

    public:
        Dt3Cpu(Dt3CpuMap<float> dt3map, core::Point2 sceneTranslation, core::Size featureSize)
               : dt3map_{std::move(dt3map)}, sceneTranslation_{std::move(sceneTranslation)},
                 featureSize_{std::move(featureSize)}
        {}

        [[nodiscard]] auto getSceneTranslation() const {return sceneTranslation_;}
        [[nodiscard]] auto getFeatureSize() const {return featureSize_;}
        [[nodiscard]] auto& getDt3Map() const {return dt3map_;}
    };

    namespace detail
    {
        struct SceneShift
        {
            core::Point2 translation;
            core::Size sceneSize;
        };

        /**
         * @brief Compute the negative and positive values for the maximum translation of the template in the dt3 window
         * @param tmpl The given template
         * @param align_vec The translation vector
         * @param featuresize The dt3 window size
         * @param extraTranslation An optional extra translation applied on the template
         * @return The negative and positive values for the maximum translation of the template in the image window
         */
        std::array<float, 2>
        minmaxTranslation(const core::LineArray& tmpl, core::Point2 const& align_vec, core::Size const& featuresize,
                          core::Point2 const& extraTranslation=core::Point2{0.f,0.f});

        /**
         * @brief Get the best match in orientation of the reference line given the featuremap
         * @tparam T The tan values type
         * @tparam U The mapped value
         * @param featuremap The given featuremap
         * @param line The reference line
         * @return A tuple containing the lineAngle and the corresponding feature
         */
        template <class Map>
        inline auto closestOrientation(Map& map, core::Line const& line) noexcept
        {
            auto line_angle = core::getAngle(line)(0,0);
            auto itlow = map.upper_bound(line_angle);
            if (itlow != std::end(map) and itlow != std::begin(map)) {
                const auto upper_bound_diff = std::abs(line_angle - itlow->first);
                itlow--;
                const auto lower_bound_diff = std::abs(line_angle - itlow->first);
                if (lower_bound_diff < upper_bound_diff)
                    return itlow;
                itlow++;
                return itlow;
            }
            itlow = std::end(map);
            itlow--;
            const float angle1 = line_angle - std::begin(map)->first;
            const float angle2 = line_angle - itlow->first;
            if (std::min(angle1, std::abs(angle1 - M_PIf)) < std::min(angle2, std::abs(angle2 - M_PIf)))
                return std::begin(map);
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
        classifyLines(std::set<T> const& angle_set, core::LineArray const& linearray) noexcept(false)
        {
            std::map<T, std::vector<Eigen::Index>> classified{};
            for(T const& angle : angle_set) classified[angle] = {};

            for (Eigen::Index i{0}; i<linearray.cols(); ++i) {
                auto const& it = closestOrientation(classified, core::getLine(linearray,i));
                it->second.emplace_back(i);
            }
            return classified;
        }

        /**
         * @brief Find the transformation to center a scene in an positive set of boundaries
         * given the image area ratio (scene size vs image size)
         * @param scene The scene lines
         * @param scene_padding The ratio between the original scene area and the image scene area
         * @return The resulting SceneShift object
         */
        SceneShift getSceneCenteredTranslation(core::LineArray const& scene, float scene_padding) noexcept;

        /**
         * @brief Propagate the distance transform in the orientation (feature) space
         * @param featuremap The feature map (32bits)
         * @param coeff The propagation coefficient
         */
        void propagateOrientation(Dt3CpuMap<float> &featuremap, float coeff) noexcept;

        template<typename Vec>
        inline Eigen::VectorXi vectorToEigenVector(const Vec &vec) noexcept {
            Eigen::VectorXi eigenVec(vec.size());
            for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(vec.size()); ++i) {
                eigenVec[i] = vec[static_cast<size_t>(i)];
            }
            return eigenVec;
        }
    }

    template<>
    inline core::Size getFeatureSize(const Dt3Cpu& featuremap) noexcept {
        return featuremap.getFeatureSize();
    }

    /**
    * @brief Build the featuremap to perform the Fast Directional Chamfer Matching algorithm
    * @param scene The scene lines
    * @param params The Featuremap parameters
    * @param pool A threadpool to parallelize feature map generation
    * @return The FDCM featuremap
    */
    template<core::Distance D=core::Distance::L2>
    Dt3Cpu buildCpuFeaturemap(
            const core::LineArray &scene, const Dt3CpuParameters &params,
            const std::shared_ptr<BS::thread_pool> &pool_ptr=std::make_shared<BS::thread_pool>()
            ) noexcept(false)
    {
        if (scene.cols() == 0)
            return {Dt3CpuMap<float>{}, core::Point2{0,0}, core::Size{0,0}};

        // Shift the scene so that all scene lines are greater than 0.
        detail::SceneShift const& sceneShift = detail::getSceneCenteredTranslation(scene, params.padding);
        const core::LineArray translatedScene = core::translate(scene, sceneShift.translation);

        // Step 1: Define a number of linearly spaced angles
        std::set<float> angles{};
        for (size_t i{0}; i < params.depth; i++)
            angles.insert(float(i) * M_PIf / float(params.depth) - M_PI_2f);

        // Step 2: Classify lines
        auto const& classified_lines = detail::classifyLines(angles, translatedScene);

        // Step 3: Build featuremap with distance transform
        Dt3CpuMap<float> dt3map{};
        std::mutex dt3map_mutex;  // Mutex to protect access to dt3map

        auto func = [&](float angle) {
            Eigen::VectorXi indices = detail::vectorToEigenVector(classified_lines.at(angle));
            core::RawImage<float> distance_transformed =
                    core::distanceTransform<float, D>(translatedScene(Eigen::all, indices), sceneShift.sceneSize);

            // Lock the mutex to safely update dt3map
            std::lock_guard<std::mutex> lock(dt3map_mutex);
            dt3map[angle] = std::move(distance_transformed);
        };

        if (pool_ptr) {
            // Submit tasks to the thread pool for each angle
            std::vector<std::future<void>> futures;
            for (const auto &angle : angles) {
                futures.push_back(pool_ptr->submit_task([=] { func(angle); }));
            }
            // Wait for all tasks to complete
            for (auto &fut : futures) {
                fut.get();
            }
        } else {
            // If no thread pool is available, run tasks sequentially
            for (const auto &angle : angles) {
                func(angle);
            }
        }

        // Step 4: Propagate orientation
        detail::propagateOrientation(dt3map, params.dt3Coeff);

        // Step 5: Line integral
        for (auto &[lineAngle, feature] : dt3map)
            core::lineIntegral(feature, lineAngle);

        return {dt3map, sceneShift.translation, sceneShift.sceneSize};
    }
} //namespace openfdcm::featuremaps
#endif //OPENFDCM_FEATUREMAPS_DT3CPU_H
