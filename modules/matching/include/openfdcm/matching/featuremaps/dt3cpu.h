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
#include "openfdcm/matching/featuremap.h"
#include "openfdcm/core/imgproc.h"
#include <BS_thread_pool.hpp>

namespace openfdcm::matching {

    struct Dt3CpuParameters {
        size_t depth{};   // The number of features (discrete orientation \in [-Pi/2, PI/2])
        float dt3Coeff{}, // The orientation propagation coefficient
        padding;   // The padding ratio (paddedSceneSize = padding * sceneSize

        explicit Dt3CpuParameters(size_t _depth=30, float _dt3Coeff=5.f, float _padding=2.2f) :
        depth{_depth}, dt3Coeff{_dt3Coeff}, padding{_padding}
        {};
    };


    template<typename T>
    struct Dt3CpuMap
    {
        std::vector<core::RawImage<T>> features;
        std::vector<float> sortedAngles;
    };

    class Dt3Cpu : public FeatureMapInstance
    {
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
         * @brief Get the index of the best match in orientation of the reference line given the featuremap
         * @param sortedAngles The given featuremap (must be sorted)
         * @param line The reference line
         * @return The index of the closest angle
         */
        size_t closestOrientationIndex(std::vector<float> const& sortedAngles, core::Line const& line);

        /**
        * @brief Classify each line given a restricted set of angles
        * @tparam T The feature type
        * @param sortedAngles The set of sorted angles
        * @param linearray The array of lines
        * @return A map associating each angle with a vector of indices
        */
        inline std::vector<std::vector<Eigen::Index>> classifyLines(std::vector<float> const& sortedAngles,
                                                                    core::LineArray const& linearray) noexcept(false)
        {
            if(linearray.cols() == 0)
                return {};
            if (sortedAngles.empty()) {
                throw std::runtime_error{"Error in closestOrientationIndex: set of sortedAngles is empty"};
            }

            if (!std::is_sorted(std::begin(sortedAngles), std::end(sortedAngles))) {
                throw std::runtime_error{"Error in closestOrientationIndex: set of sortedAngles is not sorted"};
            }

            // A vector of template line idx for each angle
            std::vector<std::vector<Eigen::Index>> tmplLineIdx(sortedAngles.size());
            for (Eigen::Index i{0}; i<linearray.cols(); ++i) {
                auto const& angleIdx = closestOrientationIndex(sortedAngles, core::getLine(linearray,i));
                tmplLineIdx[angleIdx].emplace_back(i);
            }
            return tmplLineIdx;
        }

        /**
         * @brief Find the transformation to center a scene in an positive set of boundaries
         * given the image area ratio (scene size vs image size)
         * @param scene The scene lines
         * @param scene_padding The ratio between the original scene area and the image scene area
         * @return The resulting SceneShift object
         */
        SceneShift getSceneCenteredTranslation(core::LineArray const& scene, float scene_padding) noexcept;
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
        std::vector<float> sortedAngles{}; sortedAngles.reserve(params.depth);
        for (size_t i{0}; i < params.depth; i++) {
            sortedAngles.emplace_back(float(i) * M_PIf / float(params.depth) - M_PI_2f);
        }
            
        assert(std::is_sorted(std::begin(sortedAngles), std::end(sortedAngles)));

        // Step 2: Classify lines
        auto const& indices = detail::classifyLines(sortedAngles, translatedScene);

        // Step 3: Build featuremap with distance transform
        Dt3CpuMap<float> dt3map{std::vector<core::RawImage<float>>(sortedAngles.size()), sortedAngles};
        auto const& size = sceneShift.sceneSize;

        auto func = [&](size_t angleIdx) {
            Eigen::Matrix<float, 4, Eigen::Dynamic> sceneLinesSelection =
                    translatedScene(Eigen::all, indices.at(angleIdx)).eval();
            auto colmaj_img = core::RawImage<float>::Constant(
                    size.y(), size.x(), std::numeric_limits<float>::max()).eval();
            if(sceneLinesSelection.cols() > 0)
            {
                // An image without any site points would be invalid for distance transform
                core::drawLines(colmaj_img, sceneLinesSelection, static_cast<float>(0));
                core::distanceTransform<D>(colmaj_img);
            }
            dt3map.features[angleIdx] = std::move(colmaj_img);
        };

        if (pool_ptr) {
            // Submit tasks to the thread pool for each angle
            std::vector<std::future<void>> futures;
            for (size_t angleIdx{0}; angleIdx < sortedAngles.size(); ++angleIdx) {
                futures.push_back(pool_ptr->submit_task([=] { func(angleIdx); }));
            }
            // Wait for all tasks to complete
            for (auto &fut : futures) {
                fut.get();
            }
        } else {
            // If no thread pool is available, run tasks sequentially
            for (size_t angleIdx{0}; angleIdx < sortedAngles.size(); ++angleIdx) {
                func(angleIdx);
            }
        }

        // Step 4: Propagate orientation
        core::propagateOrientation(dt3map.features, dt3map.sortedAngles, params.dt3Coeff);

        // Step 5: Line integral
        for (size_t angleIdx{0}; angleIdx < dt3map.sortedAngles.size(); ++angleIdx)
        {
            core::lineIntegral(dt3map.features[angleIdx], dt3map.sortedAngles[angleIdx]);
        }

        return {dt3map, sceneShift.translation, sceneShift.sceneSize};
    }
} //namespace openfdcm::featuremaps
#endif //OPENFDCM_FEATUREMAPS_DT3CPU_H
