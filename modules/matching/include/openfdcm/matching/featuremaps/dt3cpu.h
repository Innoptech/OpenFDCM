#ifndef OPENFDCM_FEATUREMAPS_DT3CPU_H
#define OPENFDCM_FEATUREMAPS_DT3CPU_H
#include <map>
#include "openfdcm/matching/featuremap.h"
#include "BS_thread_pool.hpp"

namespace openfdcm::matching {

    struct Dt3CpuParameters {
        size_t depth;
        float dt3Coeff, scale{1.f}, padding{1.f};
    };

    class Dt3Cpu : public FeatureMapInstance
    {
        Dt3CpuParameters params_;
        core::Point2 sceneTranslation_;
        core::Size sceneSize_;
        core::LineArray scene_;
        core::LineArray transformedScene_;
        std::shared_ptr<BS::thread_pool> pool_;

        template<typename T> using Dt3Map = std::map<const float, core::RawImage<T>>;
        Dt3Map<float> dt3map_;
    public:
        Dt3Cpu(core::LineArray scene, Dt3CpuParameters params, const BS::concurrency_t num_threads=1);

        [[nodiscard]] auto getParams() const {return params_;}
        [[nodiscard]] auto getSceneTranslation() const {return sceneTranslation_;}
        [[nodiscard]] auto getSceneSize() const {return sceneSize_;}
        [[nodiscard]] auto getScene() const {return scene_;}
        [[nodiscard]] auto getTransformedScene() const {return transformedScene_;}
        [[nodiscard]] auto getDt3Map() const {return dt3map_;}
        [[nodiscard]] auto getThreadPool() const {return std::weak_ptr(pool_);};
    };

    namespace detail
    {
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

        struct SceneShift
        {
            core::Point2 translation;
            core::Size sceneSize;
        };

        /**
         * @brief Find the transformation to center a scene in an positive set of boundaries
         * given the image area ratio (scene size vs image size)
         * @param scene The scene lines
         * @param scene_padding The ratio between the original scene area and the image scene area
         * @return The resulting SceneShift object
         */
        SceneShift getSceneCenteredTranslation(core::LineArray const& scene, float scene_padding) noexcept;

        template<typename T> using Dt3Map = std::map<const float, core::RawImage<T>>;
        void propagateOrientation(Dt3Map<float> &featuremap, float coeff) noexcept;

        /**
         * @brief Build the featuremap to perform the Fast Directional Chamfer Matching algorithm
         * @param depth The number of features (discrete orientation \in [-Pi/2, PI/2])
         * @param size The size of the features in pixels
         * @param coeff The orientation propagation coefficient
         * @param sceneScale A scale up factor applied only on distance and not on orientation
         * @param linearray The array of lines
         * @return The FDCM featuremap
         */
        detail::Dt3Map<float> buildFeaturemap(size_t depth, core::Size const &size,
                                              float coeff, core::LineArray const &linearray,
                                              float sceneScale=1.f,
                                              const std::weak_ptr<BS::thread_pool> &pool = std::weak_ptr<BS::thread_pool>()) noexcept(false);
    }

    template<>
    inline core::LineArray getSceneLines(const Dt3Cpu& featuremap) noexcept {
        return featuremap.getScene();
    }

    template<>
    inline core::Size getFeatureSize(const Dt3Cpu& featuremap) noexcept {
        return featuremap.getSceneSize();
    }
} //namespace openfdcm::featuremaps
#endif //OPENFDCM_FEATUREMAPS_DT3CPU_H
