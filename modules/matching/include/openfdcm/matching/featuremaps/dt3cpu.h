#ifndef OPENFDCM_FEATUREMAPS_DT3CPU_H
#define OPENFDCM_FEATUREMAPS_DT3CPU_H
#include <map>
#include "openfdcm/matching/featuremap.h"
#include "BS_thread_pool.hpp"

namespace openfdcm::matching {

    struct Dt3CpuParameters {
        size_t depth{};   // The number of features (discrete orientation \in [-Pi/2, PI/2])
        float dt3Coeff{}, // The orientation propagation coefficient
        padding;   // The padding ratio (paddedSceneSize = padding * sceneSize

        Dt3CpuParameters(size_t _depth=30, float _dt3Coeff=5.f, float _padding=2.2f) :
        depth{_depth}, dt3Coeff{_dt3Coeff}, padding{_padding}
        {};
    };

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

    /**
     * @brief Build the featuremap to perform the Fast Directional Chamfer Matching algorithm
     * @param scene The scene lines
     * @param params The Featuremap parameters
     * @param pool A threadpool to parallelize feature map generation
     * @return The FDCM featuremap
     */
    Dt3Cpu buildCpuFeaturemap(const core::LineArray &scene, const Dt3CpuParameters &params,
                              const std::shared_ptr<BS::thread_pool> &pool=std::make_shared<BS::thread_pool>()) noexcept(false);

    namespace detail
    {
        template<typename T> using Dt3CpuMap = std::map<const float, core::RawImage<T>>;

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

        /**
         * @brief Propagate the distance transform in the orientation (feature) space
         * @param featuremap The feature map (32bits)
         * @param coeff The propagation coefficient
         */
        void propagateOrientation(Dt3CpuMap<float> &featuremap, float coeff) noexcept;
    }

    template<>
    inline core::Size getFeatureSize(const Dt3Cpu& featuremap) noexcept {
        return featuremap.getFeatureSize();
    }
} //namespace openfdcm::featuremaps
#endif //OPENFDCM_FEATUREMAPS_DT3CPU_H
