#ifndef OPENFDCM_MATCHING_FEATUREMAPS_H
#define OPENFDCM_MATCHING_FEATUREMAPS_H
#include <memory>
#include "openfdcm/core/math.h"

namespace openfdcm::matching {
    // ************************************************************************************************************
    // Concepts
    // ************************************************************************************************************
    class FeatureMapInstance {
    };

    template<typename T>
    concept IsFeatureMapInstance = std::is_base_of_v<FeatureMapInstance, T>;

    // ************************************************************************************************************
    // Functions for featuremap implementations
    // ************************************************************************************************************
    /**
     * @brief Get the scene lines
     * @tparam T The specialized featuremap type
     * @param featuremap The given featuremap
     * @return The scene lines
     */
    template<IsFeatureMapInstance T>
    inline core::LineArray getSceneLines(const T& featuremap) noexcept;

    /**
     * @brief Get the size of the features
     * @tparam T The specialized featuremap type
     * @param featuremap The given featuremap
     * @return The size of the features in pixels
     */
    template<IsFeatureMapInstance T>
    inline core::Size getFeatureSize(const T& featuremap) noexcept;

    /**
     * @brief Evaluate a vector of templates on the featuremap
     * @tparam T The specialized featuremap type
     * @param featuremap The given featuremap
     * @param templates The given templates
     * @param translations The translations for which to evaluate the templates
     * @return The score of the evaluation
     */
    template<IsFeatureMapInstance T>
    inline std::vector<float> evaluate(const T& featuremap, const std::vector<core::LineArray>& templates,
                                       const std::vector<core::Point2>& translations);


    namespace detail
    {
        struct FeatureMapConcept
        {
            virtual ~FeatureMapConcept() noexcept = default;
            [[nodiscard]] virtual std::unique_ptr<FeatureMapConcept> clone() const = 0;
            [[nodiscard]] virtual core::LineArray getSceneLines() const = 0;
            [[nodiscard]] virtual core::Size getFeatureSize() const = 0;
            [[nodiscard]] virtual std::vector<float> evaluate(const std::vector<core::LineArray>& templates,
                                                              const std::vector<core::Point2>& translations) const = 0;
        };

        template<IsFeatureMapInstance T>
        struct FeatureMapModel : public FeatureMapConcept
        {
            explicit FeatureMapModel( T value ) noexcept : object{ std::move(value) }
                    {}

            [[nodiscard]] std::unique_ptr<FeatureMapConcept> clone() const final
            {
                return std::make_unique<FeatureMapModel<T>>(*this);
            }

            [[nodiscard]] core::LineArray getSceneLines() const final {
                return openfdcm::matching::getSceneLines(object);
            }

            [[nodiscard]] core::Size getFeatureSize() const final {
                return openfdcm::matching::getFeatureSize(object);
            }

            [[nodiscard]] std::vector<float> evaluate(const std::vector<core::LineArray>& templates,
                                                      const std::vector<core::Point2>& translations) const final {
                return openfdcm::matching::evaluate(object, templates, translations);
            }

            T object;
        };
    }

    /// Type erased FeatureMap
    class FeatureMap : public FeatureMapInstance
    {
        std::unique_ptr<detail::FeatureMapConcept> pimpl;

    public:
        template<IsFeatureMapInstance T>
        /* implicit */ FeatureMap(T const& x) : pimpl{std::make_unique<detail::FeatureMapModel<T>>(x)}
        {}

        FeatureMap(FeatureMap const& other) : pimpl{other.pimpl->clone()} {}
        FeatureMap& operator=(FeatureMap const& other) { pimpl = other.pimpl->clone(); return *this; }
        FeatureMap(FeatureMap&& other) noexcept = default;
        FeatureMap& operator=(FeatureMap&& other) noexcept = default;

        [[nodiscard]] core::LineArray getSceneLines() const {
            return pimpl->getSceneLines();
        }

        [[nodiscard]] auto getFeatureSize() const {
            return pimpl->getFeatureSize();
        }

        [[nodiscard]] auto evaluate(const std::vector<core::LineArray>& templates,
                                    const std::vector<core::Point2>& translations) const {
            return pimpl->evaluate(templates, translations);
        }
    };


    // ************************************************************************************************************
    // Free Functions for FeatureMap
    // ************************************************************************************************************
    /**
     * @brief Get the scene lines
     * @param featuremap The given featuremap
     * @return The scene lines
     */
    inline core::LineArray getSceneLines(const FeatureMap& featuremap) noexcept {
        return featuremap.getSceneLines();
    }

    /**
     * @brief Get the size of the features
     * @param featuremap The given featuremap
     * @return The size of the features in pixels
     */
    inline core::Size getFeatureSize(const FeatureMap& featuremap) noexcept {
        return featuremap.getFeatureSize();
    }

    /**
     * @brief Evaluate a vector of template on the featuremap
     * @tparam T The specialized featuremap type
     * @param featuremap The given featuremap
     * @param templates The given templates
     * @param translations The translations for which to evaluate the templates
     * @return The score of the evaluation
     */
    inline std::vector<float> evaluate(const FeatureMap& featuremap, const std::vector<core::LineArray>& templates,
                                       const std::vector<core::Point2>& translations) {
        return featuremap.evaluate(templates, translations);
    }

} //namespace openfdcm::matching
#endif //OPENFDCM_MATCHING_FEATUREMAPS_H
