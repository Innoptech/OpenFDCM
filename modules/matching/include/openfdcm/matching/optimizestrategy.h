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

#ifndef OPENFDCM_OPTIMIZESTRATEGY_H
#define OPENFDCM_OPTIMIZESTRATEGY_H
#include <memory>
#include <optional>
#include "openfdcm/matching/featuremap.h"

namespace openfdcm::matching
{

    struct OptimalTranslation
    {
        float score;
        core::Point2 translation;
    };

    // ************************************************************************************************************
    // Concepts
    // ************************************************************************************************************
    class OptimizeStrategyInstance {
    };

    template<typename T>
    concept IsOptimizeStrategyInstance = std::is_base_of_v<OptimizeStrategyInstance, T>;

    // ************************************************************************************************************
    // Functions for optimizer implementations
    // ************************************************************************************************************

    /**
     * @brief Perform the 1D optimization by translating the template along the given alignment vector
     * @tparam T The optimizer type
     * @param optimizer The optimizer used to find the optimal translation
     * @param tmpl The evaluated template
     * @param alignments The alignment vector for each template
     * @param featuremap The featuremap used to evaluate the template for a given translation
     * @return A tuple containing the score and the final translation
     */
    template<IsOptimizeStrategyInstance T>
    std::vector<std::optional<OptimalTranslation>> optimize(T const& optimizer, const std::vector<core::LineArray>& templates,
                                                std::vector<core::Point2> const& alignments, FeatureMap const& featuremap);

    namespace detail
    {
        struct OptimizeStrategyConcept
        {
            virtual ~OptimizeStrategyConcept() noexcept = default;
            [[nodiscard]] virtual std::unique_ptr<OptimizeStrategyConcept> clone() const = 0;
            [[nodiscard]] virtual std::vector<std::optional<OptimalTranslation>>
            optimize(const std::vector<core::LineArray>& templates
                    ,  std::vector<core::Point2> const& alignments, FeatureMap const& featuremap) const = 0;
        };

        template<IsOptimizeStrategyInstance T>
        struct OptimizeStrategyModel : public OptimizeStrategyConcept
        {
            explicit OptimizeStrategyModel( T value ) noexcept : object{ std::move(value) }
            {}

            [[nodiscard]] std::unique_ptr<OptimizeStrategyConcept> clone() const final
            {
                return std::make_unique<OptimizeStrategyModel<T>>(*this);
            }

            [[nodiscard]] std::vector<std::optional<OptimalTranslation>> optimize(const std::vector<core::LineArray>& templates
                    ,  std::vector<core::Point2> const& alignments, FeatureMap const& featuremap) const final
            {
                return openfdcm::matching::optimize(object, templates, alignments, featuremap);
            }

            T object;
        };
    }

    /// Type erased OptimizeStrategy
    class OptimizeStrategy : public OptimizeStrategyInstance
    {
        std::unique_ptr<detail::OptimizeStrategyConcept> pimpl;

    public:
        template<IsOptimizeStrategyInstance T>
        /* implicit */ OptimizeStrategy(T const& x) : pimpl{std::make_unique<detail::OptimizeStrategyModel<T>>(x)}
        {}

        OptimizeStrategy(OptimizeStrategy const& other) : pimpl{other.pimpl->clone()} {}
        OptimizeStrategy& operator=(OptimizeStrategy const& other) { pimpl = other.pimpl->clone(); return *this; }
        OptimizeStrategy(OptimizeStrategy&& other) noexcept = default;
        OptimizeStrategy& operator=(OptimizeStrategy&& other) noexcept = default;

        [[nodiscard]] auto optimize(const std::vector<core::LineArray>& templates
                ,  std::vector<core::Point2> const& alignments, FeatureMap const& featuremap) const
        {
            return this->pimpl->optimize(templates, alignments, featuremap);
        }
    };


    // ************************************************************************************************************
    // Free Functions for OptimizeStrategy
    // ************************************************************************************************************
    /**
     * @brief Perform the 1D optimization by translating the template along the given alignment vector
     * @param optimizer The optimizer used to find the optimal translation
     * @param tmpl The evaluated template
     * @param alignments The alignment vector for each template
     * @param featuremap The featuremap used to evaluate the template for a given translation
     * @return A tuple containing the score and the final translation for each template
     */
    inline std::vector<std::optional<OptimalTranslation>>
    optimize(const OptimizeStrategy &optimizer, const std::vector<core::LineArray>& templates,
             const  std::vector<core::Point2> &alignments, const FeatureMap &featuremap)
    {
        return optimizer.optimize(templates, alignments, featuremap);
    }
} // namespace openfdcm

#endif //OPENFDCM_OPTIMIZESTRATEGY_H
