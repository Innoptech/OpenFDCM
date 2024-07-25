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
#include "openfdcm/core/dt3.h"

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
     * @param align_vec The alignment vector
     * @param featuremap The featuremap used to evaluate the template for a given translation
     * @return A tuple containing the score and the final translation
     */
    template<IsOptimizeStrategyInstance T>
    std::optional<OptimalTranslation> optimize(T const& optimizer, const core::LineArray& tmpl,
                                               core::Point2 const& align_vec, core::FeatureMap<float> const& featuremap);

    namespace detail
    {
        struct OptimizeStrategyConcept
        {
            virtual ~OptimizeStrategyConcept() noexcept = default;
            [[nodiscard]] virtual std::unique_ptr<OptimizeStrategyConcept> clone() const = 0;
            [[nodiscard]] virtual std::optional<OptimalTranslation> optimize(const core::LineArray& tmpl
                    , core::Point2 const& align_vec, core::FeatureMap<float> const& featuremap) const = 0;
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

            [[nodiscard]] std::optional<OptimalTranslation> optimize(const core::LineArray& tmpl
                    , core::Point2 const& align_vec, core::FeatureMap<float> const& featuremap) const final
            {
                return openfdcm::matching::optimize(object, tmpl, align_vec, featuremap);
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

        [[nodiscard]] std::optional<OptimalTranslation> optimize(const core::LineArray& tmpl
                , core::Point2 const& align_vec, core::FeatureMap<float> const& featuremap) const
        {
            return this->pimpl->optimize(tmpl, align_vec, featuremap);
        }
    };


    // ************************************************************************************************************
    // Free Functions for OptimizeStrategy
    // ************************************************************************************************************
    /**
     * @brief Perform the 1D optimization by translating the template along the given alignment vector
     * @param optimizer The optimizer used to find the optimal translation
     * @param tmpl The evaluated template
     * @param align_vec The alignment vector
     * @param featuremap The featuremap used to evaluate the template for a given translation
     * @return A tuple containing the score and the final translation
     */
    inline std::optional<OptimalTranslation> optimize(const OptimizeStrategy &optimizer, const core::LineArray& tmpl,
                                                      const core::Point2 &align_vec, const core::FeatureMap<float> &featuremap)
    {
        return optimizer.optimize(tmpl, align_vec, featuremap);
    }
} // namespace openfdcm

#endif //OPENFDCM_OPTIMIZESTRATEGY_H
