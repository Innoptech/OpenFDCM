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

#ifndef OPENFDCM_SEARCHSTRATEGY_H
#define OPENFDCM_SEARCHSTRATEGY_H
#include <memory>
#include "openfdcm/core/math.h"

namespace openfdcm::matching
{
    class SearchCombination {
    public:
        SearchCombination(size_t const tmpl_idx, size_t const scene_idx) : tmpl_idx_{tmpl_idx}, scene_idx_{scene_idx}
        {}
        [[nodiscard]] size_t getTmplLineIdx() const noexcept {return tmpl_idx_;}
        [[nodiscard]] size_t getSceneLineIdx() const noexcept {return scene_idx_;}
    private:
        size_t tmpl_idx_, scene_idx_;
    };

    inline bool operator==(SearchCombination const& rhs, SearchCombination const& lhs)
    {
        return rhs.getSceneLineIdx() == lhs.getSceneLineIdx() and rhs.getTmplLineIdx() == lhs.getTmplLineIdx();
    }

    // ************************************************************************************************************
    // Concepts
    // ************************************************************************************************************
    class SearchStrategyInstance {
    };

    template<typename T>
    concept IsSearchStrategyInstance = std::is_base_of_v<SearchStrategyInstance, T>;

    // ************************************************************************************************************
    // Functions for penalty implementations
    // ************************************************************************************************************

    /**
     * @brief Establish the searching strategy
     * @tparam T The strategy type
     * @param strategy The strategy
     * @param tmpl_lines The template lines
     * @param scene_lines The scene lines
     * @return A vector of template line idx ans scene line idx combinations
     */
    template<IsSearchStrategyInstance T>
    std::vector<SearchCombination> establishSearchStrategy(
            T const& strategy, core::LineArray const& tmpl_lines, core::LineArray const& scene_lines);


    namespace detail {
        struct SearchStrategyConcept
        {
            virtual ~SearchStrategyConcept() noexcept = default;
            [[nodiscard]] virtual std::unique_ptr<SearchStrategyConcept> clone() const = 0;
            [[nodiscard]] virtual std::vector<SearchCombination> establishSearchStrategy(
                    core::LineArray const& tmpl_lines, core::LineArray const& scene_lines) const = 0;
        };

        template<IsSearchStrategyInstance T>
        struct SearchStrategyModel : public SearchStrategyConcept
        {
            explicit SearchStrategyModel( T value ) noexcept : object{ std::move(value) }
            {}

            [[nodiscard]] std::unique_ptr<SearchStrategyConcept> clone() const final
            {
                return std::make_unique<SearchStrategyModel<T>>(*this);
            }

            [[nodiscard]] std::vector<SearchCombination> establishSearchStrategy(
                    core::LineArray const& tmpl_lines, core::LineArray const& scene_lines) const final
            {
                return openfdcm::matching::establishSearchStrategy(object, tmpl_lines, scene_lines);
            }

            T object;
        };
    }


    /// Type erased SearchStrategy
    class SearchStrategy : public SearchStrategyInstance
    {
        std::unique_ptr<detail::SearchStrategyConcept> pimpl;

    public:
        template<IsSearchStrategyInstance T>
        /* implicit */ SearchStrategy(T const& x) : pimpl{std::make_unique<detail::SearchStrategyModel<T>>(x)}
        {}

        SearchStrategy(SearchStrategy const& other) : pimpl{other.pimpl->clone()} {}
        SearchStrategy& operator=(SearchStrategy const& other) { pimpl = other.pimpl->clone(); return *this; }
        SearchStrategy(SearchStrategy&& other) noexcept = default;
        SearchStrategy& operator=(SearchStrategy&& other) noexcept = default;

        [[nodiscard]] std::vector<SearchCombination> establishSearchStrategy(
                core::LineArray const& tmpl_lines, core::LineArray const& scene_lines) const
        {
            return this->pimpl->establishSearchStrategy(tmpl_lines, scene_lines);
        }
    };


    // ************************************************************************************************************
    // Free Functions for SearchStrategy
    // ************************************************************************************************************
    /**
     * @brief Establish the searching strategy
     * @tparam T The strategy type
     * @param strategy The strategy
     * @param tmpl_lines The template lines
     * @param scene_lines The scene lines
     * @return A vector of template line idx ans scene line idx combinations
     */
    inline std::vector<SearchCombination> establishSearchStrategy(
            SearchStrategy const& strategy, core::LineArray const& tmpl_lines, core::LineArray const& scene_lines)
    {
        return strategy.establishSearchStrategy(tmpl_lines, scene_lines);
    }

} // namespace openfdcm::matching
#endif //OPENFDCM_SEARCHSTRATEGY_H
