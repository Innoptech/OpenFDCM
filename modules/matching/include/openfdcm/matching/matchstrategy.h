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

#ifndef OPENFDCM_SEARCH_H
#define OPENFDCM_SEARCH_H
#include <memory>
#include <utility>
#include "openfdcm/matching/searchstrategy.h"
#include "openfdcm/matching/optimizestrategy.h"


namespace openfdcm::matching
{
    struct Match
    {
        int tmplIdx;
        float score;
        core::Mat23 transform;

        Match(int _tmplIdx, float _score, core::Mat23 _transform)
        : tmplIdx{_tmplIdx}, score{_score}, transform{std::move(_transform)}
        {}
    };

    inline bool operator<(const Match& lhs, const Match& rhs) noexcept{ return lhs.score < rhs.score; }

    // ************************************************************************************************************
    // Concepts
    // ************************************************************************************************************
    class MatchStrategyInstance {
    };

    template<typename T>
    concept IsMatchStrategyInstance = std::is_base_of_v<MatchStrategyInstance, T>;

    // ************************************************************************************************************
    // Functions for optimizer implementations
    // ************************************************************************************************************

    /**
     * @brief Search for optimal matches between the templates and the scene
     * @tparam T The MatchStrategy type
     * @param matcher The matcher used to find and optimize matches
     * @param templates The templates expressed as a vector of LineArrays
     * @param scene The scene expressed as a core::LineArray
     * @return A vector containing all the sorted matches by score (lowest to highest)
     */
    template<IsMatchStrategyInstance T>
    std::vector<Match> search(const T & matcher, const SearchStrategy &searcher, const OptimizeStrategy &optimizer,
                              const FeatureMap& featuremap, std::vector<core::LineArray> const& templates,
                              const core::LineArray& scene);


    namespace detail {
        struct MatcherConcept
        {
            virtual ~MatcherConcept() noexcept = default;
            [[nodiscard]] virtual std::unique_ptr<MatcherConcept> clone() const = 0;
            [[nodiscard]] virtual std::vector<Match> search(const SearchStrategy &searcher,
                                                            const OptimizeStrategy &optimizer,
                                                            const FeatureMap& featuremap,
                                                            std::vector<core::LineArray> const& templates,
                                                            const core::LineArray& scene) const = 0;
        };

        template<IsMatchStrategyInstance T>
        struct MatcherModel : public MatcherConcept
        {
            explicit MatcherModel( T value ) noexcept : object{ std::move(value) }
            {}

            [[nodiscard]] std::unique_ptr<MatcherConcept> clone() const final
            {
                return std::make_unique<MatcherModel<T>>(*this);
            }

            [[nodiscard]] std::vector<Match>
                    search(const SearchStrategy &searcher, const OptimizeStrategy &optimizer,
                           const FeatureMap& featuremap, const std::vector<core::LineArray>& templates,
                           const core::LineArray& scene) const final
            {
                return openfdcm::matching::search(object, searcher, optimizer, featuremap, templates, scene);
            }

            T object;
        };
    }

    /// Type erased MatchStrategy
    class MatchStrategy : public MatchStrategyInstance
    {
        std::unique_ptr<detail::MatcherConcept> pimpl;

    public:
        template<IsMatchStrategyInstance T>
        /* implicit */ MatchStrategy(T const& x) : pimpl{std::make_unique<detail::MatcherModel<T>>(x)}
        {}

        MatchStrategy(MatchStrategy const& other) : pimpl{other.pimpl->clone()} {}
        MatchStrategy& operator=(MatchStrategy const& other) { pimpl = other.pimpl->clone(); return *this; }
        MatchStrategy(MatchStrategy&& other) noexcept = default;
        MatchStrategy& operator=(MatchStrategy&& other) noexcept = default;

        [[nodiscard]] std::vector<Match> search(const SearchStrategy &searcher, const OptimizeStrategy &optimizer,
                                                const FeatureMap& featuremap, std::vector<core::LineArray> const& templates,
                                                const core::LineArray& scene) const
        {
            return this->pimpl->search(searcher, optimizer, featuremap, templates, scene);
        }
    };


    // ************************************************************************************************************
    // Free Functions for MatchStrategy
    // ************************************************************************************************************
    /**
     * @brief Search for optimal matches between the templates and the scene
     * @param matcher The matcher used to find and optimize matches
     * @param searcher The search strategy
     * @param optimizer The optimizer
     * @param templates The templates expressed as a vector of LineArrays
     * @param scene The scene expressed as a core::LineArray
     * @return A vector containing all the sorted matches by score (lowest to highest)
     */
    inline std::vector<Match> search(const MatchStrategy &matcher,
                                     const SearchStrategy &searcher,
                                     const OptimizeStrategy &optimizer,
                                     const FeatureMap& featuremap,
                                     std::vector<core::LineArray> const& templates,
                                     const core::LineArray& scene)
    {
        return matcher.search(searcher, optimizer, featuremap, templates, scene);
    }
} // namespace openfdcm::matching


#endif //OPENFDCM_SEARCH_H
