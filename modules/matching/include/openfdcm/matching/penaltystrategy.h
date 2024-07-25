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

#ifndef OPENFDCM_PENALTYSTRATEGY_H
#define OPENFDCM_PENALTYSTRATEGY_H
#include "openfdcm/matching/matchStrategy.h"

namespace openfdcm::matching
{
    class PenaltyStrategy;

    // ************************************************************************************************************
    // Concepts
    // ************************************************************************************************************
    class PenaltyStrategyInstance {
    };

    template<typename T>
    concept IsPenaltyStrategyInstance = std::is_base_of_v<PenaltyStrategyInstance, T>;

    // ************************************************************************************************************
    // Functions for penalty implementations
    // ************************************************************************************************************
    /**
     * @brief Apply a given score penalty on a vector of matches
     * @tparam T The PenaltyStrategy type
     * @param penalty The given penalty
     * @param matches The vector of matches
     * @return A vector of matches with the applied score penalty
     */
    template<IsPenaltyStrategyInstance T>
    std::vector<Match> penalize(T const& penalty, const std::vector<Match> &matches,
                                const std::vector<float> &templatelengths);


    namespace detail {
        struct PenaltyConcept
        {
            virtual ~PenaltyConcept() noexcept = default;
            [[nodiscard]] virtual std::unique_ptr<PenaltyConcept> clone() const = 0;
            [[nodiscard]] virtual std::vector<Match> penalize(const std::vector<Match> &matches,
                                                              const std::vector<float> &templatelengths) const = 0;
        };

        template<IsPenaltyStrategyInstance T>
        struct PenaltyModel : public PenaltyConcept
        {
            explicit PenaltyModel( T value ) noexcept : object{ std::move(value) }
            {}

            [[nodiscard]] std::unique_ptr<PenaltyConcept> clone() const final
            {
                return std::make_unique<PenaltyModel<T>>(*this);
            }

            [[nodiscard]] std::vector<Match> penalize(const std::vector<Match> &matches,
                                                      const std::vector<float> &templatelengths) const final
            {
                return openfdcm::matching::penalize(object, matches, templatelengths);
            }

            T object;
        };
    }

    /// Type erased PenaltyStrategy
    class PenaltyStrategy : public PenaltyStrategyInstance
    {
        std::unique_ptr<detail::PenaltyConcept> pimpl;

    public:
        template<IsPenaltyStrategyInstance T>
        /* implicit */ PenaltyStrategy(T const& x) : pimpl{std::make_unique<detail::PenaltyModel<T>>(x)}
        {}

        PenaltyStrategy(PenaltyStrategy const& other) : pimpl{other.pimpl->clone()} {}
        PenaltyStrategy& operator=(PenaltyStrategy const& other) { pimpl = other.pimpl->clone(); return *this; }
        PenaltyStrategy(PenaltyStrategy&& other) noexcept = default;
        PenaltyStrategy& operator=(PenaltyStrategy&& other) noexcept = default;

        [[nodiscard]] std::vector<Match> penalize(const std::vector<Match> &matches,
                                                  const std::vector<float> &templatelengths) const
        {
            return this->pimpl->penalize(matches, templatelengths);
        }
    };


    // ************************************************************************************************************
    // Free Functions for PenaltyStrategy
    // ************************************************************************************************************
    /**
      * @brief Apply a given score penalty on a vector of matches
      * @param penalty The given penalty
      * @param matches The vector of matches
      * @return A vector of matches with the applied score penalty
      */
    inline std::vector<Match> penalize(const PenaltyStrategy &penalty, const std::vector<Match> &matches,
                                       const std::vector<float> &templatelengths)
    {
        return penalty.penalize(matches, templatelengths);
    }
}


#endif //OPENFDCM_PENALTYSTRATEGY_H
