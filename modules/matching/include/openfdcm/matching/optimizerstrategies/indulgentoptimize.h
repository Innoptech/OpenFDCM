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

#ifndef OPENFDCM_OPTIMIZERSTRATEGIES_INDULGENTOPTIMIZE_H
#define OPENFDCM_OPTIMIZERSTRATEGIES_INDULGENTOPTIMIZE_H
#include "openfdcm/matching/optimizerstrategies/defaultoptimize.h"

namespace openfdcm::matching {
    class IndulgentOptimize : public OptimizeStrategyInstance
    {
        uint32_t indulgentNumberOfPassthroughs_;
    public:
        IndulgentOptimize(uint32_t indulgentNumberOfPassthroughs) :
        indulgentNumberOfPassthroughs_{indulgentNumberOfPassthroughs}
        {}

        [[nodiscard]] auto getNumberOfPassthroughs() const {return indulgentNumberOfPassthroughs_;}
    };


    template<>
    inline std::optional<OptimalTranslation> optimize(IndulgentOptimize const& optimizer, const core::LineArray& tmpl,
                                                      core::Point2 const& align_vec, core::FeatureMap<float> const& featuremap)
    {
        (void) optimizer;
        if (core::relativelyEqual(align_vec.array().abs().sum(),0.f))
            return std::nullopt;
        core::Point2 const& scaled_align_vec = core::rasterizeVector(align_vec);
        auto const& [min_mul, max_mul] = minmaxTranslation(tmpl, scaled_align_vec, core::getFeatureSize(featuremap));
        if (!std::isfinite(min_mul) or !std::isfinite(max_mul))
            return std::nullopt;

        const auto allowedNumPassthroughs = optimizer.getNumberOfPassthroughs();

        const float initial_score = evaluate(tmpl, featuremap);
        std::vector<core::Point2> translations{{0,0}};
        std::vector<float> scores{initial_score};
        long translation_multiplier{1};
        uint32_t passthroughs{0u};
        while (translation_multiplier <= (long)max_mul)
        {
            const core::Point2 translation = translation_multiplier*scaled_align_vec;
            const core::LineArray& translated_tmpl = core::translate(tmpl, translation);
            const float score = evaluate(translated_tmpl, featuremap);
            if (score > scores.back())
            {
                if(passthroughs >= allowedNumPassthroughs)
                    break;
                ++passthroughs;
                continue;
            }
            translations.emplace_back(translation);
            scores.emplace_back(score);
            ++translation_multiplier;
        }
        passthroughs = 0u; // reset passthroughs
        translations.emplace_back(0,0);
        scores.emplace_back(initial_score);
        translation_multiplier = -1;
        while (translation_multiplier >= (long)min_mul)
        {
            const core::Point2 translation = translation_multiplier*scaled_align_vec;
            const core::LineArray& translated_tmpl = core::translate(tmpl, translation);
            const float score = evaluate(translated_tmpl, featuremap);
            if (score > scores.back())
            {
                if(passthroughs >= allowedNumPassthroughs)
                    break;
                ++passthroughs;
                continue;
            }
            translations.emplace_back(translation);
            scores.emplace_back(score);
            --translation_multiplier;
        }
        size_t const best_score_idx = std::distance(scores.begin(), std::min_element(scores.begin(), scores.end()));
        return OptimalTranslation{scores.at(best_score_idx), translations.at(best_score_idx)};
    }
} //namespace openfdcm::optimizerstrategies
#endif //OPENFDCM_OPTIMIZERSTRATEGIES_INDULGENTOPTIMIZE_H
