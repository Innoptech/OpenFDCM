#include "openfdcm/matching/optimizestrategies/indulgentoptimize.h"
#include "openfdcm/core/drawing.h"

namespace openfdcm::matching
{
    template<>
    std::optional<OptimalTranslation> optimize(IndulgentOptimize const& optimizer, const core::LineArray& tmpl,
                                                      core::Point2 const& align_vec, FeatureMap const& featuremap)
    {
        (void) optimizer;
        if (core::relativelyEqual(align_vec.array().abs().sum(),0.f))
            return std::nullopt;
        core::Point2 const& scaled_align_vec = core::rasterizeVector(align_vec);
        auto const& [min_mul, max_mul] = minmaxTranslation(tmpl, scaled_align_vec, getFeatureSize(featuremap));
        if (!std::isfinite(min_mul) or !std::isfinite(max_mul))
            return std::nullopt;

        const auto allowedNumPassthroughs = optimizer.getNumberOfPassthroughs();

        const float initial_score = evaluate(featuremap, {tmpl}, {core::Point2{0,0}}).at(0);
        std::vector<core::Point2> translations{{0,0}};
        std::vector<float> scores{initial_score};
        long translation_multiplier{1};
        uint32_t passthroughs{0u};
        while (translation_multiplier <= (long)max_mul)
        {
            const core::Point2 translation = translation_multiplier*scaled_align_vec;
            const float score = evaluate(featuremap, {tmpl}, {translation}).at(0);
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
            const float score = evaluate(featuremap, {tmpl}, {translation}).at(0);
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
}