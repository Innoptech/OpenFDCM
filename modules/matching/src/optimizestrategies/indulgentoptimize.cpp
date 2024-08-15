#include "openfdcm/matching/optimizestrategies/indulgentoptimize.h"
#include "openfdcm/core/drawing.h"

namespace openfdcm::matching
{
    template<>
    std::vector<std::optional<OptimalTranslation>>
    optimize(IndulgentOptimize const& optimizer, const std::vector<core::LineArray>& templates,
             std::vector<core::Point2> const& alignments, FeatureMap const& featuremap) {
        assert(templates.size() == alignments.size());

        std::vector<std::optional<OptimalTranslation>> result(templates.size());

        auto func = [&](size_t tmpl_idx) {
            const auto &align_vec = alignments.at(tmpl_idx);
            const auto &tmpl = templates.at(tmpl_idx);

            if (core::relativelyEqual(align_vec.array().abs().sum(), 0.f)) {
                result[tmpl_idx] = std::nullopt;
                return;
            }

            core::Point2 const &scaled_align_vec = core::rasterizeVector(align_vec);
            auto const &[min_mul, max_mul] = minmaxTranslation(featuremap, tmpl, scaled_align_vec);

            if (!std::isfinite(min_mul) || !std::isfinite(max_mul)) {
                result[tmpl_idx] = std::nullopt;
                return;
            }

            const auto allowedNumPassthroughs = optimizer.getNumberOfPassthroughs();
            const float initial_score = evaluate(featuremap, {tmpl}, {{core::Point2{0, 0}}}).at(0).at(0);

            std::vector<core::Point2> translations{{0, 0}};
            std::vector<float> scores{initial_score};
            long translation_multiplier{1};
            uint32_t passthroughs{0u};

            // Positive translations
            while (translation_multiplier <= (long) max_mul) {
                const core::Point2 translation = translation_multiplier * scaled_align_vec;
                const float score = evaluate(featuremap, {tmpl}, {{translation}}).at(0).at(0);

                if (score > scores.back()) {
                    if (passthroughs >= allowedNumPassthroughs) break;
                    ++passthroughs;
                    continue;
                }

                translations.emplace_back(translation);
                scores.emplace_back(score);
                ++translation_multiplier;
            }

            // Reset passthroughs and handle negative translations
            passthroughs = 0u;
            translations.emplace_back(0, 0);
            scores.emplace_back(initial_score);
            translation_multiplier = -1;

            while (translation_multiplier >= (long) min_mul) {
                const core::Point2 translation = translation_multiplier * scaled_align_vec;
                const float score = evaluate(featuremap, {tmpl}, {{translation}}).at(0).at(0);

                if (score > scores.back()) {
                    if (passthroughs >= allowedNumPassthroughs) break;
                    ++passthroughs;
                    continue;
                }

                translations.emplace_back(translation);
                scores.emplace_back(score);
                --translation_multiplier;
            }

            size_t const best_score_idx = std::distance(scores.begin(), std::min_element(scores.begin(), scores.end()));

            // Lock to safely access the result vector
            result[tmpl_idx] = OptimalTranslation{scores.at(best_score_idx), translations.at(best_score_idx)};
        };

        const auto& pool_ptr = optimizer.getPool().lock();
        if (pool_ptr) {
            // Use the thread pool to run tasks in parallel
            std::vector<std::future<void>> futures;
            for (size_t tmpl_idx = 0; tmpl_idx < templates.size(); ++tmpl_idx) {
                futures.push_back(pool_ptr->submit_task([=] { func(tmpl_idx); }));
            }

            // Wait for all tasks to complete
            for (auto &fut: futures) {
                fut.get();
            }
        } else {
            // Run tasks sequentially if no thread pool is available
            for (size_t tmpl_idx = 0; tmpl_idx < templates.size(); ++tmpl_idx) {
                func(tmpl_idx);
            }
        }

        return result;
    }
}