#include "openfdcm/matching/optimizestrategies/batchoptimize.h"
#include "openfdcm/core/drawing.h"

namespace openfdcm::matching
{
    template<>
    std::vector<std::optional<OptimalTranslation>>
    optimize(BatchOptimize const& optimizer, const std::vector<core::LineArray>& templates,
             std::vector<core::Point2> const& alignments, FeatureMap const& featuremap)
    {
        assert(templates.size() == alignments.size());

        std::vector<std::optional<OptimalTranslation>> result(templates.size());

        auto func = [&](size_t tmpl_idx) {
            const auto& align_vec = alignments[tmpl_idx];
            const auto& tmpl = templates[tmpl_idx];

            // Early exit for zero alignment vectors
            if (core::relativelyEqual(align_vec.array().abs().sum(), 0.f)) {
                result[tmpl_idx] = std::nullopt;
                return;
            }

            // Precompute values
            const core::Point2 scaled_align_vec = core::rasterizeVector(align_vec);
            const auto& [min_mul, max_mul] = minmaxTranslation(featuremap, tmpl, scaled_align_vec);

            // Handle invalid translation multipliers
            if (!std::isfinite(min_mul) || !std::isfinite(max_mul)) {
                result[tmpl_idx] = std::nullopt;
                return;
            }

            // Precompute initial score
            const float initial_score = evaluate(featuremap, {tmpl}, {{core::Point2{0, 0}}})[0][0];
            std::vector<core::Point2> translations;
            std::vector<float> scores;

            // Preallocate memory for translations and scores
            const auto expected_translations = static_cast<size_t>(std::abs(max_mul - min_mul) + 1);
            translations.reserve(expected_translations);
            scores.reserve(expected_translations);

            translations.emplace_back(0, 0);
            scores.push_back(initial_score);

            const auto batchSize = static_cast<long>(optimizer.getBatchSize());

            // Positive translations loop (precompute translation_multiplier * scaled_align_vec once per loop iteration)
            for (long translation_multiplier = 1; translation_multiplier <= static_cast<long>(max_mul); translation_multiplier+=batchSize) {
                std::vector<core::Point2> batchtranslations{};
                // Produce batched evaluation
                for(long batch_translation_mul{translation_multiplier};
                batch_translation_mul<translation_multiplier+batchSize && batch_translation_mul<= static_cast<long>(max_mul);
                ++batch_translation_mul)
                {
                    batchtranslations.emplace_back(batch_translation_mul * scaled_align_vec);
                }
                const auto batchscores = evaluate(featuremap, {tmpl}, {batchtranslations})[0];

                auto minIter = std::min_element(std::begin(batchscores), std::end(batchscores));
                int argmin = std::distance(std::begin(batchscores), minIter);

                if (batchscores[argmin] > scores.back()) break;

                translations.emplace_back(batchtranslations[argmin]);
                scores.emplace_back(batchscores[argmin]);

                if (batchscores[argmin] < batchscores.back()) break;
            }

            // Negative translations loop (reset before processing)
            for (long translation_multiplier = -1; translation_multiplier >= static_cast<long>(min_mul); translation_multiplier-=batchSize) {
                std::vector<core::Point2> batchtranslations{};
                // Produce batched evaluation
                for(long batch_translation_mul{translation_multiplier};
                    batch_translation_mul>translation_multiplier-batchSize && batch_translation_mul>= static_cast<long>(min_mul);
                    --batch_translation_mul)
                {
                    batchtranslations.emplace_back(batch_translation_mul * scaled_align_vec);
                }
                const auto batchscores = evaluate(featuremap, {tmpl}, {batchtranslations})[0];

                auto minIter = std::min_element(std::begin(batchscores), std::end(batchscores));
                int argmin = std::distance(std::begin(batchscores), minIter);

                if (batchscores[argmin] > scores.back()) break;

                translations.emplace_back(batchtranslations[argmin]);
                scores.emplace_back(batchscores[argmin]);

                if (batchscores[argmin] < batchscores.back()) break;
            }

            // Find the translation with the best score
            size_t best_score_idx = std::distance(std::begin(scores), std::min_element(std::begin(scores), std::end(scores)));
            result[tmpl_idx] = OptimalTranslation{scores[best_score_idx], translations[best_score_idx]};
        };

        // Optimize by checking if thread pool is available or process sequentially
        const auto& pool_ptr = optimizer.getPool().lock();
        if (pool_ptr) {
            // Use the thread pool to process the tasks in parallel
            std::vector<std::future<void>> futures;
            futures.reserve(templates.size());  // Preallocate futures vector size
            for (size_t tmpl_idx = 0; tmpl_idx < templates.size(); ++tmpl_idx) {
                futures.push_back(pool_ptr->submit_task([=] { func(tmpl_idx); }));
            }

            // Wait for all tasks to complete
            for (auto& fut : futures) {
                fut.get();
            }
        } else {
            // Process sequentially if no thread pool is available
            for (size_t tmpl_idx = 0; tmpl_idx < templates.size(); ++tmpl_idx) {
                func(tmpl_idx);
            }
        }

        return result;
    }

}