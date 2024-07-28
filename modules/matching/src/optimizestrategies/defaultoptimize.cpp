#include "openfdcm/matching/optimizestrategies/defaultoptimize.h"
#include "openfdcm/core/drawing.h"

namespace openfdcm::matching
{
    /**
     * @brief Compute the negative and positive values for the maximum translation of the template in the image window
     * @param tmpl The given template
     * @param align_vec The translation vector
     * @param featuresize The image window size
     * @return The negative and positive values for the maximum translation of the template in the image window
     */
    std::tuple<float, float> minmaxTranslation(
            const core::LineArray& tmpl, core::Point2 const& align_vec, core::Size const& featuresize) noexcept
    {
        if (core::allClose(align_vec, core::Point2{0,0}))
            return {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
        Eigen::Array<float,2,1> size = featuresize.cast<float>();
        auto const& [min_point, max_point] = core::minmaxPoint(tmpl);

        if (((size -1 - max_point.array()) < 0).any())
            return {NAN, NAN};
        if ((min_point.array() < 0).any())
            return {NAN, NAN};

        // Evaluate intersection of the four image boundaries for two lines
        // The reason is that align_vec can be negative or positive
        Eigen::Array<float,2,4> multipliers{};
        multipliers.block<2,1>(0,0) = - max_point;
        multipliers.block<2,1>(0,1) = - min_point;
        multipliers.block<2,1>(0,2) = (size - max_point.array() - 1.f);
        multipliers.block<2,1>(0,3) = (size - min_point.array() - 1.f);
        multipliers.colwise() /= align_vec.array();
        Eigen::Array<bool,2,4> const multiplier_sign = Eigen::Array<bool, 2,4>::NullaryExpr(
                [&multipliers](Eigen::Index row, Eigen::Index col) {return std::signbit(multipliers(row, col));});

        Eigen::Array<float,2,4> const pos_coeffs = (multiplier_sign)
                .select(std::numeric_limits<float>::infinity(), multipliers);
        Eigen::Array<float,2,4> const neg_coeffs = (multiplier_sign)
                .select(multipliers, -std::numeric_limits<float>::infinity());

        // extremum_coeffs:
        // | negative_x, negative_y |
        // | positive_x, positive_y |
        Eigen::Array<float,2,2> extremum_coeffs{
            {neg_coeffs.row(0).maxCoeff<Eigen::PropagateNaN>(), neg_coeffs.row(1).maxCoeff<Eigen::PropagateNaN>()},
            {pos_coeffs.row(0).minCoeff<Eigen::PropagateNaN>(), pos_coeffs.row(1).minCoeff<Eigen::PropagateNaN>()}
        };

        if (extremum_coeffs.allFinite())
            return {extremum_coeffs.row(0).maxCoeff(), extremum_coeffs.row(1).minCoeff()};
        if (extremum_coeffs.col(0).allFinite())
            return {extremum_coeffs(0,0), extremum_coeffs(1,0)};
        return {extremum_coeffs(0,1), extremum_coeffs(1,1)};
    }

    template<>
    std::optional<OptimalTranslation> optimize(DefaultOptimize const& optimizer, const core::LineArray& tmpl,
                                                      core::Point2 const& align_vec, FeatureMap const& featuremap)
    {
        (void) optimizer;
        if (core::relativelyEqual(align_vec.array().abs().sum(),0.f))
            return std::nullopt;
        core::Point2 const& scaled_align_vec = core::rasterizeVector(align_vec);
        auto const& [min_mul, max_mul] = minmaxTranslation(tmpl, scaled_align_vec, featuremap.getFeatureSize());
        if (!std::isfinite(min_mul) or !std::isfinite(max_mul))
            return std::nullopt;

        const float initial_score = evaluate(featuremap, {tmpl}, {core::Point2{0,0}}).at(0);
        std::vector<core::Point2> translations{core::Point2{0,0}};
        std::vector<float> scores{initial_score};
        long translation_multiplier{1};
        while (translation_multiplier <= (long)max_mul)
        {
            const core::Point2 translation = translation_multiplier*scaled_align_vec;
            const float score = evaluate(featuremap, {tmpl}, {translation}).at(0);
            if (score > scores.back())
                break;
            translations.emplace_back(translation);
            scores.emplace_back(score);
            ++translation_multiplier;
        }
        translations.emplace_back(core::Point2{0,0});
        scores.emplace_back(initial_score);
        translation_multiplier = -1;
        while (translation_multiplier >= (long)min_mul)
        {
            const core::Point2 translation = translation_multiplier*scaled_align_vec;
            const float score = evaluate(featuremap, {tmpl}, {translation}).at(0);
            if (score > scores.back())
                break;
            translations.emplace_back(translation);
            scores.emplace_back(score);
            --translation_multiplier;
        }
        size_t const best_score_idx = std::distance(scores.begin(), std::min_element(scores.begin(), scores.end()));
        return OptimalTranslation{scores.at(best_score_idx), translations.at(best_score_idx)};
    }
}