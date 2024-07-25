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

#ifndef OPENFDCM_DEFAULTOPTIMIZE_H
#define OPENFDCM_DEFAULTOPTIMIZE_H
#include "openfdcm/matching/optimizestrategy.h"

namespace openfdcm::matching
{
    class DefaultOptimize : public OptimizeStrategyInstance
    {
    public:
        DefaultOptimize() = default;
    };

    /**
     * @brief Compute the negative and positive values for the maximum translation of the template in the image window
     * @param tmpl The given template
     * @param align_vec The translation vector
     * @param featuresize The image window size
     * @return The negative and positive values for the maximum translation of the template in the image window
     */
    inline std::tuple<float, float> minmaxTranslation(
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

    inline float evaluate(const core::LineArray& tmpl, core::FeatureMap<float> const& featuremap) noexcept
    {
        Eigen::VectorXf scores(tmpl.cols());
        for(long i{0};i<tmpl.cols();++i)
        {
            core::Line const& line = core::getLine(tmpl, i);
            auto const& it = core::closestOrientation(featuremap, line);
            Eigen::Matrix<int,2,1> point1{core::p1(line).cast<int>()}, point2{core::p2(line).cast<int>()};
            core::RawImage<float> const feature = it->second;
            float const lookup_p1 = feature(point1.y(), point1.x());
            float const lookup_p2 = feature(point2.y(), point2.x());
            scores(i) = lookup_p1-lookup_p2;
        }
        return scores.array().abs().sum();
    }

    template<>
    inline std::optional<OptimalTranslation> optimize(DefaultOptimize const& optimizer, const core::LineArray& tmpl,
                                                      core::Point2 const& align_vec, core::FeatureMap<float> const& featuremap)
    {
        (void) optimizer;
        if (core::relativelyEqual(align_vec.array().abs().sum(),0.f))
            return std::nullopt;
        core::Point2 const& scaled_align_vec = core::rasterizeVector(align_vec);
        auto const& [min_mul, max_mul] = minmaxTranslation(tmpl, scaled_align_vec, core::getFeatureSize(featuremap));
        if (!std::isfinite(min_mul) or !std::isfinite(max_mul))
            return std::nullopt;

        const float initial_score = evaluate(tmpl, featuremap);
        std::vector<core::Point2> translations{core::Point2{0,0}};
        std::vector<float> scores{initial_score};
        long translation_multiplier{1};
        while (translation_multiplier <= (long)max_mul)
        {
            const core::Point2 translation = translation_multiplier*scaled_align_vec;
            const core::LineArray& translated_tmpl = core::translate(tmpl, translation);
            const float score = evaluate(translated_tmpl, featuremap);
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
            const core::LineArray& translated_tmpl = core::translate(tmpl, translation);
            const float score = evaluate(translated_tmpl, featuremap);
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

#endif //OPENFDCM_DEFAULTOPTIMIZE_H
