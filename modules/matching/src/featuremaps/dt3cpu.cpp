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
#include "openfdcm/matching/featuremaps/dt3cpu.h"

namespace openfdcm::matching
{
    namespace detail {
        std::array<float, 2>
        minmaxTranslation(const core::LineArray& tmpl, core::Point2 const& align_vec, core::Size const& featuresize,
                          core::Point2 const& extraTranslation)
        {
            if (core::allClose(align_vec, core::Point2{0,0}))
                return {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
            Eigen::Array<float,2,1> size = featuresize.cast<float>();
            auto [min_point, max_point] = core::minmaxPoint(tmpl);
            min_point += extraTranslation;
            max_point += extraTranslation;

            if (((size - 1 - max_point.array() ) < 0).any())
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

        size_t closestOrientationIndex(std::vector<float> const& sortedAngles, core::Line const& line)
        {
            assert(!sortedAngles.empty());

            // Get the angle of the line
            float line_angle = core::getAngle(line)(0, 0);

            // Perform binary search to find the first element that is >= line_angle
            auto itlow = std::upper_bound(std::begin(sortedAngles), std::end(sortedAngles), line_angle);

            if (itlow != std::end(sortedAngles) and itlow != std::begin(sortedAngles)) {
                const auto upper_bound_diff = std::abs(line_angle - *itlow);
                itlow--;
                const auto lower_bound_diff = std::abs(line_angle - *itlow);
                if (lower_bound_diff < upper_bound_diff)
                    return std::distance(std::begin(sortedAngles), itlow);;
                itlow++;
                return std::distance(std::begin(sortedAngles), itlow);;
            }
            itlow = std::end(sortedAngles);
            itlow--;
            const float angle1 = line_angle - *std::begin(sortedAngles);
            const float angle2 = line_angle - *itlow;
            if (std::min(angle1, std::abs(angle1 - M_PIf)) < std::min(angle2, std::abs(angle2 - M_PIf)))
                return 0;
            return sortedAngles.size() - 1;
        }

        void propagateOrientation(Dt3CpuMap<float> &featuremap, float coeff) noexcept {
            assert(featuremap.sortedAngles.size() == featuremap.features.size());

            // Precompute constants
            const int m = static_cast<int>(featuremap.sortedAngles.size());
            const int one_and_a_half_cycle_forward = static_cast<int>(std::ceil(1.5 * m));
            const int one_and_a_half_cycle_backward = -static_cast<int>(std::floor(1.5 * m));

            auto propagate = [&](int start, int end, int step) {
                for (int c = start; c != end; c += step) {
                    int c1 = (m + ((c - step) % m)) % m;
                    int c2 = (m + (c % m)) % m;

                    const float angle1 = featuremap.sortedAngles[c1];
                    const float angle2 = featuremap.sortedAngles[c2];

                    const float h = std::abs(angle1 - angle2);
                    const float min_h = std::min(h, std::abs(h - M_PIf));

                    featuremap.features[c2] = featuremap.features[c2].min(featuremap.features[c1] + coeff * min_h);
                }
            };

            propagate(0, one_and_a_half_cycle_forward, 1);
            propagate(m, one_and_a_half_cycle_backward, -1);
        }

        SceneShift getSceneCenteredTranslation(core::LineArray const &scene, float scene_padding) noexcept {
            auto const &[min_point, max_point] = core::minmaxPoint(scene);
            core::Point2 const distanceminmax = max_point - min_point;
            float const corrected_ratio = std::max(1.f, scene_padding);
            core::Point2 const required_max = corrected_ratio * distanceminmax.maxCoeff() * core::Point2{1.f, 1.f};
            core::Point2 const center_diff = required_max / 2.f - (max_point + min_point) / 2.f;
            return {center_diff, (required_max.array() + 1.f).ceil().cast<size_t>()};
        }
    } // namespace detail

    template<>
    std::array<float, 2>
    minmaxTranslation(const Dt3Cpu& featuremap, const core::LineArray& tmpl, core::Point2 const& align_vec)
    {
        return detail::minmaxTranslation(tmpl, align_vec, featuremap.getFeatureSize(), featuremap.getSceneTranslation());
    }

    template<>
    std::vector<std::vector<float>> evaluate(const Dt3Cpu& featuremap, const std::vector<core::LineArray>& templates,
                                             const std::vector<std::vector<core::Point2>>& translations)
    {
        const auto& dt3map = featuremap.getDt3Map();
        std::vector<std::vector<float>> scores(templates.size());

        // Preallocate Eigen structures
        Eigen::Matrix<int, 2, 1> point1, point2;
        for (size_t tmpl_idx = 0; tmpl_idx < templates.size(); ++tmpl_idx)
        {
            const auto& tmpl = templates[tmpl_idx];
            const auto& tmpl_translations = translations[tmpl_idx];
            scores[tmpl_idx].reserve(tmpl_translations.size());

            // Identify the line closest orientations, preallocate vector
            std::vector<size_t> lineOrientationIdx{}; lineOrientationIdx.reserve(tmpl.cols());
            for (long i = 0; i < tmpl.cols(); ++i)
            {
                const core::Line& line = core::getLine(tmpl, i);
                lineOrientationIdx[i] = detail::closestOrientationIndex(dt3map.sortedAngles, line);
            }

            // Translate the template on the transformed scene
            for (const auto& translation : tmpl_translations)
            {
                const core::LineArray& translatedTmpl = core::translate(tmpl, featuremap.getSceneTranslation() + translation);

                // Preallocate score array
                Eigen::VectorXf score_per_line(translatedTmpl.cols());

                for (long i = 0; i < translatedTmpl.cols(); ++i)
                {
                    const core::Line& line = core::getLine(translatedTmpl, i);
                    const auto& feature = dt3map.features[lineOrientationIdx[i]];

                    // Avoid casting inside the loop
                    point1 = core::p1(line).template cast<int>();
                    point2 = core::p2(line).template cast<int>();

                    // Access feature data directly
                    float lookup_p1 = feature(point1.y(), point1.x());
                    float lookup_p2 = feature(point2.y(), point2.x());

                    // Calculate score per line
                    score_per_line(i) = std::abs(lookup_p1 - lookup_p2);
                }
                scores[tmpl_idx].emplace_back(score_per_line.sum());
            }
        }
        return scores;
    }

}