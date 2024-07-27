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

#include "openfdcm/matching/matchstrategies/defaultmatch.h"

using namespace openfdcm::core;

namespace openfdcm::matching
{
    SceneShift getSceneCenteredTranslation(LineArray const& scene, float scene_padding) noexcept
    {
        auto const& [min_point, max_point] = minmaxPoint(scene);
        Point2 const distanceminmax = max_point - min_point;
        float const corrected_ratio = std::max(1.f, scene_padding);
        Point2 const required_max = corrected_ratio * distanceminmax.maxCoeff() * Point2{1.f, 1.f};
        Point2 const center_diff = required_max / 2.f - (max_point + min_point) / 2.f;
        return {center_diff, (required_max.array() + 1.f).ceil().cast<size_t>()};
    }

    template<>
    std::vector<Match> search(const DefaultMatch& matcher,
                              const SearchStrategy& searcher,
                              const OptimizeStrategy& optimizer,
                              std::vector<LineArray> const& templates,
                              LineArray const& originalScene)
    {
        if (originalScene.size() == 0 || templates.empty())
            return {};

        std::vector<Match> all_matches{};

        // Apply scene ratio
        const auto sceneRatio = matcher.getSceneRatio();
        const auto scene = sceneRatio * originalScene;

        // Shift the scene so that all scene lines are greater than 0.
        // DT3 requires that all line points have positive values
        SceneShift const& scene_shift = getSceneCenteredTranslation(scene, matcher.getScenePadding());
        LineArray const& shifted_scene = translate(scene, scene_shift.translation);

        // Build DT3 feature map
        const auto distanceScale = 1.f / sceneRatio; // Removes the effect of scaling down the scene
        auto const& featuremap = buildFeaturemap(
                matcher.getDepth(), scene_shift.sceneSize, matcher.getCoeff(), shifted_scene, distanceScale);

        // Matching
        for (size_t i = 0; i < templates.size(); ++i)
        {
            const LineArray& originalTmpl = templates.at(i);
            if (originalTmpl.size() == 0) continue;
            const auto& tmpl = sceneRatio * originalTmpl;
            for (SearchCombination const& combination : establishSearchStrategy(searcher, tmpl, originalScene))
            {
                const auto& scene_line = getLine(shifted_scene, combination.getSceneLineIdx());
                const auto& tmpl_line = getLine(tmpl, combination.getTmplLineIdx());
                const auto& align_vec = normalize(scene_line);
                for (auto const& initial_transf : align(tmpl_line, scene_line))
                {
                    const LineArray& aligned_tmpl = transform(tmpl, initial_transf);
                    std::optional<OptimalTranslation> const& result = optimize(
                            optimizer, aligned_tmpl, align_vec, featuremap);
                    if (result.has_value())
                    {
                        OptimalTranslation const& opt_transl = result.value();
                        Mat23 combined = combine(-scene_shift.translation,
                                                 combine(opt_transl.translation, initial_transf));
                        combined.block<2, 1>(0, 2) /= sceneRatio;
                        all_matches.push_back(
                                Match{int(i), opt_transl.score, combined}
                        );
                    }
                }
            }
        }

        std::sort(all_matches.begin(), all_matches.end());

        return all_matches;
    }
}