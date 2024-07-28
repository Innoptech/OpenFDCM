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

    template<>
    std::vector<Match> search(const DefaultMatch& matcher,
                              const SearchStrategy& searcher,
                              const OptimizeStrategy& optimizer,
                              const FeatureMap& featuremap,
                              std::vector<LineArray> const& templates)
    {
        if (templates.empty())
            return {};

        LineArray const& originalScene = featuremap.getSceneLines();
        std::vector<Match> all_matches{};

        // Matching
        for (size_t i = 0; i < templates.size(); ++i)
        {
            const LineArray& tmpl = templates.at(i);
            if (tmpl.size() == 0) continue;
            for (SearchCombination const& combination : establishSearchStrategy(searcher, tmpl, originalScene))
            {
                const auto& scene_line = getLine(originalScene, combination.getSceneLineIdx());
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
                        Mat23 combined = combine(opt_transl.translation, initial_transf);
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