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
                              std::vector<LineArray> const& templates,
                              LineArray const& originalScene)
    {
        if (templates.empty() or originalScene.cols() == 0 or featuremap.getFeatureSize() == core::Size{0,0})
            return {};

        std::vector<Match> all_matches{};

        // Matching
        std::vector<core::LineArray> aligned_templates{}; aligned_templates.reserve(templates.size()*2);
        std::vector<int> template_indices{}; template_indices.reserve(aligned_templates.size());
        std::vector<core::Point2> alignments{}; alignments.reserve(aligned_templates.size());
        std::vector<Mat23> transforms{}; transforms.reserve(aligned_templates.size());

        for (int tmpl_idx{0}; tmpl_idx<templates.size(); ++tmpl_idx)
        {
            auto const& tmpl = templates[tmpl_idx];
            if (tmpl.size() == 0) continue;
            for (SearchCombination const& combination : establishSearchStrategy(searcher, tmpl, originalScene))
            {
                const auto& scene_line = getLine(originalScene, combination.getSceneLineIdx());
                const auto& tmpl_line = getLine(tmpl, combination.getTmplLineIdx());
                const auto& align_vec = normalize(scene_line);

                auto const& [transf, transf_rev] = align(tmpl_line, scene_line);
                transforms.emplace_back(transf);
                transforms.emplace_back(transf_rev);
                template_indices.emplace_back(tmpl_idx);
                template_indices.emplace_back(tmpl_idx);
                aligned_templates.emplace_back(transform(tmpl, transf));
                aligned_templates.emplace_back(transform(tmpl, transf_rev));
                alignments.emplace_back(align_vec);
                alignments.emplace_back(align_vec);
            }
        }

        std::vector<std::optional<OptimalTranslation>> const& results = optimize(
                optimizer, aligned_templates, alignments, featuremap);

        for(size_t res_idx{0}; res_idx<aligned_templates.size(); ++res_idx)
        {
            auto const& res = results.at(res_idx);
            if (res.has_value())
            {

                OptimalTranslation const& opt_transl = *res;
                Mat23 combined = combine(opt_transl.translation, transforms.at(res_idx));
                all_matches.push_back(
                        Match{template_indices[res_idx], opt_transl.score, combined}
                );
            }
        }


        std::sort(all_matches.begin(), all_matches.end(),
                  [](const Match& lhs, const Match& rhs){return lhs.score < rhs.score;});

        return all_matches;
    }
}