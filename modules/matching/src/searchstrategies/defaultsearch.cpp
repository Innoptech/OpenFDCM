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

#include "openfdcm/matching/searchstrategies/defaultsearch.h"

namespace openfdcm::matching
{
    template<> std::vector<SearchCombination>
    establishSearchStrategy(DefaultSearch const& strategy, core::LineArray const& tmpl_lines, core::LineArray const& scene_lines)
    {
        const Eigen::VectorXf& scene_lengths = core::getLength(scene_lines);
        const Eigen::VectorXf& tmpl_lengths = core::getLength(tmpl_lines);
        const std::vector<long>& sorted_scene_idx = core::argsort(scene_lengths, std::greater<>());
        const std::vector<long>& sorted_tmpl_idx = core::argsort(tmpl_lengths, std::greater<>());
        const Eigen::VectorXf& sorted_scene_len = scene_lengths(sorted_scene_idx);

        std::vector<SearchCombination> combinations{};
        for (auto it = sorted_tmpl_idx.begin()
                ; it != sorted_tmpl_idx.begin() + (int)std::min((size_t)tmpl_lines.cols(), strategy.getMaxTmplLines()); ++it)
        {
            const size_t closest_scene_idx = core::binarySearch<>(sorted_scene_len, tmpl_lengths(*it), std::greater{});
            const CenteredRange& range = getCenteredRange(
                    closest_scene_idx, sorted_scene_len.size(), strategy.getMaxSceneLines());
            for(size_t i{range.begin};i<range.end;++i)
                combinations.emplace_back(*it, sorted_scene_idx.at(i));
        }
        return combinations;
    }
}