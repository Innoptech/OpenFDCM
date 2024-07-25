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

#include "openfdcm/matching/penaltystrategies/defaultpenalty.h"

namespace openfdcm::matching
{
    template<>
    std::vector<Match> penalize(DefaultPenalty const& penalty, std::vector<Match> const& matches,
                                const std::vector<float> &templatelengths)
    {
        try {
            std::vector<Match> applied_penalty{};
            for(Match const& match : matches)
            {
                size_t const tmpl_idx = match.tmplIdx;
                auto len = std::max(templatelengths.at(tmpl_idx), 1e-6f);
                float const applied_score = match.score/len;
                applied_penalty.emplace_back(tmpl_idx, applied_score, match.transform);
            }
            return applied_penalty;
        } catch (const std::out_of_range& e)
        {
            // Verify that max index in match is lower than templatelengths size
            auto max_it = std::max_element(
                    std::begin(matches),
                    std::end(matches),
                    [](const auto& a, const auto& b) {
                        return a.tmplIdx < b.tmplIdx;
                    }
            );
            if(max_it->tmplIdx >= templatelengths.size())
            {
                throw std::out_of_range{
                        "In penalize, the size of templatelengths is not consistent with match template indices"};
            }
        }
    }
}