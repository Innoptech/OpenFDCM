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

#ifndef OPENFDCM_DEFAULTSTRATEGY_H
#define OPENFDCM_DEFAULTSTRATEGY_H
#include "openfdcm/matching/searchstrategy.h"

namespace openfdcm::matching
{
    struct CenteredRange{ size_t begin, end; };

    /**
     * @brief Find a range that is centered with a given index and that has a given max length
     * @param center_idx The center index of the ranger
     * @param vec_size The size of the vector to consider
     * @param max_length The max length of the range
     * @return The resulting centered range
     */
    inline CenteredRange getCenteredRange(size_t const center_idx, size_t const vec_size, const size_t max_length) noexcept
    {
        CenteredRange range{};
        range.begin = std::max(0, int(center_idx) - int(max_length / 2));
        range.end = std::min(size_t(range.begin + max_length), vec_size);
        range.begin = (size_t)std::max(0, int(range.end) - int(max_length));
        return range;
    }

    /** BaseStrategy establish a search strategy for which each N longest template lines
     *  are matched against the M most similar scene lines in terms of length.
     */
    class DefaultSearch : public SearchStrategyInstance
    {
    public:
        DefaultSearch(size_t const max_tmpl_lines, size_t const max_scene_lines)
        : max_tmpl_lines_{max_tmpl_lines}, max_scene_lines_{max_scene_lines}
        {}

        [[nodiscard]] size_t getMaxTmplLines() const noexcept {return max_tmpl_lines_;}
        [[nodiscard]] size_t getMaxSceneLines() const noexcept {return max_scene_lines_;}

    private:
        size_t max_tmpl_lines_, max_scene_lines_;
    };

} // namespace openfdcm::matching

#endif //OPENFDCM_DEFAULTSTRATEGY_H
