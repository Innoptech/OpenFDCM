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

#ifndef OPENFDCM_DEFAULTMATCH_H
#define OPENFDCM_DEFAULTMATCH_H
#include "openfdcm/matching/matchStrategy.h"

namespace openfdcm::matching
{
    class DefaultMatch : public MatcherInstance
    {
    public:
        DefaultMatch(size_t depth, float coeff, float sceneRatio=1.f, float scenePadding=1.f)
        : depth_{depth}, coeff_{coeff}, sceneRatio_{sceneRatio}, scenePadding_{scenePadding}
        {}
        [[nodiscard]] size_t getDepth() const noexcept {return depth_;}
        [[nodiscard]] float getCoeff() const noexcept {return coeff_;}
        [[nodiscard]] float getSceneRatio() const noexcept {return sceneRatio_;}
        [[nodiscard]] float getScenePadding() const noexcept {return scenePadding_;}
    private:
        size_t depth_;
        float coeff_, sceneRatio_, scenePadding_;
    };

    struct SceneShift
    {
        core::Point2 translation;
        core::Size sceneSize;
    };

    /**
     * @brief Find the transformation to center a scene in an positive set of boundaries
     * given the image area ratio (scene size vs image size)
     * @param scene The scene lines
     * @param scene_padding The ratio between the original scene area and the image scene area
     * @return The resulting SceneShift object
     */
    SceneShift getSceneCenteredTranslation(core::LineArray const& scene, float scene_padding) noexcept;
} // namespace openfdcm::matching
#endif //OPENFDCM_DEFAULTMATCH_H
