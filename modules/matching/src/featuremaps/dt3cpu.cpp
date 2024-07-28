#include <utility>

#include "openfdcm/matching/featuremaps/dt3cpu.h"
#include "openfdcm/core/imgproc.h"

namespace openfdcm::matching
{
    namespace detail {
        /**
         * @brief Propagate the distance transform in the orientation (feature) space
         * @param featuremap The feature map (32bits)
         * @param coeff The propagation coefficient
         */
        void propagateOrientation(detail::Dt3Map<float> &featuremap, float coeff) noexcept {
            // Collect angles
            std::vector<float> angles;
            angles.reserve(featuremap.size());
            for (const auto &item: featuremap) {
                angles.push_back(item.first);
            }

            // Precompute constants
            const int m = static_cast<int>(featuremap.size());
            const int one_and_a_half_cycle_forward = static_cast<int>(std::ceil(1.5 * m));
            const int one_and_a_half_cycle_backward = -static_cast<int>(std::floor(1.5 * m));

            auto propagate = [&](int start, int end, int step) {
                for (int c = start; c != end; c += step) {
                    int c1 = (m + ((c - step) % m)) % m;
                    int c2 = (m + (c % m)) % m;

                    const float angle1 = angles[c1];
                    const float angle2 = angles[c2];

                    const float h = std::abs(angle1 - angle2);
                    const float min_h = std::min(h, std::abs(h - M_PIf));

                    featuremap[angle2] = featuremap[angle2].min(featuremap[angle1] + coeff * min_h);
                }
            };

            propagate(0, one_and_a_half_cycle_forward, 1);
            propagate(m, one_and_a_half_cycle_backward, -1);
        }

        inline Eigen::VectorXi vectorToEigenVector(const std::vector<long> &vec) noexcept {
            Eigen::VectorXi eigenVec(vec.size());
            for (size_t i = 0; i < vec.size(); ++i) {
                eigenVec[i] = vec[i];
            }
            return eigenVec;
        }

        Dt3Map<float> buildFeaturemap(size_t depth, core::Size const &size,
                                      float coeff, core::LineArray const &linearray,
                                      float sceneScale,
                                      const std::weak_ptr<BS::thread_pool> &pool) noexcept(false) {
            // Step 1: Define a number of linearly spaced angles
            std::set<float> angles{};
            for (size_t i{0}; i < depth; i++)
                angles.insert(float(i) * M_PIf / float(depth) - M_PI_2f);

            // Step 2: Classify lines
            auto classified_lines = detail::classifyLines(angles, linearray);

            // Step 3: Build featuremap with distance transform
            detail::Dt3Map<float> dt3map{};

            auto func = [&](float angle) {
                Eigen::VectorXi indices = vectorToEigenVector(classified_lines[angle]);
                core::RawImage<float> distance_transformed =
                        core::distanceTransform<float>(linearray(Eigen::all, indices), size) * sceneScale;
                dt3map[angle] = std::move(distance_transformed);
            };

            auto pool_ptr = pool.lock();
            if (pool_ptr) {
                // Submit tasks to the thread pool for each angle
                std::vector<std::future<void>> futures;
                for (const auto &angle : angles) {
                    futures.push_back(pool_ptr->submit_task([=] { func(angle); }));
                }
                // Wait for all tasks to complete
                for (auto &fut : futures) {
                    fut.get();
                }
            } else {
                // If no thread pool is available, run tasks sequentially
                for (const auto &angle : angles) {
                    func(angle);
                }
            }

            // Step 4: Propagate orientation
            propagateOrientation(dt3map, coeff);

            // Step 5: Line integral
            for (auto &[lineAngle, feature] : dt3map)
                core::lineIntegral(feature, lineAngle);

            return dt3map;
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


    Dt3Cpu::Dt3Cpu(core::LineArray scene, Dt3CpuParameters params, const BS::concurrency_t num_threads)
    : params_{params}, scene_{std::move(scene)}, pool_{std::make_shared<BS::thread_pool>(num_threads)}
    {
        assert(scene_.size() > 0);
        // Apply scene ratio
        const auto sceneScale= params_.scale;
        transformedScene_ = sceneScale * scene_;

        // Shift the scene so that all scene lines are greater than 0.
        // DT3 requires that all line points have positive values
        detail::SceneShift const& sceneShift = detail::getSceneCenteredTranslation(transformedScene_, params_.padding);
        sceneTranslation_ = sceneShift.translation;
        sceneSize_ = sceneShift.sceneSize;
        transformedScene_ = core::translate(transformedScene_, sceneTranslation_);

        const auto distanceScale = 1.f / params_.scale; // Removes the effect of scaling down the scene
        dt3map_ = detail::buildFeaturemap(params_.depth, sceneShift.sceneSize,
                                          params_.dt3Coeff, transformedScene_, distanceScale, this->getThreadPool());
    }

    template<>
    std::vector<float> evaluate(const Dt3Cpu& featuremap, const std::vector<core::LineArray>& templates,
                                       const std::vector<core::Point2>& translations)
    {
        assert(templates.size() == translations.size());
        auto const& dt3map = featuremap.getDt3Map();
        auto const sceneScale = featuremap.getParams().scale;

        std::vector<float> scores{}; scores.reserve(templates.size());
        for (size_t tmpl_idx{0}; tmpl_idx<templates.size(); ++tmpl_idx)
        {
            // Translate the template on the transformed scene
            core::LineArray const& scaledTmpl = templates.at(tmpl_idx)*sceneScale;
            core::Point2 const& scaledTranslation = translations.at(tmpl_idx)*sceneScale;
            core::LineArray const& tmpl = core::translate(scaledTmpl,
                                                          featuremap.getSceneTranslation()+scaledTranslation);
            Eigen::VectorXf score_per_line(tmpl.cols());
            for(long i{0};i<tmpl.cols();++i)
            {
                core::Line const& line = core::getLine(tmpl, i);
                auto const& it = detail::closestOrientation(dt3map, line);
                Eigen::Matrix<int,2,1> point1{core::p1(line).cast<int>()}, point2{core::p2(line).cast<int>()};
                core::RawImage<float> const feature = it->second;
                float const lookup_p1 = feature(point1.y(), point1.x());
                float const lookup_p2 = feature(point2.y(), point2.x());
                score_per_line(i) = lookup_p1-lookup_p2;
            }
            scores.emplace_back(score_per_line.array().abs().sum());
        }
        return scores;
    }
}