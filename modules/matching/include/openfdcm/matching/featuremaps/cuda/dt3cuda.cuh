#ifndef OPENFDCM_CUDA_DT3CUDA_CUH
#define OPENFDCM_CUDA_DT3CUDA_CUH
#include "openfdcm/matching/featuremaps/dt3cpu.h"
#include "openfdcm/core/cuda/cuimgproc.cuh"
#include "openfdcm/core/cuda/adapters.cuh"
#include <BS_thread_pool.hpp>

#include <utility>
#include <vector>

namespace openfdcm::matching {

    namespace cuda {

        struct Dt3CudaParameters {
            size_t depth{};   // The number of features (discrete orientation \in [-Pi/2, PI/2])
            float dt3Coeff{}, // The orientation propagation coefficient
            padding;   // The padding ratio (paddedSceneSize = padding * sceneSize

            explicit Dt3CudaParameters(size_t _depth = 30, float _dt3Coeff = 5.f, float _padding = 2.2f) :
                    depth{_depth}, dt3Coeff{_dt3Coeff}, padding{_padding} {};
        };

        struct Dt3CudaMap {
            std::vector<core::cuda::CudaArray<float>> features;
            std::vector<float> sortedAngles;
        };


        class Dt3Cuda : public FeatureMapInstance {
            Dt3CudaMap dt3map_;
            core::Point2 sceneTranslation_;
            core::Size featureSize_;
            std::shared_ptr<core::cuda::CudaStreamPool> streamPool_;

        public:
            Dt3Cuda(Dt3CudaMap dt3map, core::Point2 sceneTranslation, core::Size featureSize,
                    std::shared_ptr<core::cuda::CudaStreamPool> streamPool)
                    : dt3map_{std::move(dt3map)}, sceneTranslation_{std::move(sceneTranslation)},
                      featureSize_{std::move(featureSize)}, streamPool_{std::move(streamPool)} {}

            [[nodiscard]] auto getSceneTranslation() const { return sceneTranslation_; }

            [[nodiscard]] auto getFeatureSize() const { return featureSize_; }

            [[nodiscard]] auto &getDt3Map() const { return dt3map_; }

            [[nodiscard]] std::weak_ptr<core::cuda::CudaStreamPool> getStreamPool() const { return streamPool_; }
        };

        /**
        * @brief Build the featuremap to perform the Fast Directional Chamfer Matching algorithm
        * @param scene The scene lines
        * @param params The Featuremap parameters
        * @param pool A threadpool to generate workers for the cuda map generation
        * @param pool A streampool to parallelize feature map generation on cuda
        * @return The FDCM featuremap
        */
        template<core::cuda::Distance D = core::cuda::Distance::L2>
        inline cuda::Dt3Cuda buildCudaFeaturemap(const core::LineArray &scene, const cuda::Dt3CudaParameters &params,
                                                 const std::shared_ptr<BS::thread_pool> &threadPool = std::make_shared<BS::thread_pool>(),
                                                 const std::shared_ptr<core::cuda::CudaStreamPool> &streamPool =
                                                 std::make_shared<core::cuda::CudaStreamPool>()) noexcept(false) {
            if (scene.cols() == 0)
                return {{}, core::Point2{0, 0}, core::Size{0, 0}, streamPool};

            // Shift the scene so that all scene lines are greater than 0.
            openfdcm::matching::detail::SceneShift const &sceneShift =
                    openfdcm::matching::detail::getSceneCenteredTranslation(scene, params.padding);
            const core::LineArray translatedScene = core::translate(scene, sceneShift.translation);

            cuda::Dt3CudaMap dt3map{};
            dt3map.features.reserve(params.depth);
            dt3map.sortedAngles.reserve(params.depth);

            // Step 1: Define a number of linearly spaced angles
            for (size_t i{0}; i < params.depth; i++) {
                dt3map.features.emplace_back(sceneShift.sceneSize.y(), sceneShift.sceneSize.x());
                dt3map.sortedAngles.emplace_back(float(i) * M_PIf / float(params.depth) - M_PI_2f);
            }
            assert(std::is_sorted(std::begin(dt3map.sortedAngles), std::end(dt3map.sortedAngles)));

            // Step 2: Classify lines
            auto const &indices = matching::detail::classifyLines(dt3map.sortedAngles, translatedScene);

            auto func = [&](size_t angleIdx) {
                // Acquire a stream from the pool
                auto streamWrapper = streamPool->getStream();

                auto const &orientedSceneLine = scene(Eigen::all, indices.at(angleIdx)).eval();
                core::cuda::CudaArray cuScene(translatedScene(Eigen::all, indices));

                auto &d_img = dt3map.features[angleIdx];

                // Initialize the image with max values
                setAll(d_img, 0.f, streamWrapper);
                core::cuda::synchronize(streamWrapper);
                drawLines(d_img, cuScene, 0, streamWrapper);
                core::cuda::synchronize(streamWrapper);

                core::cuda::distanceTransform<D>(dt3map.features.at(angleIdx), cuScene, streamWrapper);
                streamPool->returnStream(std::move(streamWrapper));
            };

            if (threadPool) {
                // Submit tasks to the thread pool for each angle
                std::vector<std::future<void>> futures;
                for (size_t angleIdx{0}; angleIdx < dt3map.sortedAngles.size(); ++angleIdx) {
                    futures.push_back(threadPool->submit_task([=] { func(angleIdx); }));
                }
                // Wait for all tasks to complete
                for (auto &fut: futures) {
                    fut.get();
                }
            } else {
                // If no thread pool is available, run tasks sequentially
                for (size_t angleIdx{0}; angleIdx < dt3map.sortedAngles.size(); ++angleIdx) {
                    func(angleIdx);
                }
            }

            // Step 4: Propagate orientation
            auto streamWrapper = streamPool->getStream();
            propagateOrientation(dt3map.features, dt3map.sortedAngles, params.dt3Coeff, streamWrapper);
            streamPool->returnStream(std::move(streamWrapper));

            // Step 5: Line integral
            for (size_t angleIdx{0}; angleIdx < dt3map.sortedAngles.size(); ++angleIdx) {
                streamWrapper = streamPool->getStream();
                lineIntegral(dt3map.features[angleIdx], dt3map.sortedAngles[angleIdx], streamWrapper);
                streamPool->returnStream(std::move(streamWrapper));
            }

            return {std::move(dt3map), sceneShift.translation, sceneShift.sceneSize, streamPool};
        }
    }

    template<>
    inline core::Size getFeatureSize(const cuda::Dt3Cuda& featuremap) noexcept {
        return featuremap.getFeatureSize();
    }
} //namespace openfdcm::cuda
#endif //OPENFDCM_CUDA_DT3CUDA_CUH
