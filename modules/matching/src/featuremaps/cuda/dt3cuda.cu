#include "openfdcm/matching/featuremaps/cuda/dt3cuda.cuh"
#include "../../../../../cmake-build-debug/_deps/eigen3-src/Eigen/Core"
#include <numeric>  // For std::exclusive_scan

namespace openfdcm::matching
{
    template<>
    std::array<float, 2>
    minmaxTranslation(const cuda::Dt3Cuda& featuremap, const core::LineArray& tmpl, core::Point2 const& align_vec)
    {
        return detail::minmaxTranslation(tmpl, align_vec, featuremap.getFeatureSize(), featuremap.getSceneTranslation());
    }

    void copyCudaArrayVectorToDevice(std::vector<core::cuda::CudaArray<float>> const& hostArray,
                                     core::cuda::CudaArray<float>*& deviceArray) {
        // Allocate space for the CudaArray objects on the device
        cudaMalloc(&deviceArray, hostArray.size() * sizeof(core::cuda::CudaArray<float>));

        // Copy each individual CudaArray to the device (just the pointers)
        for (size_t i = 0; i < hostArray.size(); ++i) {
            cudaMemcpy(deviceArray + i, &hostArray[i], sizeof(core::cuda::CudaArray<float>), cudaMemcpyHostToDevice);
        }
    }

    __global__
    void evaluateKernel(core::cuda::CudaArray<float>* d_array,
                        core::cuda::cuVector<size_t, -1> const& cuTmplIdxPerTransMap,
                        core::cuda::CudaArray<float, 2, -1> const& cuTrans,
                        core::cuda::cuVector<size_t, -1> const& cuLineIndices,
                        core::cuda::CudaArray<float, 4, -1> const& cuTmpl,
                        core::cuda::cuVector<size_t, -1> const& cuLineOrientationIdx,
                        core::cuda::cuVector<float, -1>& scores)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx >= cuTmplIdxPerTransMap.size())
            return;

        auto tmplIdx = cuTmplIdxPerTransMap(idx);
        auto translation_x = cuTrans(0, idx);
        auto translation_y = cuTrans(1, idx);
        auto orientationIdx  = cuLineOrientationIdx(idx);

        auto startLineIdx = cuLineIndices(tmplIdx);

        size_t endLineIdx;
        if(tmplIdx == cuTmplIdxPerTransMap.size() -1) {
            endLineIdx = cuTmpl.cols();
        } else {
            endLineIdx = cuLineIndices(tmplIdx+1);
        }

        // Iterate over each line
        scores(idx) = 0.f;
        for(size_t lineIdx{startLineIdx}; lineIdx < endLineIdx; ++lineIdx)
        {
            int p1x(cuTmpl(0,lineIdx)), p1y(cuTmpl(1,lineIdx)),
            p2x(cuTmpl(2,lineIdx)), p2y(cuTmpl(3,lineIdx));

            const auto& feature = d_array[orientationIdx];

            float lookup_p1 = feature(p1y, p1x);
            float lookup_p2 = feature(p2y, p2x);

            // Calculate score per line
            scores(idx) += fabs(lookup_p1 - lookup_p2);
        }

    }

    template<>
    std::vector<std::vector<float>> evaluate(const cuda::Dt3Cuda& featuremap,
                                             const std::vector<core::LineArray>& templates,
                                             const std::vector<std::vector<core::Point2>>& translations)
    {
        if(templates.empty()) return {};
        assert(templates.size() == translations.size());
        //------------------------------------------------------------------------------------------------
        // Allocate cuda memory
        //------------------------------------------------------------------------------------------------
        std::vector<size_t> translationSizePerTmpl{}; translationSizePerTmpl.reserve(translations.size());
        for(const auto& trans : translations) {
            translationSizePerTmpl.emplace_back(trans.size());
        }
        size_t transSize = std::accumulate(std::begin(translationSizePerTmpl), std::end(translationSizePerTmpl), 0ULL);

        std::vector<size_t> tmplIdxPerTrans{}; tmplIdxPerTrans.reserve(transSize);
        for(size_t tmplIdx{0}; tmplIdx < templates.size(); ++tmplIdx) {
            for(size_t transIdx{0}; transIdx < translations[tmplIdx].size(); ++transIdx) {
                tmplIdxPerTrans.emplace_back(tmplIdx);
            }
        }

        Eigen::Map<const Eigen::Vector<size_t, -1>> tmplIdxPerTransMap(tmplIdxPerTrans.data(), transSize);
        Eigen::Map<const Eigen::Array<float, 2, -1>> transMap(translations.data()->data()->data(), 2, transSize);
        core::cuda::CudaArray const cuTmplIdxPerTransMap(tmplIdxPerTransMap); // translations
        core::cuda::CudaArray const cuTrans(transMap); // translations


        std::vector<size_t> numLinesPerTmpl{}; numLinesPerTmpl.reserve(templates.size());
        for(const auto& tmpl : templates) {
            numLinesPerTmpl.emplace_back(tmpl.cols());
        }
        size_t tmplSize = std::accumulate(std::begin(numLinesPerTmpl), std::end(numLinesPerTmpl), 0ULL);

        std::vector<size_t> lineIndices(numLinesPerTmpl.size());
        std::exclusive_scan(std::begin(numLinesPerTmpl), std::end(numLinesPerTmpl), std::begin(lineIndices), 0);

        Eigen::Map<const Eigen::Vector<size_t, -1>> lineIndicesMap(lineIndices.data(), lineIndices.size());
        Eigen::Map<const Eigen::Array<float, 4, -1>> tmplMap(templates.data()->data(), 4, tmplSize);

        core::cuda::CudaArray const cuLineIndices(lineIndicesMap); // Start indices of lines per template
        core::cuda::CudaArray const cuTmpl(tmplMap); // templates

        const auto& dt3map = featuremap.getDt3Map();

        // Identify the line closest orientations, preallocate vector
        const auto& tmplAngles = openfdcm::core::getAngle(tmplMap);
        std::vector<size_t> lineOrientationIdx{}; lineOrientationIdx.reserve(tmplSize);
        for (long i = 0; i < tmplSize; ++i)
        {
            const core::Line& line = core::getLine(tmplMap, i);
            lineOrientationIdx[i] = detail::closestOrientationIndex(dt3map.sortedAngles, line);
        }
        Eigen::Map<const Eigen::Vector<size_t, -1>> lineOrientationIdxMap(lineOrientationIdx.data(), tmplSize);
        core::cuda::CudaArray const cuLineOrientationIdx(lineOrientationIdxMap); // line orientation indices

        // Copy an array of CudaArray to the GPU
        core::cuda::CudaArray<float>* d_array;
        copyCudaArrayVectorToDevice(dt3map.features, d_array);

        //------------------------------------------------------------------------------------------------
        // Evaluate
        //------------------------------------------------------------------------------------------------
        auto const& streamPool = featuremap.getStreamPool().lock();
        auto streamWrapper = streamPool->getStream();

        core::cuda::cuVector<float, -1> cuScores(transSize);

        // Set up grid and block sizes
        int threadsPerBlock = 256;
        int blocksPerGrid = (transSize + threadsPerBlock - 1) / threadsPerBlock;

        evaluateKernel<<<blocksPerGrid, threadsPerBlock, 0, streamWrapper->getStream()>>>(
                d_array,cuTmplIdxPerTransMap,cuTrans,cuLineIndices,cuTmpl,cuLineOrientationIdx, cuScores);
        core::cuda::synchronize(streamWrapper);

        streamPool->returnStream(std::move(streamWrapper));
        // Free device memory for CudaArray objects
        cudaFree(d_array);


        //------------------------------------------------------------------------------------------------
        // Copy scores back to CPU
        //------------------------------------------------------------------------------------------------
        std::vector<std::vector<float>> score{}; score.reserve(templates.size());
        for(size_t tmplIdx{0}; tmplIdx < templates.size(); ++tmplIdx) {
            score.emplace_back(translations[tmplIdx].size());
        }

        core::cuda::copyToCpu(Eigen::Map<Eigen::Vector<float,-1>>(score.data()->data(), transSize), cuScores);
        return score;
    }
}