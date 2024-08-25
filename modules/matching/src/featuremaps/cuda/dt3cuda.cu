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

#include "openfdcm/matching/featuremaps/cuda/dt3cuda.cuh"
#include <numeric>  // For std::exclusive_scan

namespace openfdcm::matching
{
    template<>
    std::array<float, 2>
    minmaxTranslation(const cuda::Dt3Cuda& featuremap, const core::LineArray& tmpl, core::Point2 const& align_vec)
    {
        return detail::minmaxTranslation(tmpl, align_vec, featuremap.getFeatureSize(), featuremap.getSceneTranslation());
    }

    template<typename DerivedArray>
    void copyCudaArrayVectorToDevice(std::vector<std::shared_ptr<DerivedArray>> const& hostArray,
                                     DerivedArray*& deviceArray) {
        // Allocate space for the CudaArray objects on the device
        cudaMalloc(&deviceArray, hostArray.size() * sizeof(DerivedArray));

        // Copy each individual CudaArray to the device (just the pointers)
        for (size_t i = 0; i < hostArray.size(); ++i) {
            cudaMemcpy(deviceArray + i, hostArray[i].get(), sizeof(DerivedArray), cudaMemcpyHostToDevice);
        }
    }

    __global__
    void evaluateKernel(core::cuda::CudaArray<float,-1,-1>* d_array,
                        core::cuda::cuVector<size_t, -1> const& cuTmplIdxPerTrans,
                        core::cuda::CudaArray<float, 2, -1> const& cuTrans,
                        core::cuda::cuVector<size_t, -1> const& cuLineIndices,
                        core::cuda::CudaArray<float, 4, -1> const& cuTmpl,
                        core::cuda::cuVector<size_t, -1> const& cuLineOrientationIdx,
                        core::cuda::cuVector<float, -1>& scores)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx >= cuTmplIdxPerTrans.size())
            return;

        auto tmplIdx = cuTmplIdxPerTrans(idx);
        auto translation_x = cuTrans(0, idx);
        auto translation_y = cuTrans(1, idx);
        auto orientationIdx  = cuLineOrientationIdx(idx);

        auto startLineIdx = cuLineIndices(tmplIdx);

        size_t endLineIdx;
        if(tmplIdx == cuTmplIdxPerTrans.size() -1) {
            endLineIdx = cuTmpl.cols();
        } else {
            endLineIdx = cuLineIndices(tmplIdx+1);
        }

        // Iterate over each line
        scores(idx) = 0.f;
        for(size_t lineIdx{startLineIdx}; lineIdx < endLineIdx; ++lineIdx)
        {
            int p1x(cuTmpl(0,lineIdx)+translation_x), p1y(cuTmpl(1,lineIdx)+translation_y),
            p2x(cuTmpl(2,lineIdx)+translation_x), p2y(cuTmpl(3,lineIdx)+translation_y);

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

        core::cuda::cuVector<size_t, -1> const cuTmplIdxPerTrans(tmplIdxPerTrans.data(), transSize); // translations
        core::cuda::CudaArray<float,2,-1> const cuTrans(translations.data()->data()->data(), 2, transSize); // translations


        std::vector<size_t> numLinesPerTmpl{}; numLinesPerTmpl.reserve(templates.size());
        for(const auto& tmpl : templates) {
            numLinesPerTmpl.emplace_back(tmpl.cols());
        }
        size_t tmplSize = std::accumulate(std::begin(numLinesPerTmpl), std::end(numLinesPerTmpl), 0ULL);

        std::vector<size_t> lineIndices(numLinesPerTmpl.size());
        std::exclusive_scan(std::begin(numLinesPerTmpl), std::end(numLinesPerTmpl), std::begin(lineIndices), 0);

        core::cuda::cuVector<size_t, -1> const cuLineIndices(lineIndices.data(), lineIndices.size()); // Start indices of lines per template
        core::cuda::CudaArray<float,4,-1> const cuTmpl(templates.data()->data(), 4, tmplSize); // templates

        const auto& dt3map = featuremap.getDt3Map();

        // Identify the line closest orientations, preallocate vector
        Eigen::Map<const Eigen::Array<float, 4, -1>> tmplMap(templates.data()->data(), 4, tmplSize);
        const auto& tmplAngles = openfdcm::core::getAngle(tmplMap);
        std::vector<size_t> lineOrientationIdx{}; lineOrientationIdx.reserve(tmplSize);
        for (long i = 0; i < tmplSize; ++i)
        {
            const core::Line& line = core::getLine(tmplMap, i);
            lineOrientationIdx[i] = detail::closestOrientationIndex(dt3map.sortedAngles, line);
        }
        core::cuda::cuVector<size_t, -1> cuLineOrientationIdx(lineOrientationIdx.data(), tmplSize);

        // Copy an array of CudaArray to the GPU
        core::cuda::CudaArray<float,-1,-1>* d_array;
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
                d_array,cuTmplIdxPerTrans,cuTrans,cuLineIndices,cuTmpl,cuLineOrientationIdx, cuScores);
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