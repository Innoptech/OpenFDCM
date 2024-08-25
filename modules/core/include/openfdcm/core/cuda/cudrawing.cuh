#ifndef OPENFDCM_CUDA_CUDRAWING_CUH
#define OPENFDCM_CUDA_CUDRAWING_CUH
#include "openfdcm/core/cuda/cumath.cuh"
#include "openfdcm/core/cuda/cuarray.cuh"

namespace openfdcm::core::cuda {
    struct Box
    {
        float xmin, xmax, ymin, ymax;
    };

    /**
     * Clip a line segment based on a given CropBox.
     *
     * This function takes a line segment represented as a Line and crops it
     * based on the provided CropBox. The Cohen-Sutherland line clipping algorithm is used to determine
     * whether the line lies inside or outside the CropBox and clip it accordingly. A Line that is
     * completely outside the CropBox is set to zero.
     *
     * @param line The input line segment to be cropped.
     * @param cropbox The box defining the crop zone.
     * @param deleteOob If true, delete the Out of Bound lines. Else replace them by a singular (0,0) point.
     * @return A cropped Line segment.
     */
    __device__
    Line clipLine(Line const& line, const Box &cropbox);

    /**
     * @brief Rasterize a vector in such a way that the angle is conserved and either x or y has a value of 1.
     * @param align_vec The desired vector to rasterize
     * @return The rasterized vector
     */
     __device__
    inline Point2 rasterizeVector(Point2 const& align_vec) noexcept
    {
        const float tan_angle = align_vec.y()/align_vec.x();
        if (tan_angle >= -1.0 and tan_angle < 1) // [-PI/4, 0[ U [0, PI/4[ U [-PI, -3PI/4] U [3PI/4, PI]
        {
            bool cond1{align_vec.x() < 0};
            return {1 - 2*cond1, tan_angle - 2.0*cond1*tan_angle};
        }
        bool cond2{align_vec.y() < 0};
        return {1 / tan_angle - 2.0*cond2*(1 / tan_angle), 1 - 2*cond2};
    }

    /**
     * @brief  Rasterize a line between two points
     * @param line The line to rasterize
     * @return The rasterized line expressed as an array of points
     */
    __device__
    inline Eigen::Array<Eigen::Index, 2, -1> rasterizeLine(Line const& line) noexcept
    {
        if (p2(line).isApprox(p1(line)))
            return p1(line).array().round().cast<Eigen::Index>();
        Point2 const& line_vec = p2(line) - p1(line);
        Point2 const& rastvec = rasterizeVector(line_vec);
        if (relativelyEqual(rastvec.x(), 0.0f))
        {
            int const size = int(line_vec.y() / rastvec.y()) + 1;
            Eigen::Matrix<float, 2, -1> rasterization(2, size);
            rasterization.row(0).setConstant(p1(line).x());
            rasterization.row(1) = Eigen::Matrix<float, 1, -1>::LinSpaced(size, p1(line).y(), p2(line).y());
            return rasterization.array().round().cast<Eigen::Index>();
        }
        if (relativelyEqual(rastvec.y(), 0.0f))
        {
            int const size = int(line_vec.x() / rastvec.x()) + 1;
            Eigen::Matrix<float, 2, -1> rasterization(2, size);
            rasterization.row(0) = Eigen::Matrix<float, 1, -1>::LinSpaced(size, p1(line).x(), p2(line).x());
            rasterization.row(1).setConstant(p1(line).y());
            return rasterization.array().round().cast<Eigen::Index>();
        }

        int size = static_cast<int>(max(line_vec.x() / rastvec.x(), line_vec.y() / rastvec.y())) + 1;
        Eigen::Matrix<float, 2, -1> rasterization(2, size);
        rasterization.row(0) = Eigen::Matrix<float, 1, -1>::LinSpaced(size, p1(line).x(), p2(line).x());
        rasterization.row(1) = Eigen::Matrix<float, 1, -1>::LinSpaced(size, p1(line).y(), p2(line).y());
        return rasterization.array().round().cast<Eigen::Index>();
    }

    __global__
    inline void drawLinesKernel(CudaArray<float,-1,-1>& d_img, cuLineArray const& culinearray, float const color) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= culinearray.cols()) return;

        // Each thread processes one line
        Eigen::Map<LineArray> linearray(culinearray.getArray(), culinearray.rows(), culinearray.cols());
        auto const& line = clipLine(getLine(linearray, idx), Box{0, static_cast<float>(d_img.cols()-1),
                                                                 0, static_cast<float>(d_img.rows()-1)});
        auto const& rasterization = rasterizeLine(line);

        // Draw the rasterized line on the image (column-major indexing)
        for (int i = 0; i < rasterization.cols(); ++i) {
            auto x = rasterization(0, i);
            auto y = rasterization(1, i);

            // Ensure the point is within bounds
            if (x >= 0 && x < d_img.cols() && y >= 0 && y < d_img.rows()) {
                d_img(y, x) = color;
            }
        }
    }

    /**
     * @brief Draw a rasterized line on a greyscale image
     * @param src The inputoutput image
     * @param linearray The array of lines to draw
     * @param color The greyscale color to draw
     * @param stream The cuda stream used to draw
     */
    __host__
    inline void drawLines(CudaArray<float,-1,-1>& img, cuLineArray const& culinearray,
                          float const color, CudaStreamPtr const& stream) noexcept(false)
    {
        if (culinearray.cols() == 0) return;

        auto num_lines = culinearray.cols();
        int threadsPerBlock = 256;
        auto blocksPerGrid = (num_lines + threadsPerBlock - 1) / threadsPerBlock;
        drawLinesKernel<<<blocksPerGrid, threadsPerBlock, 0, stream->getStream()>>>(img, culinearray, color);
    }
} //namespace openfdcm::cuda
#endif //OPENFDCM_CUDA_CUDRAWING_CUH
