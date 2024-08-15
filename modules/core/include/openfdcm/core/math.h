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

#ifndef OPENFDCM_MATH_H
#define OPENFDCM_MATH_H
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <functional>
#include <type_traits>
#include <Eigen/Dense>

#define 	M_PI   3.14159265358979323846 /* pi */
#define 	M_PIf   3.14159265358979323846f /* pi */
#define 	M_PI_2   1.57079632679489661923 /* pi/2 */
#define 	M_PI_2f   1.57079632679489661923f /* pi/2 */
#define 	M_PI_4   0.78539816339744830962 /* pi/4 */
#define 	M_PI_4f   0.78539816339744830962f /* pi/4 */

namespace openfdcm::core
{
    template<typename T> using RawImage = Eigen::Array<T, -1, -1, Eigen::ColMajor>;

    template<typename T> using DenseBase = Eigen::DenseBase<T>;
    template<typename T> using MatrixBase = Eigen::MatrixBase<T>;
    using Size = Eigen::Vector<size_t, 2>;
    using Point2 = Eigen::Vector<float, 2>;
    using Mat22 = Eigen::Matrix<float, 2, 2>;
    using Mat23 = Eigen::Matrix<float, 2, 3>;
    using Line = Eigen::Vector<float, 4>; // x1, y1, x2, y2
    using LineArray = Eigen::Matrix<float, 4, -1>;


    /**
    * @brief Sort the indices of a vector.
    * @tparam T The type of the vector
    * @tparam Compare The comparison function object
    * @param vec The vector to sort
    * @return The indices of the sorted array
    */
    template<typename T, typename Compare>
    inline std::vector<long> argsort(
            std::vector<T> const& vec, Compare comp) noexcept
    {
        std::vector<long> ind(std::max(vec.cols(), vec.rows()));
        std::iota(ind.begin(), ind.end(), 0);
        std::sort(ind.begin(), ind.end()
                , [&vec, &comp](size_t const i1, size_t const i2) {return comp(vec.at(i1), vec.at(i2));});
        return ind;
    }

    /**
    * @brief Sort the indices of a vector.
    * @tparam T The type of the vector
    * @param vec The vector to sort
    * @return The indices of the sorted array
    */
    template<typename T, typename Compare>
    inline std::vector<long> argsort(std::vector<T> const& vec) noexcept
    {
        return argsort(vec, std::less<>());
    }

    /**
    * @brief Sort the indices of a vector.
    * @tparam Derived The DenseBase type of the vector
    * @tparam Compare The comparison function object
    * @param vec The vector to sort
    * @return The indices of the sorted array
    */
    template<typename Derived, typename Compare>
    inline std::vector<long> argsort(
            Eigen::DenseBase<Derived> const& vec, Compare comp) noexcept
    {
        static_assert(Derived::RowsAtCompileTime == 1 or Derived::ColsAtCompileTime == 1);
        std::vector<long> ind(std::max(vec.cols(), vec.rows()));
        std::iota(ind.begin(), ind.end(), 0);
        std::sort(ind.begin(), ind.end()
                , [&vec, &comp](long const i1, long const i2) {return comp(vec(i1), vec(i2));});
        return ind;
    }

    /**
    * @brief Sort the indices of a vector.
    * @tparam The comparison function object
    * @param vec The vector to sort
    * @return The indices of the sorted array
    */
    template<typename Derived, typename Compare>
    inline std::vector<long> argsort(Eigen::DenseBase<Derived> const& vec) noexcept
    {
        return argsort(vec, std::less<>());
    }

    /**
     * @brief Find the closest value in the given sorted vector using binary search
     * @tparam T The vector type
     * @param sorted_vec The sorted vector
     * @param value The value to search for
     * @return The index of the closest value in the given sorted vector
     */
    template<typename Derived, class Compare>
    inline size_t binarySearch(Eigen::DenseBase<Derived> const& sorted_vec,
                               typename Derived::RealScalar const value, Compare comp) noexcept {
        static_assert(Derived::RowsAtCompileTime == 1 or Derived::ColsAtCompileTime == 1);
        auto it = std::lower_bound(sorted_vec.begin(), sorted_vec.end(), value, comp);
        if (it == sorted_vec.begin()) return 0;
        if (it == sorted_vec.end()) return std::distance(sorted_vec.begin(), it - 1);
        return std::abs(value - *it) < std::abs(value - *(it - 1)) ?
               std::distance(sorted_vec.begin(), it) : std::distance(sorted_vec.begin(), it - 1);
    }

    /**
     * @brief Find the closest value in the given sorted vector using binary search
     * @tparam T The vector type
     * @param sorted_vec The sorted vector
     * @param value The value to search for
     * @return The index of the closest value in the given sorted vector
     */
    template<typename Derived>
    inline size_t binarySearch(Eigen::DenseBase<Derived> const& sorted_vec,
                               typename Derived::RealScalar const value) noexcept {
        return binarySearch(sorted_vec, value, std::less{});
    }

    /**
     * @brief Find the min max points (boundary) of an array of lines
     * @param linearray The given array of lines
     * @return A tuple containing the min and max point
     */
    inline std::tuple<Point2, Point2> minmaxPoint(LineArray const& linearray) noexcept
    {
        Point2 const& max_point = linearray.reshaped(2, linearray.cols()*2).rowwise().maxCoeff();
        Point2 const& min_point = linearray.reshaped(2, linearray.cols()*2).rowwise().minCoeff();
        return {min_point, max_point};
    }

    /**
     * @brief Return true if two number are equals relatively to their respective precision
     * @tparam U The type of the numbers
     * @param a The first number to compare
     * @param b The second number to compare
     * @param rtol The relative tolerance
     * @param atol The absolute tolerance
     * @return true if two number are equals relatively to their respective precision
     */
    template<typename T, typename U>
    inline bool relativelyEqual(
            T const& a, U const& b,
            double const& rtol=1e-10, double const& atol=std::numeric_limits<T>::epsilon()) noexcept
    {
        return std::fabs(a - b) <= atol + rtol * std::max(std::fabs(a), std::fabs(b));
    }

    /**
     * @brief Returns true if two arrays are element-wise equal within a tolerance
     * from https://stackoverflow.com/a/15052131/10631984
     *
     * @tparam DerivedA The first matrix type
     * @tparam DerivedB The second matrix type
     * @param a The first matrix
     * @param b The second matrix
     * @param rtol The relative tolerance parameter
     * @param atol The absolute tolerance parameter
     * @return true if two arrays are element-wise equal within a tolerance.
     */
    template<typename DerivedA, typename DerivedB>
    inline bool allClose(DenseBase<DerivedA> const& a, DenseBase<DerivedB> const& b,
                         typename DerivedA::RealScalar const& rtol = 0.0,
                         typename DerivedA::RealScalar const& atol = 1e-5) noexcept
    {
        return ((a.derived().array() - b.derived().array()).array().abs() <= (atol + rtol * b.derived().array().abs())).all();
    }


    /**
     * @brief Constrains an angle between [-pi/2, pi/2)
     * @tparam T The type of the angle
     * @param x The angle in radians
     * @return The constrained angle in radians
     */
    template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    inline T constrainHalfAngle( T const y) noexcept(false)
    {
        T x = fmod(y+M_PI_2, M_PI);
        x += M_PI * (x < 0);
        return x - M_PI_2;
    }

    /**
     * @brief Constrains an array of angles between [-pi/2, pi/2)
     * @tparam Derived The type of the array
     * @param x The angle in radians
     * @return The constrained array of angles in radians
     */
    template<typename Derived>
    inline auto constrainHalfAngle(DenseBase<Derived> const& x) noexcept(false)
    {
        return x.unaryExpr([](auto const& elem) { return constrainHalfAngle(elem); });
    }

    /**
     * @brief Constrains an angle between [-pi, pi)
     * @tparam T The type of the angle
     * @param x The angle in radians
     * @return The constrained angle in radians
     */
    template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    inline T constrainAngle(T const x) noexcept(false)
    {
        T x_mod = fmod(x + M_PI, 2*M_PI);
        x_mod += 2*M_PI * (x_mod < 0.0);
        return x_mod - M_PI;
    }

    /**
     * @brief Constrains an array of angles between [-pi, pi)
     * @tparam Derived The type of the array
     * @param x The angle in radians
     * @return The constrained array of angles in radians
     */
    template<typename Derived>
    inline auto constrainAngle(DenseBase<Derived> const& x) noexcept(false)
    {
        return x.unaryExpr([](auto const& elem) { return constrainAngle(elem); });
    }


    inline double wrapMax(const double& x, const double& max)
    {
        return std::fmod(max + std::fmod(x, max), max);
    }

    inline double wrapMinMax(const double& x, const double& min, const double& max)
    {
        return min + wrapMax(x - min, max - min);
    }

    inline auto width(const Size& size){ return size.x(); }
    inline auto height(const Size& size){ return size.y(); }


    // ----------------------------------------------------------------------------
    // Line functionalities
    // ----------------------------------------------------------------------------

    inline Point2 p1(Line const& line) noexcept { return line.block<2,1>(0,0); }
    inline Point2 p2(Line const& line) noexcept { return line.block<2,1>(2,0); }
    inline Line getLine(LineArray const& linearray, long const idx) noexcept {return linearray.block<4,1>(0,idx);}

    inline Eigen::Matrix<float,2,-1> getCenter(LineArray const& linearray) noexcept {
        return (linearray.block(2,0,2,linearray.cols()) + linearray.block(0,0,2,linearray.cols()))/2;
    }

    /**
     * @brief Get the angle of a line or an array of lines in [-PI/2, PI/2[
     * @param linearray The line or the linearray of lines
     * @return The angle of each line expressed as a vector
     */
    inline Eigen::Matrix<float,1,-1> getAngle(LineArray const& linearray) noexcept {
        auto const diff = (linearray.block(2,0,2,linearray.cols()) - linearray.block(0,0,2,linearray.cols()));
        auto const y_x = diff.block(1,0,1,linearray.cols()).array()/diff.block(0,0,1,linearray.cols()).array();
        return y_x.array().atan();
    }

    /**
     * @brief Get the length of a line or an array of lines
     * @param linearray The line or the array of lines
     * @return The length of each line expressed as a vector
     */
    inline Eigen::Matrix<float,1,-1> getLength(LineArray const& linearray) noexcept {
        return (linearray.block(2,0,2,linearray.cols()) - linearray.block(0,0,2,linearray.cols())).colwise().norm();
    }

    /**
     * @brief Get the lengths of templates represented by line arrays.
     *
     * This function calculates the total length of each template represented by a vector
     * of line arrays. It returns a vector containing the total lengths of all templates.
     *
     * @param templates A vector of line arrays representing templates.
     * @return A vector containing the total lengths of all templates.
     */
    inline auto getTemplateLengths(const std::vector<openfdcm::core::LineArray> &templates)
    {
        std::vector<float> templatelengths; templatelengths.reserve(templates.size());
        for(const auto &tmpl : templates) templatelengths.emplace_back(openfdcm::core::getLength(tmpl).sum());
        return templatelengths;
    }

    /**
     * @brief Normalize each a line to a unit vector
     * @param linearray The line or the array of lines
     * @return The resulting unit vectors
     */
    inline Eigen::Matrix<float,2,-1> normalize(LineArray const& linearray) noexcept {
        return (linearray.block(2,0,2,linearray.cols()) - linearray.block(0,0,2,linearray.cols())).colwise().normalized();
    }

    /**
     * @brief Transform a line or an array of line given a 2x3 transformation matrix
     * @param linearray The line or the array of lines
     * @param transform_mat The 2x3 transformation matrix
     * @return The transformed line or array or lines
     */
    inline LineArray transform(LineArray const& linearray, Mat23 const& transform_mat) noexcept {
        auto const rotated = transform_mat.block<2,2>(0,0) * linearray.reshaped(2, linearray.cols()*2);
        return (rotated.array().colwise() + transform_mat.block<2,1>(0,2).array()).reshaped(4, linearray.cols());
    }

    /**
     * @brief Translate a line or an array of lines
     * @param linearray The line or the array of lines
     * @param translation The 2x1 translation vector
     * @return The translated line or array or lines
     */
    inline LineArray translate(LineArray const& linearray, Point2 const& translation) noexcept {
        return (linearray.reshaped(2,linearray.cols()*2).colwise() + translation).reshaped(4, linearray.cols());
    }

    /**
     * @brief Rotate a line or an array of lines
     * @param linearray The line or the array of lines
     * @param rotation The 2x2 rotation matrix
     * @return The rotated line or array or lines
     */
    inline LineArray rotate(LineArray const& linearray, Mat22 const& rotation) noexcept {
        return (rotation * linearray.reshaped(2, linearray.cols()*2)).reshaped(4, linearray.cols());
    }

    /**
    * @brief Rotate a line or an array of lines
    * @param array The line or the array of lines
    * @param rotation The 2x2 rotation matrix
    * @return The rotated line or array or lines
    */
    inline LineArray rotate(LineArray const& linearray, Mat22 const& rotation, Point2 const& rot_point) noexcept {
        const Point2 transl_vector = rot_point - rotation * rot_point;
        Mat23 transform_mat{};
        transform_mat.block<2,2>(0,0) = rotation;
        transform_mat.block<2,1>(0,2) = transl_vector;
        return transform(linearray, transform_mat);
    }

    /**
    * @brief Get the two possible transformation matrices to align the center and direction
     * of the alignment line with the reference line.
    * @param alignment_line The line to align
    * @param ref_line The reference line used to align
    * @return A tuple containing the two possible transformation matrices
    */
    inline std::array<Mat23, 2> align(const Line& alignment_line, const Line& ref_line) noexcept {
        const Point2 tmpl_d{normalize(alignment_line)}, align_d{normalize(ref_line)};

        // Rotation
        const float cos = align_d.x()*tmpl_d.x() + align_d.y()*tmpl_d.y();
        const float sin = align_d.y()*tmpl_d.x() - align_d.x()*tmpl_d.y();
        const Mat22 rot1{{ cos, -sin },{ sin,  cos }};
        const Point2 transl_vector1 = getCenter(ref_line) - getCenter(rotate(alignment_line,rot1));
        const Mat23 transform_mat1{
                { cos, -sin, transl_vector1.x()},
                { sin,  cos, transl_vector1.y()}
        };
        const Mat22 rot2{{ -cos, sin },{ -sin,  -cos }};
        const Point2 transl_vector2 = getCenter(ref_line) - getCenter(rotate(alignment_line,rot2));
        const Mat23 transform_mat2{
                { -cos, sin, transl_vector2.x()},
                { -sin,  -cos, transl_vector2.y()}
        };
        return {transform_mat1, transform_mat2};
    }

    /**
     * @brief Combine a transform and a translation
     * @param transform1 The transform
     * @param transform2 The translation
     * @return The combined transform
     */
    inline Mat23 combine(Mat23 const& transform, Point2 const& translation) noexcept {
        Mat23 result{};
        result.block<2,2>(0,0) = transform.block<2,2>(0,0);
        result.block<2,1>(0,2) = transform.block<2,1>(0,2) + transform.block<2,2>(0,0) * translation;
        return result;
    }

    /**
     * @brief Combine a translation and a transform
     * @param transform1 The transform
     * @param transform2 The translation
     * @return The combined transform
     */
    inline Mat23 combine(Point2 const& translation, Mat23 const& transform) noexcept {
        Mat23 result{};
        result.block<2,2>(0,0) = transform.block<2,2>(0,0);
        result.block<2,1>(0,2) = transform.block<2,1>(0,2) + translation;
        return result;
    }
} //namespace openfdcm

#endif //OPENFDCM_MATH_H