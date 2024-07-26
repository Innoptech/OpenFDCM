
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

#include "pybind11/pybind11.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "openfdcm/matching/matchstrategies/defaultmatch.h"

#include "openfdcm/matching/optimizerstrategies/defaultoptimize.h"
#include "openfdcm/matching/optimizerstrategies/indulgentoptimize.h"

#include "openfdcm/matching/penaltystrategies/defaultpenalty.h"
#include "openfdcm/matching/penaltystrategies/exponentialpenalty.h"

#include "openfdcm/matching/searchstrategies/defaultsearch.h"
#include "openfdcm/matching/searchstrategies/concentricrange.h"

//-------------------------------------------------------------------------------
// PYTHON BINDINGS
//-------------------------------------------------------------------------------
namespace py = pybind11;
using namespace pybind11::literals;
using namespace openfdcm;
using namespace openfdcm::matching;

void matching(py::module_ &m) {
    // ---------------------------------------------------------
    // 1D optimize strategies
    // ---------------------------------------------------------
    py::class_<OptimizeStrategy>(m, "OptimizeStrategy")
            .def(py::init<const DefaultOptimize&>())
            .def(py::init<const IndulgentOptimize&>())
            .def("__repr__", [](const OptimizeStrategy &a) {
                return "<OptimizeStrategy>";
            });

    py::class_<DefaultOptimize>(m, "DefaultOptimize")
            .def(py::init<>())
            .def("__repr__", [](const DefaultOptimize &a) {
                return "<DefaultOptimize>";
            });

    py::class_<IndulgentOptimize>(m, "IndulgentOptimize")
            .def(py::init<uint32_t>())
            .def("get_number_of_passthroughs", &IndulgentOptimize::getNumberOfPassthroughs)
            .def("__repr__", [](const IndulgentOptimize &a) {
                return "<IndulgentOptimize: number of passthroughs=" + std::to_string(a.getNumberOfPassthroughs()) + ">";
            });

    py::implicitly_convertible<DefaultOptimize, OptimizeStrategy>();
    py::implicitly_convertible<IndulgentOptimize, OptimizeStrategy>();

    // ---------------------------------------------------------
    // Penalty strategies
    // ---------------------------------------------------------
    py::class_<PenaltyStrategy>(m, "PenaltyStrategy")
            .def(py::init<const DefaultPenalty&>())
            .def(py::init<const ExponentialPenalty&>())
            .def("__repr__", [](const PenaltyStrategy &a) {
                return "<PenaltyStrategy>";
            });

    py::class_<DefaultPenalty>(m, "DefaultPenalty")
            .def(py::init<>())
            .def("__repr__", [](const DefaultPenalty &a) {
                return "<DefaultPenalty>";
            });

    py::class_<ExponentialPenalty>(m, "ExponentialPenalty")
            .def(py::init<float>())
            .def("get_tau", &ExponentialPenalty::getTau)
            .def("__repr__", [](const ExponentialPenalty &a) {
                return "<ExponentialPenalty: tau=" + std::to_string(a.getTau()) + ">";
            });

    py::implicitly_convertible<DefaultPenalty, PenaltyStrategy>();
    py::implicitly_convertible<ExponentialPenalty, PenaltyStrategy>();

    // ---------------------------------------------------------
    // Search strategies
    // ---------------------------------------------------------
    py::class_<SearchStrategy>(m, "SearchStrategy")
            .def(py::init<const DefaultSearch&>())
            .def(py::init<const ConcentricRangeStrategy&>())
            .def("__repr__", [](const SearchStrategy &a) {
                return "<SearchStrategy>";
            });

    py::class_<DefaultSearch>(m, "DefaultSearch")
            .def(py::init<size_t const, size_t const>())
            .def("get_max_tmpl_lines", &DefaultSearch::getMaxTmplLines)
            .def("get_max_scene_lines", &DefaultSearch::getMaxSceneLines)
            .def("__repr__", [](const DefaultSearch &a) {
                return "<DefaultSearch: max tmpl lines=" + std::to_string(a.getMaxTmplLines()) +
                       ", max scene lines=" + std::to_string(a.getMaxSceneLines()) + ">";
            });

    py::class_<ConcentricRangeStrategy>(m, "ConcentricRangeStrategy")
            .def(py::init<size_t const, size_t const, core::Point2, float const, float const>())
            .def("get_max_tmpl_lines", &ConcentricRangeStrategy::getMaxTmplLines)
            .def("get_max_scene_lines", &ConcentricRangeStrategy::getMaxSceneLines)
            .def("get_center_position", &ConcentricRangeStrategy::getCenterPosition)
            .def("get_low_radius_boundary", &ConcentricRangeStrategy::getLowBoundary)
            .def("get_high_radius_boundary", &ConcentricRangeStrategy::getHighBoundary)
            .def("__repr__", [](const ConcentricRangeStrategy &a) {
                return "<ConcentricRangeStrategy: get_max_tmpl_lines=" + std::to_string(a.getMaxTmplLines()) +
                       ", max scene lines=" + std::to_string(a.getMaxSceneLines()) +
                       ", center position=(" + std::to_string(a.getCenterPosition().x()) + ", " + std::to_string(a.getCenterPosition().y()) + ")" +
                       ", low radius boundary=" + std::to_string(a.getLowBoundary()) +
                       ", high radius boundary=" + std::to_string(a.getHighBoundary()) + ">";
            });

    py::implicitly_convertible<DefaultSearch, SearchStrategy>();
    py::implicitly_convertible<ConcentricRangeStrategy, SearchStrategy>();

    // ---------------------------------------------------------
    // Match strategies
    // ---------------------------------------------------------
    py::class_<MatchStrategy>(m, "MatchStrategy")
            .def(py::init<const DefaultMatch&>())
            .def("__repr__", [](const MatchStrategy &a) {
                return "<MatchStrategy>";
            });

    py::class_<DefaultMatch>(m, "DefaultMatch")
            .def(py::init<size_t, float, float, float>())
            .def("get_depth", &DefaultMatch::getDepth)
            .def("get_coeff", &DefaultMatch::getCoeff)
            .def("get_scene_ratio", &DefaultMatch::getSceneRatio)
            .def("get_scene_padding", &DefaultMatch::getScenePadding)
            .def("__repr__", [](const DefaultMatch &a) {
                return "<DefaultMatch: depth=" + std::to_string(a.getDepth()) +
                       ", coeff=" + std::to_string(a.getCoeff()) +
                       ", scene ratio=" + std::to_string(a.getSceneRatio()) +
                       ", scene padding=" + std::to_string(a.getScenePadding()) + ">";
            });

    // Register implicit conversion using pybind11 type casters
    py::implicitly_convertible<DefaultMatch, MatchStrategy>();

    py::class_<Match>(m, "Match")
            .def(py::init<>())
            .def_readwrite("tmpl_idx", &Match::tmplIdx)
            .def_readwrite("score", &Match::score)
            .def_readwrite("transform", &Match::transform)
            .def("__repr__", [](const Match &a) {
                std::ostringstream oss;
                oss << "<Match tmplIdx=" << a.tmplIdx
                    << ", score=" << a.score
                    << ", transform=\n" << a.transform << ">";
                return oss.str();
            });

    m.def("search", [](const MatchStrategy &matcher,
                       const SearchStrategy &searcher,
                       const OptimizeStrategy &optimizer,
                       std::vector<Eigen::Matrix<float, 4, -1>> const& templates,
                       Eigen::Matrix<float, 4, -1> const& scene)
          {
              return matching::search(matcher, searcher, optimizer, templates, scene);
          },
          "matcher"_a, "searcher"_a, "optimizer"_a, "templates"_a, "scene"_a);
}