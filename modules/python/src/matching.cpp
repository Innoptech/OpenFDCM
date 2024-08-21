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

#include "openfdcm/matching/featuremaps/dt3cpu.h"

#include "openfdcm/matching/matchstrategies/defaultmatch.h"

#include "openfdcm/matching/optimizestrategies/defaultoptimize.h"
#include "openfdcm/matching/optimizestrategies/indulgentoptimize.h"
#include "openfdcm/matching/optimizestrategies/batchoptimize.h"

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
    // Featuremaps
    // ---------------------------------------------------------
    py::class_<FeatureMap>(m, "FeatureMap")
            .def(py::init<const Dt3Cpu&>())
            .def("__repr__", [](const FeatureMap &a) {
                return "<FeatureMap>";
            });

    py::class_<Dt3Cpu>(m, "Dt3Cpu")
            .def(py::init<detail::Dt3CpuMap<float>, core::Point2, core::Size>())
            .def("get_scene_translation", &Dt3Cpu::getSceneTranslation)
            .def("get_feature_size", &Dt3Cpu::getFeatureSize)
            .def("get_dt3_map", &Dt3Cpu::getDt3Map)
            .def("__repr__", [](const Dt3Cpu &a) {
                std::string repr = "<Dt3Cpu: scene translation=(" +
                                   std::to_string(a.getSceneTranslation().x()) + ", " +
                                   std::to_string(a.getSceneTranslation().y()) + "), " +
                                   "feature size=(" + std::to_string(a.getFeatureSize().x()) + ", " +
                                   std::to_string(a.getFeatureSize().y()) + ")>";
                return repr;
            });

    py::class_<BS::thread_pool, std::shared_ptr<BS::thread_pool>>(m, "ThreadPool")
            // Use shared pointers for the constructors
            .def(py::init<>())  // Default constructor
            .def(py::init<BS::concurrency_t>(), "num_threads"_a)  // Constructor with num_threads
            .def("get_tasks_queued", &BS::thread_pool::get_tasks_queued, "Get the number of tasks queued.")
            .def("get_tasks_running", &BS::thread_pool::get_tasks_running, "Get the number of tasks currently running.")
            .def("get_tasks_total", &BS::thread_pool::get_tasks_total, "Get the total number of unfinished tasks.")
            .def("get_thread_count", &BS::thread_pool::get_thread_count, "Get the number of threads in the pool.")
            .def("get_thread_ids", &BS::thread_pool::get_thread_ids, "Get the thread IDs for all threads in the pool.")
            .def("purge", &BS::thread_pool::purge, "Purge all tasks waiting in the queue.")
                    // __repr__ method
            .def("__repr__", [](const BS::thread_pool &tp) {
                return "<ThreadPool: threads=" + std::to_string(tp.get_thread_count()) +
                       ", tasks queued=" + std::to_string(tp.get_tasks_queued()) +
                       ", tasks running=" + std::to_string(tp.get_tasks_running()) + ">";
            });

    py::class_<Dt3CpuParameters>(m, "Dt3CpuParameters")
            .def(py::init<size_t, float, float>(), "depth"_a=30, "dt3Coeff"_a=5.f, "padding"_a=2.2f)
            .def_readwrite("depth", &Dt3CpuParameters::depth)
            .def_readwrite("dt3_coeff", &Dt3CpuParameters::dt3Coeff)
            .def_readwrite("padding", &Dt3CpuParameters::padding)
            .def("__repr__", [](const Dt3CpuParameters &p) {
                return "<Dt3CpuParameters: depth=" + std::to_string(p.depth) +
                       ", dt3_coeff=" + std::to_string(p.dt3Coeff) +
                       ", padding=" + std::to_string(p.padding) + ">";
            });

    m.def("build_cpu_featuremap", &buildCpuFeaturemap,
          "scene"_a, "params"_a=Dt3CpuParameters{}, "pool"_a=std::make_shared<BS::thread_pool>(),
          "Builds the Dt3Cpu featuremap given a scene, parameters, and an optional thread pool."
    );

    py::implicitly_convertible<Dt3Cpu, FeatureMap>();

    // ---------------------------------------------------------
    // 1D optimize strategies
    // ---------------------------------------------------------
    py::class_<OptimizeStrategy>(m, "OptimizeStrategy")
            .def(py::init<const DefaultOptimize&>())
            .def(py::init<const IndulgentOptimize&>())
            .def(py::init<const BatchOptimize&>())
            .def("__repr__", [](const OptimizeStrategy &a) {
                return "<OptimizeStrategy>";
            });

    py::class_<DefaultOptimize>(m, "DefaultOptimize")
            .def(py::init<std::shared_ptr<BS::thread_pool>>(), "pool"_a = std::make_shared<BS::thread_pool>(),
                 "Constructor that accepts a thread pool with an optional default pool.")
            .def(py::init<BS::concurrency_t>(), "num_threads"_a, "Constructor that accepts a number of threads.")
            .def("get_pool", &DefaultOptimize::getPool, "Returns the thread pool as a shared pointer.")
            .def("__repr__", [](const DefaultOptimize &a) {
                return "<DefaultOptimize>";
            });

    py::class_<IndulgentOptimize>(m, "IndulgentOptimize")
            .def(py::init<uint32_t, std::shared_ptr<BS::thread_pool>>(),
                 "indulgent_number_of_passthroughs"_a,
                 "pool"_a = std::make_shared<BS::thread_pool>(),
                 "Constructor that accepts the number of passthroughs and an optional thread pool.")
            .def(py::init<uint32_t, BS::concurrency_t>(),
                 "indulgent_number_of_passthroughs"_a,
                 "num_threads"_a, "Constructor that accepts a number of threads.")
            .def("get_number_of_passthroughs", &IndulgentOptimize::getNumberOfPassthroughs,
                 "Get the number of indulgent passthroughs.")
            .def("get_pool", &IndulgentOptimize::getPool, "Returns the thread pool as a shared pointer.")
            .def("__repr__", [](const IndulgentOptimize &a) {
                return "<IndulgentOptimize: number_of_passthroughs=" +
                       std::to_string(a.getNumberOfPassthroughs()) + ">";
            });

    py::class_<BatchOptimize>(m, "BatchOptimize")
            .def(py::init<size_t, std::shared_ptr<BS::thread_pool>>(), "batch_size"_a, "pool"_a = std::make_shared<BS::thread_pool>(),
                 "Constructor that accepts a thread pool with an optional default pool.")
            .def(py::init<size_t, BS::concurrency_t>(), "batch_size"_a, "num_threads"_a, "Constructor that accepts a number of threads.")
            .def("get_batch_size", &BatchOptimize::getBatchSize, "Returns the batch size.")
            .def("get_pool", &BatchOptimize::getPool, "Returns the thread pool as a shared pointer.")
            .def("__repr__", [](const BatchOptimize &a) {
                return "<BatchOptimize>";
            });

    py::implicitly_convertible<DefaultOptimize, OptimizeStrategy>();
    py::implicitly_convertible<IndulgentOptimize, OptimizeStrategy>();
    py::implicitly_convertible<BatchOptimize, OptimizeStrategy>();

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
            .def(py::init<>(), "Penalize the matches given the template length (1/(n)")
            .def("__repr__", [](const DefaultPenalty &a) {
                return "<DefaultPenalty>";
            });

    py::class_<ExponentialPenalty>(m, "ExponentialPenalty")
            .def(py::init<float>(), "tau"_a, "Penalize the matches given the template length (1/(n^tau)")
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
            .def(py::init<size_t const, size_t const>(), "max_tmpl_lines"_a, "max_scene_lines"_a)
            .def("get_max_tmpl_lines", &DefaultSearch::getMaxTmplLines)
            .def("get_max_scene_lines", &DefaultSearch::getMaxSceneLines)
            .def("__repr__", [](const DefaultSearch &a) {
                return "<DefaultSearch: max tmpl lines=" + std::to_string(a.getMaxTmplLines()) +
                       ", max scene lines=" + std::to_string(a.getMaxSceneLines()) + ">";
            });

    py::class_<ConcentricRangeStrategy>(m, "ConcentricRangeStrategy")
            .def(py::init<size_t const, size_t const, core::Point2, float const, float const>(),
                    "max_tmpl_lines"_a, "max_scene_lines"_a, "center_position"_a, "low_boundary"_a, "high_boundary"_a)
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
            .def(py::init<>())
            .def("__repr__", [](const DefaultMatch &a) {
                return "<DefaultMatch>";
            });

    // Register implicit conversion using pybind11 type casters
    py::implicitly_convertible<DefaultMatch, MatchStrategy>();

    py::class_<Match>(m, "Match")
            .def(py::init<int, float, core::Mat23>())
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
                       const FeatureMap &featuremap,
                       const std::vector<core::LineArray>& templates,
                       const core::LineArray& scene)
          {
              return matching::search(matcher, searcher, optimizer, featuremap, templates, scene);
          },
          "matcher"_a, "searcher"_a, "optimizer"_a, "featuremap"_a, "templates"_a, "scene"_a,
          "Search for optimal matches between the templates and the scene");

    m.def("penalize", [](const PenaltyStrategy &penalty, const std::vector<Match> &matches,
                         const std::vector<float> &templatelengths)
          {
              return matching::penalize(penalty, matches, templatelengths);
          },
          "penalty"_a, "matches"_a, "templatelengths"_a,
          "Apply a given score penalty on a vector of matches");

    m.def("get_template_lengths", &core::getTemplateLengths, "templates"_a,
          "Get the lengths of templates represented by line arrays");

    m.def("sort_matches", [](std::vector<Match> &matches)
    {
        std::sort(std::begin(matches), std::end(matches),
                  [](const Match& lhs, const Match& rhs) { return lhs.score < rhs.score; });
        return matches;
    }, "matches"_a, "Sort the matches by score, with the best score (lowest) first.");



}