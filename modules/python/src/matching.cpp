
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
#include <pybind11/operators.h>

#include "openfdcm/matching/matchstrategies/defaultmatch.h"
#include "openfdcm/matching/searchstrategies/defaultsearch.h"
#include "openfdcm/matching/optimizerstrategies/defaultoptimize.h"

//-------------------------------------------------------------------------------
// PYTHON BINDINGS
//-------------------------------------------------------------------------------
namespace py = pybind11;
using namespace pybind11::literals;

using namespace openfdcm::matching;

void matching(py::module_ &m) {
    //-------------------------------------------------------------------------------------------------------
    // Dataframe
    //-------------------------------------------------------------------------------------------------------
    // Prototype class
    py::class_<openfdcm::dataframe::Prototype>(m, "Prototype")
            .def(py::init<const int&, const int&>(), "template_line_idx"_a, "scene_line_idx"_a)
            .def("get_template_line_idx", &openfdcm::dataframe::Prototype::get_template_line_idx,
                 "Get the template line index")
            .def("get_scene_line_idx", &openfdcm::dataframe::Prototype::get_scene_line_idx, "Get the scene line index")
            .def("__repr__",
                 [](const openfdcm::dataframe::Prototype &proto) {
                     std::string r("Prototype: (");
                     r += "template line idx: " + std::to_string(proto.get_template_line_idx()) +
                          ", scene line idx: " + std::to_string(proto.get_scene_line_idx()) + ")\n";
                     return r;
                 })
            ;

    // TemplatePrototype class
    py::class_<openfdcm::dataframe::TemplatePrototype>(m, "TemplatePrototype")
            .def(py::init<const int&, const std::vector<openfdcm::dataframe::Prototype>&>(),
                    "template_idx"_a, "prototypes"_a)
            .def("get_template_idx", &openfdcm::dataframe::TemplatePrototype::get_template_idx,
                 "Get the template index")
            .def("get_prototypes", &openfdcm::dataframe::TemplatePrototype::get_prototypes,
                 "Get the all the prototypes relative to this template")
            .def("__repr__",
                 [](const openfdcm::dataframe::TemplatePrototype &proto) {
                     std::string r("TemplatePrototype: (");
                     r += "template idx: " + std::to_string(proto.get_template_idx()) +
                          ", number of prototypes: " + std::to_string(proto.get_prototypes().size()) + ")\n";
                     return r;
                 })
            ;

    //PrototypeResult class
    py::class_<openfdcm::dataframe::PrototypeResult>(m, "PrototypeResult")
            .def(py::init<openfdcm::TemplateState, const double&>(), "state"_a, "score"_a)
            .def("state", &openfdcm::dataframe::PrototypeResult::state,
                 "Get the prototype optimal state")
            .def("score", &openfdcm::dataframe::PrototypeResult::score,
                 "Get the prototype optimal evaluation score")
            .def("__repr__",
                 [](const openfdcm::dataframe::PrototypeResult &result) {
                     std::string r("PrototypeResult: (");
                     r += "score: " + std::to_string(result.score()) + ")\n";
                     return r;
                 })
            ;

    //TemplatePrototypeResult class
    using tmpl_res = openfdcm::dataframe::TemplatePrototypeResult;
    py::class_<tmpl_res>(m, "TemplatePrototypeResult")
            .def(py::init<const int&, std::vector<openfdcm::dataframe::PrototypeResult>>(),
                    "template_idx"_a, "prototype_result"_a)
            .def("get_template_idx", &tmpl_res::get_template_idx,
                 "Get the template index")
            .def("get_prototype_result", py::overload_cast<>(&tmpl_res::get_prototype_result, py::const_),
                 "Get the all the PrototypeResults for this given template")
            .def("__repr__",
                 [](const openfdcm::dataframe::TemplatePrototypeResult &result) {
                     std::string r("TemplatePrototypeResult: (");
                     r += "template idx: " + std::to_string(result.get_template_idx()) +
                          ", number of prototypes result: " +
                          std::to_string(result.get_prototype_result().size()) + ")\n";
                     return r;
                 })
            ;

    //-------------------------------------------------------------------------------------------------------
    // Evaluation components
    //-------------------------------------------------------------------------------------------------------
    // Strategy class
    py::class_<openfdcm::BaseStrategy>(m, "BaseStrategy")
            .def(py::init<const int& , const int&>(), "max_scene_lines"_a, "max_template_lines"_a)
            .def("establish_strategy", &openfdcm::BaseStrategy::establish_strategy,
                 "scene"_a, "templateDataset"_a,
                 "Establish a strategy (search sequence)")
            .def("get_max_scene_lines", &openfdcm::BaseStrategy::get_max_scene_lines,
                 "Get max scene lines number")
            .def("get_max_template_lines", &openfdcm::BaseStrategy::get_max_template_lines,
                 "Get max template lines number")
            .def("__repr__",
                 [](const openfdcm::BaseStrategy &strategy) {
                     std::string r("BaseStrategy: (");
                     r += "max scene lines: " + std::to_string(strategy.get_max_scene_lines()) +
                          ", max template lines: " + std::to_string(strategy.get_max_template_lines()) + ")\n";
                     return r;
                 })
            ;

    // RadiusRangeStrategy class (perspective-dependant strategy)
    py::class_<openfdcm::RadiusRangeStrategy, openfdcm::BaseStrategy>(m, "RadiusRangeStrategy")
            .def(py::init<const int&, const int&, const int&, const int& >(),
                    "max_scene_lines"_a, "max_template_lines"_a, "_lower_bound"_a, "_upper_bound"_a)
            .def("establish_strategy", &openfdcm::RadiusRangeStrategy::establish_strategy,
                 "scene"_a, "templateDataset"_a,
                 "Establish a strategy (search sequence)")
            .def("__repr__",
                 [](const openfdcm::BaseStrategy &strategy) {
                     std::string r("RadiusRangeStrategy: (");
                     r += "max scene lines: " + std::to_string(strategy.get_max_scene_lines()) +
                          ", max template lines: " + std::to_string(strategy.get_max_template_lines()) + ")\n";
                     return r;
                 })
            ;

    // BaseOptimizer class
    py::class_<openfdcm::BaseOptimizer>(m, "BaseOptimizer")
            .def(py::init<>())
            .def("optimize", &openfdcm::BaseOptimizer::optimize,
                 "Dt3"_a, "template"_a, "template_line_idx"_a, "scene_line"_a, "metric"_a
                 "Minimize Dt3 score according to template line alignment over scene_line")
            .def("__repr__",
                 [](const openfdcm::BaseOptimizer &optimizer) {
                     (void) optimizer;
                     std::string r("BaseOptimizer implementation\n");
                     return r;
                 })
            ;

    // BaseMetric class
    py::class_<openfdcm::BaseMetric>(m, "BaseMetric")
            .def(py::init<>())
            .def("eval", &openfdcm::BaseMetric::eval, "Eval a metric given dt3 score and template length")
            .def("__repr__",
                 [](const openfdcm::BaseMetric &metric) {
                     (void) metric;
                     std::string r("BaseMetric implementation)\n");
                     return r;
                 })
            ;

    // PenalisedMetric class
    py::class_<openfdcm::PenalisedMetric, openfdcm::BaseMetric>(m, "PenalisedMetric")
            .def(py::init<const double&>(), "penalty"_a)
            .def("get_penalty", &openfdcm::PenalisedMetric::get_penalty, "Get penalty coefficient")
            .def("__repr__",
                 [](const openfdcm::PenalisedMetric &metric) {
                     std::string r("PenalisedMetric: (");
                     r += "penalty: " + std::to_string(metric.get_penalty()) + ")\n";
                     return r;
                 })
            ;

    // Search class
    py::class_<openfdcm::Search>(m, "Search")
            .def(py::init<openfdcm::BaseStrategy*, openfdcm::BaseMetric*, openfdcm::BaseOptimizer*>(),
                    "strategy"_a, "metric"_a, "optimizer"_a)
            .def("search", &openfdcm::Search::search, "dt3"_a, "scene"_a, "list of templates"_a,
                 "Evaluate a template dataset over a scene")
            .def("__repr__",
                 [](const openfdcm::Search &evaluator) {
                     (void) evaluator;
                     std::string r("Search implementation\n");
                     return r;
                 })
            ;

    //-------------------------------------------------------------------------------------------------------
    // Functions
    //-------------------------------------------------------------------------------------------------------
    m.def("get_best_protos", &openfdcm::dataframe::get_best_protos,
          "_template_results"_a, "n_best"_a=1, "Finds the N best prototypes.");

    m.def("eval", py::overload_cast<const openfdcm::Dt3&, const openfdcm::BaseTemplate&, const openfdcm::BaseMetric&>
            (&openfdcm::eval), "dt3"_a, "template"_a, "metric"_a, "Evaluate a template on DT3 given a metric");
}