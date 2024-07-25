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
#include <pybind11/eigen.h>

#include "openfdcm/core/math.h"
#include "openfdcm/core/dt3.h"
#include "openfdcm/core/serialization.h"

//-------------------------------------------------------------------------------
// PYTHON BINDINGS
//-------------------------------------------------------------------------------
namespace py = pybind11;
using namespace pybind11::literals;
using namespace openfdcm;

void core(py::module_ &m)
{

    m.def("draw_lines", [](py::array_t<PYIMG>& pyimg, LineArray const& linearray, const float& color){
        cv::Mat mat = pyimg_to_cv(pyimg);
        openfdcm::drawLines(mat, linearray, color);
    }, "pyimg"_a, "linearray"_a, "color"_a=0, "Draw one or multiple lines on an image");

    m.def("serialize_lines", [](const std::vector<openfdcm::Line>& lines) -> py::bytes {
              std::stringstream ss;
              std::ostreambuf_iterator<char> it_out{ss};
              openfdcm::serializeLines(lines, it_out);
              return py::bytes(ss.str());
          }, "lines"_a, "Serialize a list of lines");

    m.def("deserialize_lines", [](const std::string& serial_data) -> std::vector<openfdcm::Line>{
        std::stringstream ss(serial_data);
        std::istreambuf_iterator<char> it_in{ss}, end;
        return openfdcm::deserializeLines(it_in, end);
        }, "serial_data"_a,"Deserialize a list of lines");

    m.def("write", &openfdcm::write, "filepath"_a, "lines"_a,
          "Write serialized lines in file");

    m.def("read", &openfdcm::read, "filepath"_a, "Read serialized lines from file");

    m.def("is_evaluable", &openfdcm::is_evaluable, "size"_a, "line"_a,
         "Verify if line is evaluable within the bounds of DT3");

    m.def("eval", py::overload_cast<const openfdcm::Dt3&, const openfdcm::BaseTemplate&>
            (&openfdcm::eval), "dt3"_a, "template"_a, "Evaluate a template on DT3");

    m.def("aling", &openfdcm::aling, "template"_a, "line_idx"_a, "align_line"_a,
          "A function to align a template on a line");

    m.def("fastRotate", &openfdcm::fastRotate, "template"_a, "center_rot"_a,
          "A function to turn a template 180 degrees around the center of one of its lines");
}

