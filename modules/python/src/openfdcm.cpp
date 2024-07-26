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
#include "openfdcm/core/version.h"

//-------------------------------------------------------------------------------
// PYTHON BINDINGS
//-------------------------------------------------------------------------------
namespace py = pybind11;
using namespace pybind11::literals;

void core(py::module_ &);
void matching(py::module_ &);
void example(py::module_ &);

PYBIND11_MODULE(openfdcm, m) {
    m.doc() = "A modern C++ open implementation of Fast Directional Chamfer Matching with few improvements.\n";
    core(m);
    matching(m);

    m.attr("__version__") = OPENFDCM_PROJECT_VER;
}