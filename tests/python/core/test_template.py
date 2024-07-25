'''
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
'''

import unittest
import math
from copy import copy
import numpy as np
import openfdcm

class TestConfig:
    ref_linecount = 60
    img_tmpl_size = [300, 200]


def make_rotation(lineAngle: float):
    sin, cos = math.sin(lineAngle), math.cos(lineAngle)
    return np.array([[cos, -sin], [sin,cos]])

class TestTemplate(unittest.TestCase):
    def setUp(self):
        lines = []
        self.total_length = 0
        min_dim = min(TestConfig.img_tmpl_size[0], TestConfig.img_tmpl_size[1]) - 10
        for i in range(TestConfig.ref_linecount//2):
            lineAngle = i/(TestConfig.ref_linecount/2)*math.pi/2
            line1 = openfdcm.Line([0,0], [min_dim, 0]).rotate(make_rotation(lineAngle)).translate(([5.0,5.0]))
            self.total_length += min_dim
            lines.append(line1)
        
        for i in range(TestConfig.ref_linecount//2):
            lineAngle = i/(float(TestConfig.ref_linecount)/2)*math.pi/2/2-math.pi/2/2;
            line2 = openfdcm.Line([0,0], [min_dim, 0]).rotate(make_rotation(lineAngle)).translate(([5.0,TestConfig.img_tmpl_size[1]-5.0]))
            self.total_length += min_dim
            lines.append(line2)
        self.tmpl = openfdcm.Template(lines)

    def test_tmpl_linecount(self):
            self.assertEqual(self.tmpl.line_count(), TestConfig.ref_linecount)

    def test_tmpl_length(self):
        self.assertLess(abs(self.tmpl.length() - self.total_length), TestConfig.ref_linecount)

    def test_tmpl_op(self):
        try:
            pt = np.array([1,1])
            rot = np.eye(2)
            tmpl = copy(self.tmpl)
            tmpl *= rot
            tmpl += pt
            tmpl = tmpl + pt
            tmpl -= pt
            tmpl = tmpl - pt
        except Exception:
            self.fail("Template operations raised an Exception unexpectedly!")

    def test_template_state_constructor(self):
        try:
           state = openfdcm.TemplateState(np.eye(2), np.zeros(2))
        except Exception:
            self.fail("Template operations raised an Exception unexpectedly!")

if __name__ == '__main__':
    unittest.main()
