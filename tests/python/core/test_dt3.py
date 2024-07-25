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
    scene_size = [300, 200]
    depth = 60
    coeff = 50
    admissible_noise = 0.7  # percent error
    
def make_rotation(lineAngle: float):
    sin, cos = math.sin(lineAngle), math.cos(lineAngle)
    return np.array([[cos, -sin], [sin,cos]])


class TestDt3(unittest.TestCase):
    def setUp(self):
        lines = []
        min_dim = min(TestConfig.scene_size[0], TestConfig.scene_size[1]) - 10
        for i in range(TestConfig.ref_linecount//2):
            lineAngle = i/(TestConfig.ref_linecount/2)*math.pi/2
            line1 = openfdcm.Line([0,0], [min_dim, 0]).rotate(make_rotation(lineAngle)).translate(([5.0,5.0]))
            lines.append(line1)
        
        for i in range(TestConfig.ref_linecount//2):
            lineAngle = i/(float(TestConfig.ref_linecount)/2)*math.pi/2/2-math.pi/2/2;
            line2 = openfdcm.Line([0,0], [min_dim, 0]).rotate(make_rotation(lineAngle)).translate(([5.0,TestConfig.scene_size[1]-5.0]))
            lines.append(line2)
        self.lines = lines
        self.scene = openfdcm.Scene(lines, TestConfig.scene_size)
        self.dt3 = openfdcm.Dt3(TestConfig.scene_size, TestConfig.depth, TestConfig.coeff, lines)

    def test_dt3_depth(self):
        self.assertEqual(self.dt3.depth(), TestConfig.depth)

    def test_dt3_size(self):
        self.assertListEqual(self.dt3.size().tolist(), self.scene.size().tolist())

    def test_dt3_coeff(self):
        self.assertEqual(self.dt3.coeff(), TestConfig.coeff)

    def test_dt3_is_evaluable(self):
        max_x, max_y = TestConfig.scene_size[0]-1, TestConfig.scene_size[1]-1
        self.assertTrue(openfdcm.is_evaluable(TestConfig.scene_size, openfdcm.Line([0,0], [max_x,max_y])))
        self.assertTrue(openfdcm.is_evaluable(TestConfig.scene_size, openfdcm.Line([0,0], [0,0])))

        self.assertFalse(openfdcm.is_evaluable(TestConfig.scene_size, openfdcm.Line([-1,0], [max_x,max_y])))
        self.assertFalse(openfdcm.is_evaluable(TestConfig.scene_size, openfdcm.Line([0,-1], [max_x,max_y])))
        self.assertFalse(openfdcm.is_evaluable(TestConfig.scene_size, openfdcm.Line([0,0], [max_x+1,max_y+1])))

    def test_validate_eval(self):
        perfect_template = openfdcm.Template(self.lines)
        is_eval, score = openfdcm.eval(self.dt3, perfect_template)
        self.assertTrue(is_eval)
        self.assertLess(score/perfect_template.length(), TestConfig.admissible_noise)

if __name__ == '__main__':
    unittest.main()