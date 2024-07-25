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
    pass


class TestLine(unittest.TestCase):
    def setUp(self):
        self.pt1 = np.array([0,0]).astype(int)
        self.pt2 = 10*np.array([1,1]).astype(int)
        self.line = openfdcm.Line(self.pt1, self.pt2)

    def test_line_copy(self):
        try:
            b = copy(self.line)
        except Exception:
            self.fail("Line copy raised an Exception unexpectedly!")

    def test_line_center(self):
        self.assertLess(np.sum(np.abs(self.line.center() - self.pt2/2)), 1)

    def test_line_angle(self):
        self.assertLess(abs(self.line.lineAngle() - math.atan(self.pt2[1]/self.pt2[0])), 1e-5)

    def test_line_length(self):
        self.assertLess(abs(self.line.length()-np.linalg.norm(self.pt2)), 1)

    def test_line_rotate(self):
        try:
            lineAngle = -self.line.lineAngle()
            rot = np.array([[math.cos(lineAngle), -math.sin(lineAngle)], [math.sin(lineAngle), math.cos(lineAngle)]])
            b = copy(self.line)
            center = b.center()
            b -= center
            b.rotate(rot)
            b += center
        except Exception:
            self.fail("Line rotation raised an Exception unexpectedly!")

        self.assertLess(abs(b.length()-np.linalg.norm(self.pt2)), 1)
        self.assertEqual(b.lineAngle(), 0)
        self.assertLess(np.linalg.norm(b.center() - self.pt2//2), 1)

    def test_line_op(self):
        try:
            rot = np.eye(2)
            line = copy(self.line)
            line *= rot
            line += self.pt2
            line = line + self.pt2
            line -= self.pt2
            line = line - self.pt2
        except Exception:
            self.fail("Line operations raised an Exception unexpectedly!")

if __name__ == '__main__':
    unittest.main()