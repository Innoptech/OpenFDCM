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
import numpy as np
import openfdcm

class TestConfig:
    ref_linecount = 60
    scene_size = [300, 200]

def make_rotation(lineAngle: float):
    sin, cos = math.sin(lineAngle), math.cos(lineAngle)
    return np.array([[cos, -sin], [sin,cos]])


class TestSerialize(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.total_length = 0
        min_dim = min(TestConfig.scene_size[0], TestConfig.scene_size[1]) - 10
        for i in range(TestConfig.ref_linecount//2):
            lineAngle = i/(TestConfig.ref_linecount/2)*math.pi/2
            line1 = openfdcm.Line([0,0], [min_dim, 0]).rotate(make_rotation(lineAngle)).translate(([5.0,5.0]))
            self.total_length += min_dim
            self.lines.append(line1)

    def test_restitution(self):
        serialized_lines = openfdcm.serialize_lines(self.lines)
        deserialized_lines = openfdcm.deserialize_lines(serialized_lines)
        self.assertListEqual(deserialized_lines, self.lines)

if __name__ == '__main__':
    unittest.main()