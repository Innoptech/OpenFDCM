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
    depth = 60
    coeff = 50

    max_scene_lines = 5
    max_tmpl_lines = 5

    admissible_score = 1.0
    admissible_transl = 1.0
    admissible_rot_divergence = 1.0
    
def make_rotation(lineAngle: float):
    sin, cos = math.sin(lineAngle), math.cos(lineAngle)
    return np.array([[cos, -sin], [sin,cos]])


class TestDataframe(unittest.TestCase):
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

        self.strategy = openfdcm.BaseStrategy(TestConfig.max_scene_lines, TestConfig.max_tmpl_lines)
        self.metric = openfdcm.PenalisedMetric(1.0)
        self.optimizer = openfdcm.BaseOptimizer()
        self.evaluator = openfdcm.Search(self.strategy, self.metric, self.optimizer)

    def test_evaluation(self):
        perfect_template = openfdcm.Template(self.lines)
        perfect_template.rotate(make_rotation(math.pi))
        perfect_template.translate(np.array(TestConfig.scene_size))

        tmpl_idx, best_proto = openfdcm.get_best_protos(self.evaluator.search(self.dt3, self.scene, [perfect_template]), 1)[0]
        self.assertLess(best_proto.score(), TestConfig.admissible_score)
        self.assertLess(np.linalg.norm(best_proto.state().inplane_trans), TestConfig.admissible_transl)
        rot_divergence = np.matmul(best_proto.state().inplane_rot,np.array(TestConfig.scene_size)) - np.array(TestConfig.scene_size)
        self.assertLess(np.linalg.norm(rot_divergence), TestConfig.admissible_rot_divergence)