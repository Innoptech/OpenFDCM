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
import cv2
import os
import pytest
from pathlib import Path

@pytest.fixture(scope="module")
def script_loc(request):
    '''
    Return the directory of the currently running test script
    From https://stackoverflow.com/a/44935451/10631984
    '''
    return Path(request.fspath).parent

class TestConfig:
    assets_dir = Path(__file__).parent.parent.parent.joinpath("assets")
    flash_frames = {
        0: "test_template/capture_0.exr",
        math.pi/4: "test_template/capture_45.exr",
        math.pi/2: "test_template/capture_90.exr",
        math.pi*3/4: "test_template/capture_135.exr",
        math.pi: "test_template/capture_180.exr",
        math.pi*5/4: "test_template/capture_225.exr",
        math.pi*6/4: "test_template/capture_270.exr",
        math.pi*7/4: "test_template/capture_315.exr"
    }
    no_flash_frame = "test_template/capture_no.exr"
    tmpl_serial_ref = "test_template_python.encoded"

def read(path):
    img_name, img_ext = os.path.splitext(path)
    assert(img_ext == ".exr")
    frame = cv2.imread(os.path.join(TestConfig.assets_dir, path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

class TestMultiflash(unittest.TestCase):
    def setUp(self):
        self.mfcontainers = []
        no_flash_img = read(TestConfig.no_flash_frame)

        for flash_angle, filename in TestConfig.flash_frames.items():
            frame = read(filename)
            frame -= no_flash_img
            frame = cv2.max(frame, 0)
            frame = cv2.min(frame, 1)
            frame = (frame * 255).astype(np.uint8)
            self.mfcontainers.append(openfdcm.FrameContainer(frame, flash_angle))

    def test_multiflash(self):
        lines = openfdcm.compute_multiflash(self.mfcontainers)
        ref_lines = openfdcm.read_serialized_lines(os.path.join(TestConfig.assets_dir, TestConfig.tmpl_serial_ref))
        self.assertListEqual(lines, ref_lines)