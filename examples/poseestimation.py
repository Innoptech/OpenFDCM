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

import cv2
import numpy as np
from pathlib import Path
import openfdcm

def apply_transform(template: np.ndarray, transform: np.ndarray):
    num_lines = template.shape[1]
    transformed_template = np.zeros_like(template)
    for i in range(num_lines):
        point1 = np.dot(transform[:2, :2], template[:2, i]) + transform[:2, 2]
        point2 = np.dot(transform[:2, :2], template[2:, i]) + transform[:2, 2]
        transformed_template[:2, i] = point1
        transformed_template[2:, i] = point2
    return transformed_template

def draw_lines(image: np.ndarray, lines: np.ndarray):
    for i in range(lines.shape[1]):
        pt1 = (int(lines[0, i]), int(lines[1, i]))
        pt2 = (int(lines[2, i]), int(lines[3, i]))
        cv2.line(image, pt1, pt2, (255, 0, 0), 2)
    return image

def display_best_match(scene_image: np.ndarray, best_matches: list[openfdcm.Match], templates: list[np.ndarray]):
    for match in best_matches:
        best_match_template = templates[match.tmpl_idx]
        transformed_template = apply_transform(best_match_template, match.transform)
        result_image = draw_lines(scene_image, transformed_template)
    cv2.imshow("Best Match", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_files_recursive(directory, extension):
    return list(Path(directory).rglob(f"*{extension}"))


if __name__ == "__main__":
    for imagepth in find_files_recursive("./examples/assets", ".jpg"):
        scene_dir: Path = imagepth.parent
        obj_dir: Path = scene_dir.parent
        templates_dir: Path = scene_dir.parent / "templates"

        scene_image = cv2.imread(str(imagepth))
        scene = openfdcm.read(str(scene_dir/"camera_0.scene"))

        templates = []
        for tmpl_path in find_files_recursive(templates_dir, ".tmpl"):
            templates.append(openfdcm.read(str(tmpl_path)))

        # Perform template matching
        max_tmpl_lines, max_scene_lines = 4, 20  # Combinatory search parameters.
        depth = 60              # The [0, pi] discretization.
        scene_padding = 1.0     # Pad the scene images used in the FDCM algorithm, use if best match may appear on image boundaries.
        coeff = 5.0             # A weighting factor to enhance the angular cost vs distance cost in FDCM algorithm.
        num_threads = 4

        threadpool = openfdcm.ThreadPool(num_threads)
        search_strategy = openfdcm.DefaultSearch(max_tmpl_lines, max_scene_lines)
        optimizer_strategy = openfdcm.BatchOptimize(10, threadpool)
        matcher = openfdcm.DefaultMatch()

        featuremap_params = openfdcm.Dt3CpuParameters(depth=depth, dt3Coeff=coeff, padding=scene_padding)
        featuremap = openfdcm.build_cpu_featuremap(scene, featuremap_params, threadpool)
        matches = openfdcm.search(matcher, search_strategy, optimizer_strategy, featuremap, templates, scene)

        if matches:
            best_matches = matches[:5]
            display_best_match(scene_image, best_matches, templates)
