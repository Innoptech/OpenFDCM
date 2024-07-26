import pytest
import numpy as np
from openfdcm import DefaultMatch, DefaultSearch, DefaultOptimize, search

def make_rotation(line_angle):
    sin = np.sin(line_angle)
    cos = np.cos(line_angle)
    return np.array([[cos, -sin], [sin, cos]])

def create_lines(line_number, length):
    line_array = np.zeros((4, line_number))
    for i in range(line_number):
        line_angle = float(i) / float(line_number) * np.pi - np.pi / 2
        rotation_matrix = make_rotation(line_angle)
        endpoint = np.dot(rotation_matrix, np.array([length, 0]))
        line_array[:, i] = np.array([0, 0, endpoint[0], endpoint[1]])
    return line_array

def all_close(a, b, atol=1e-5):
    return np.allclose(a, b, atol=atol)

def test_default_match():
    max_tmpl_lines, max_scene_lines = 4, 4
    depth = 4
    scene_ratio = 1.0
    scene_padding = 2.2
    coeff = 50.0
    search_strategy = DefaultSearch(max_tmpl_lines, max_scene_lines)
    optimizer_strategy = DefaultOptimize()
    matcher = DefaultMatch(depth, coeff, scene_ratio, scene_padding)
    tmpl = create_lines(4, 10)

    rotation = np.array([[-1, 0], [0, -1]])
    templates = [np.dot(rotation, tmpl.reshape(2, -1)).reshape(4, -1)]

    scene = tmpl
    matches = search(matcher, search_strategy, optimizer_strategy, templates, scene)
    result_rotation = matches[0].transform[0:2, 0:2]
    result_translation = matches[0].transform[0:2, 2]

    print("Matches:\n", matches)

    assert len(matches) == max_tmpl_lines * max_scene_lines * 2
    assert all_close(result_rotation, rotation)
    assert all_close(result_translation, np.array([0, 0]))

    translation = np.array([[5], [-5]])
    templates = [(tmpl.reshape(2,-1) + translation).reshape(4,-1)]
    matches = search(matcher, search_strategy, optimizer_strategy, templates, scene)
    result_rotation = matches[0].transform[0:2, 0:2]
    result_translation = matches[0].transform[0:2, 2]

    assert len(matches) == max_tmpl_lines * max_scene_lines * 2
    assert all_close(result_rotation, np.eye(2))
    assert all_close(result_translation, -translation)
    assert matches[0].score == 0.0

    templates = [tmpl]
    scene = np.zeros((4, 0))
    matches = search(matcher, search_strategy, optimizer_strategy, templates, scene)
    assert len(matches) == 0

    templates = []
    scene = tmpl
    matches = search(matcher, search_strategy, optimizer_strategy, templates, scene)
    assert len(matches) == 0

    templates = [np.zeros((4, 0))]
    matches = search(matcher, search_strategy, optimizer_strategy, templates, scene)
    assert len(matches) == 0

def test_scale_down_scene():
    max_tmpl_lines, max_scene_lines = 4, 4
    depth = 4
    scene_ratio = 0.3
    scene_padding = 2.2
    coeff = 5.0
    matcher = DefaultMatch(depth, coeff, scene_ratio, scene_padding)
    search_strategy = DefaultSearch(max_tmpl_lines, max_scene_lines)
    optimizer_strategy = DefaultOptimize()
    tmpl = create_lines(4, 10)

    rotation = np.array([[-1, 0], [0, -1]])
    templates = [np.dot(rotation, tmpl.reshape(2, -1)).reshape(4, -1).astype(np.float64)]
    scene = tmpl
    matches = search(matcher, search_strategy, optimizer_strategy, templates, scene.astype(np.float64))
    result_rotation = matches[0].transform[0:2, 0:2]
    result_translation = matches[0].transform[0:2, 2]

    assert len(matches) == max_tmpl_lines * max_scene_lines * 2
    assert all_close(result_rotation, rotation)
    assert all_close(result_translation, np.array([0, 0]))
    assert matches[0].score < 2.4

if __name__ == "__main__":
    pytest.main()
