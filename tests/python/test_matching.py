import pytest
import numpy as np
from openfdcm import DefaultMatch, DefaultSearch, DefaultOptimize, search

def make_rotation(line_angle):
    sin = np.sin(line_angle)
    cos = np.cos(line_angle)
    return np.array([[cos, -sin], [sin, cos]])

def create_lines(line_number, length):
    line_array = np.zeros((4, line_number))
    for i, line_angle in enumerate(np.logspace(np.log10(2 * np.pi), np.log10(4 * np.pi), line_number)):
        rotation_matrix = make_rotation(line_angle)
        endpoint = np.matmul(rotation_matrix, np.array([length, 0]))
        line_array[:, i] = np.array([0, 0, endpoint[0], endpoint[1]])
    return line_array

def all_close(a, b, atol=1e-5):
    return np.allclose(a, b, atol=atol)

def apply_transform(lines, transform):
    return (np.matmul(transform[:2, :2], lines.reshape(2, -1)) + transform[:2, 2:3]).reshape(4, -1)

def run_test(scene_ratio):
    max_tmpl_lines, max_scene_lines = 4, 10
    depth = 30
    scene_padding = 2.2
    coeff = 5.0
    search_strategy = DefaultSearch(max_tmpl_lines, max_scene_lines)
    optimizer_strategy = DefaultOptimize()
    matcher = DefaultMatch(depth, coeff, scene_ratio, scene_padding)
    number_of_lines = 10
    line_length = 100
    tmpl = create_lines(number_of_lines, line_length)

    scene_transform = np.array([[-1, 0, line_length], [0, -1, line_length]])
    scene = apply_transform(tmpl, scene_transform)
    matches = search(matcher, search_strategy, optimizer_strategy, [tmpl], scene)
    best_match_transform = matches[0].transform
    best_match_rotation = best_match_transform[:2, :2]
    best_match_translation = best_match_transform[:2, 2]

    assert len(matches) == min(max_tmpl_lines, number_of_lines) * min(number_of_lines, max_scene_lines) * 2
    assert all_close(scene_transform[:2, :2], best_match_rotation)
    assert all_close(scene_transform[:2, 2], best_match_translation, 1e0 * 1 / scene_ratio)

    scene_transform = np.array([[1, 0, 0], [0, 1, 0]])
    scene = apply_transform(tmpl, scene_transform)
    matches = search(matcher, search_strategy, optimizer_strategy, [tmpl], scene)
    best_match_rotation = matches[0].transform[:2, :2]
    best_match_translation = matches[0].transform[:2, 2]

    assert len(matches) == max_tmpl_lines * max_scene_lines * 2
    assert all_close(scene_transform[:2, :2], best_match_rotation)
    assert all_close(scene_transform[:2, 2], best_match_translation, 1e0 * 1 / scene_ratio)

    scene = np.zeros((4, 0))
    matches = search(matcher, search_strategy, optimizer_strategy, [tmpl], scene)
    assert len(matches) == 0

    templates = []
    scene = tmpl
    matches = search(matcher, search_strategy, optimizer_strategy, [], scene)
    assert len(matches) == 0

    templates = [np.zeros((4, 0))]
    matches = search(matcher, search_strategy, optimizer_strategy, templates, scene)
    assert len(matches) == 0

@pytest.mark.parametrize("scene_ratio", [1.0, 0.3])
def test_matching(scene_ratio):
    run_test(scene_ratio)

if __name__ == "__main__":
    pytest.main()
